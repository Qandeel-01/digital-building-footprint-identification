"""bfbench evaluation harness.

Runs the metric stubs from `bfbench.metrics` over the sampled tiles
produced by `bfbench prepare`, and writes a tidy long-format Parquet
file with one row per (aoi, tile, metric).

Contract:

    * The harness runs end-to-end even when the metric functions are
      stubs raising NotImplementedError. In that case the corresponding
      rows have `value = NaN` and `status = "not_implemented"`. The
      moment a metric is implemented the same run produces real values
      without any other code changing.

    * The harness is deterministic: it reads the sampled tile list
      produced by prepare (with its recorded seed and hash), and
      records the same seed + hash in the eval manifest.

    * The harness is resumable: per-tile results are staged to
      `<work_dir>/eval/tiles/<z>_<x>_<y>.json` and the final Parquet
      is a concatenation. Re-running skips tiles whose staged file
      already exists.

    * Every area/distance/density is computed in an equal-area CRS,
      enforced by the metric-side assertions (see bfbench.data.equal_area).

Prerequisites (produced by `bfbench prepare` + `bfbench run`):

    <work_dir>/sampled_tiles.json      seed + tile IDs (from prepare)
    <work_dir>/reference.geojson       merged, deduped ref polygons
    <work_dir>/predicted.geojson       ONE geojson from filter-buildings
                                       (produced by `bfbench run`)

If `predicted.geojson` is absent, the harness raises early with a
message telling the caller to run the prediction stage first. That
keeps the eval independent of upstream `abd`'s NCCL / Windows issues.
"""

from __future__ import annotations

import json
import logging
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import mercantile
import pandas as pd
import yaml
from shapely.geometry import box

from bfbench.data.equal_area import DEFAULT_EQUAL_AREA_EPSG
from bfbench.metrics import (
    boundary_distance,
    count_error,
    instance_prf,
    match_instances,
    merge_rate,
    pixel_iou,
    split_rate,
)

_LOG = logging.getLogger("bfbench.eval")


@dataclass(frozen=True)
class EvalConfig:
    aoi_name: str
    work_dir: Path
    equal_area_epsg: int
    iou_threshold: float
    pixel_resolution_m: float


def _load(path: Path) -> EvalConfig:
    with path.open("r", encoding="utf-8") as fp:
        y = yaml.safe_load(fp)
    aoi = y["aoi"]
    ref = y["reference"]["dedup"]
    ev  = y.get("eval", {})
    return EvalConfig(
        aoi_name=aoi["name"],
        work_dir=Path(y["run"]["work_dir"]),
        equal_area_epsg=int(ref["equal_area_epsg"]),
        iou_threshold=float(ev.get("iou_threshold", 0.5)),
        pixel_resolution_m=float(ev.get("pixel_resolution_m", 0.5)),
    )


def _tile_bounds_wgs84(tile: mercantile.Tile) -> tuple[float, float, float, float]:
    """(west, south, east, north) lon/lat bounds for one tile."""
    b = mercantile.bounds(tile)
    return (b.west, b.south, b.east, b.north)


def _clip_by_centroid(gdf: gpd.GeoDataFrame,
                      tile_polygon_ea) -> gpd.GeoDataFrame:
    """Return the subset of `gdf` whose centroid lies inside `tile_polygon_ea`.

    Centroid-in-polygon is the correct rule for the tile-scope numerator
    in phase-2 metrics (see docs/phase1-diagnosis.md finding 1).
    Assumes gdf is already in the equal-area CRS.
    """
    mask = gdf.geometry.centroid.within(tile_polygon_ea)
    return gdf[mask].reset_index(drop=True)


def _tile_footprint_ea(tile: mercantile.Tile,
                       equal_area_epsg: int) -> Any:
    """Return the tile bbox as a shapely polygon in the equal-area CRS.

    Reprojects the WGS84 tile bounds through a one-off GeoDataFrame so
    the polygon uses the same projection pipeline that pyproj/geopandas
    apply everywhere else (avoids drift from ad-hoc pyproj Transformer
    calls).
    """
    w, s, e, n = _tile_bounds_wgs84(tile)
    box_gdf = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326")
    return box_gdf.to_crs(f"EPSG:{equal_area_epsg}").geometry.iloc[0]


def _safe_call(fn, *args, metric_name: str, **kwargs) -> tuple[Any, str, str]:
    """Invoke a metric function; return (value, status, note).

    - ok / value                -> (value, "ok", "")
    - NotImplementedError       -> (NaN or {}, "not_implemented", exc msg)
    - any other exception       -> (NaN or {}, "error", traceback)

    The harness does not swallow errors silently: unknown exceptions
    are recorded so a subtle bug (bad CRS, empty GeoDataFrame edge
    case) is visible in the parquet.
    """
    try:
        val = fn(*args, **kwargs)
        return val, "ok", ""
    except NotImplementedError as e:
        return float("nan"), "not_implemented", f"{metric_name}: {e!s}"
    except Exception as e:  # noqa: BLE001
        return float("nan"), "error", f"{metric_name}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _rows_for_dict(base: dict, prefix: str, val: Any, status: str, note: str) -> list[dict]:
    """Expand a dict-returning metric into one row per key."""
    if isinstance(val, dict):
        out = []
        for k, v in val.items():
            row = dict(base)
            row["metric"] = f"{prefix}.{k}"
            row["value"] = float("nan") if v is None else float(v) if not isinstance(v, (int | float)) else float(v)
            row["status"] = status
            row["note"] = note
            out.append(row)
        return out
    row = dict(base)
    row["metric"] = prefix
    row["value"] = float("nan") if val is None else (float(val) if isinstance(val, (int | float)) else float("nan"))
    row["status"] = status
    row["note"] = note
    return [row]


def _evaluate_tile(tile: mercantile.Tile,
                   pred_gdf_ea: gpd.GeoDataFrame,
                   ref_gdf_ea:  gpd.GeoDataFrame,
                   aoi_name: str,
                   cfg: EvalConfig) -> list[dict]:
    """Compute every metric on one tile. Returns a list of tidy rows."""
    tile_poly_ea = _tile_footprint_ea(tile, cfg.equal_area_epsg)
    pred_tile = _clip_by_centroid(pred_gdf_ea, tile_poly_ea)
    ref_tile  = _clip_by_centroid(ref_gdf_ea,  tile_poly_ea)

    base = {
        "aoi": aoi_name,
        "tile_z": tile.z, "tile_x": tile.x, "tile_y": tile.y,
        "n_pred": len(pred_tile),
        "n_ref":  len(ref_tile),
    }

    rows: list[dict] = []

    match_val, match_status, match_note = _safe_call(
        match_instances, pred_tile, ref_tile, cfg.iou_threshold,
        metric_name="match_instances",
    )
    # `match_instances` returns a MatchResult (not a scalar); we record
    # only the status here. Downstream metrics that consume MatchResult
    # use the successful result if available, else a fallback empty one
    # so the pipeline still exercises them and reports their own status.
    row = dict(base)
    row["metric"] = "match_instances.status"
    row["value"] = float("nan")
    row["status"] = match_status
    row["note"] = match_note
    rows.append(row)

    # Fall through with a placeholder MatchResult so downstream metrics
    # still run and report their own status (this makes NotImplementedError
    # visible for e.g. instance_prf even when match_instances is also
    # unimplemented).
    from bfbench.metrics.instance import MatchResult  # noqa: PLC0415
    mr = match_val if match_status == "ok" else MatchResult()

    prf_val, prf_status, prf_note = _safe_call(
        instance_prf, mr, metric_name="instance_prf",
    )
    rows += _rows_for_dict(base, "instance_prf", prf_val, prf_status, prf_note)

    for name, fn in (("merge_rate", merge_rate),
                     ("split_rate", split_rate)):
        v, s, note = _safe_call(fn, mr, metric_name=name)
        row = dict(base)
        row["metric"] = name
        row["value"] = float("nan") if not isinstance(v, (int | float)) else float(v)
        row["status"] = s
        row["note"] = note
        rows.append(row)

    ce_val, ce_status, ce_note = _safe_call(
        count_error, pred_tile, ref_tile, metric_name="count_error",
    )
    rows += _rows_for_dict(base, "count_error", ce_val, ce_status, ce_note)

    b = tile_poly_ea.bounds
    piv, piv_status, piv_note = _safe_call(
        pixel_iou, pred_tile, ref_tile,
        bounds=b, resolution=cfg.pixel_resolution_m,
        metric_name="pixel_iou",
    )
    row = dict(base)
    row["metric"] = "pixel_iou"
    row["value"] = float("nan") if not isinstance(piv, (int | float)) else float(piv)
    row["status"] = piv_status
    row["note"] = piv_note
    rows.append(row)

    bd_val, bd_status, bd_note = _safe_call(
        boundary_distance, mr, metric_name="boundary_distance",
    )
    rows += _rows_for_dict(base, "boundary_distance", bd_val, bd_status, bd_note)

    # Numerical hygiene: NaN is fine, but +inf or -inf turns downstream
    # aggregates into all-NaN. Never smuggle those into the parquet.
    for r in rows:
        if isinstance(r["value"], float) and math.isinf(r["value"]):
            r["value"] = float("nan")
            r["status"] = "error"
            r["note"] = (r.get("note", "") + " [infinity replaced with NaN]").strip()

    return rows


def _read_sampled_tiles(work_dir: Path) -> tuple[list[mercantile.Tile], dict]:
    sp = work_dir / "sampled_tiles.json"
    if not sp.is_file():
        raise FileNotFoundError(
            f"sampled_tiles.json missing at {sp}. Run "
            "`bfbench prepare <config.yaml>` first."
        )
    payload = json.loads(sp.read_text(encoding="utf-8"))
    tiles = [mercantile.Tile(x, y, z) for z, x, y in payload["tiles"]]
    return tiles, payload


def _read_pred_gdf(work_dir: Path, ea_crs: str) -> gpd.GeoDataFrame:
    p = work_dir / "predicted.geojson"
    if not p.is_file():
        raise FileNotFoundError(
            f"predicted.geojson missing at {p}. Run `bfbench run "
            "<config.yaml>` (predict + vectorize + filter) first, or "
            "invoke `bfbench eval --synthetic-predictions` for a smoke run."
        )
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    return g.to_crs(ea_crs)


def _read_ref_gdf(work_dir: Path, ea_crs: str) -> gpd.GeoDataFrame:
    r = work_dir / "reference.geojson"
    if not r.is_file():
        raise FileNotFoundError(
            f"reference.geojson missing at {r}. Run `bfbench prepare "
            "<config.yaml>` first."
        )
    g = gpd.read_file(r)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    return g.to_crs(ea_crs)


def _synth_predictions(ref_gdf_ea: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cheap smoke-mode predictions: duplicate the reference but nudged
    a few metres to the east. Guarantees exercise of match_instances +
    downstream code without needing a real inference run.
    """
    shifted = ref_gdf_ea.copy()
    shifted["geometry"] = shifted.geometry.translate(xoff=2.0, yoff=0.0)
    return shifted


def evaluate(cfg_path: Path,
             *,
             synthetic_predictions: bool = False) -> Path:
    """Run the harness. Returns the path to the tidy Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = _load(cfg_path)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    ea_crs = f"EPSG:{cfg.equal_area_epsg}"

    tiles, sample_payload = _read_sampled_tiles(cfg.work_dir)
    ref_gdf_ea = _read_ref_gdf(cfg.work_dir, ea_crs)
    pred_gdf_ea = (
        _synth_predictions(ref_gdf_ea) if synthetic_predictions
        else _read_pred_gdf(cfg.work_dir, ea_crs)
    )

    stage_dir = cfg.work_dir / "eval" / "tiles"
    stage_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info("eval aoi=%s tiles=%d pred=%d ref=%d",
              cfg.aoi_name, len(tiles), len(pred_gdf_ea), len(ref_gdf_ea))

    all_rows: list[dict] = []
    for i, tile in enumerate(tiles):
        stage = stage_dir / f"{tile.z}_{tile.x}_{tile.y}.json"
        if stage.exists():
            rows = json.loads(stage.read_text(encoding="utf-8"))
        else:
            rows = _evaluate_tile(tile, pred_gdf_ea, ref_gdf_ea,
                                  cfg.aoi_name, cfg)
            stage.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        all_rows.extend(rows)
        if (i + 1) % 25 == 0:
            _LOG.info("  eval progress: %d/%d tiles", i + 1, len(tiles))

    df = pd.DataFrame(all_rows)
    out = cfg.work_dir / "eval" / "results.parquet"
    df.to_parquet(out, index=False)

    manifest = {
        "config_source": str(cfg_path.resolve()),
        "aoi": cfg.aoi_name,
        "equal_area_epsg": cfg.equal_area_epsg,
        "iou_threshold": cfg.iou_threshold,
        "pixel_resolution_m": cfg.pixel_resolution_m,
        "sample_seed": sample_payload["seed"],
        "sample_hash": sample_payload["tile_ids_hash"],
        "n_tiles": len(tiles),
        "synthetic_predictions": bool(synthetic_predictions),
        "n_rows": len(all_rows),
        "n_metrics_ok": int((df["status"] == "ok").sum()) if len(df) else 0,
        "n_metrics_not_implemented": int((df["status"] == "not_implemented").sum())
            if len(df) else 0,
        "n_metrics_error": int((df["status"] == "error").sum()) if len(df) else 0,
    }
    (cfg.work_dir / "eval" / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    _LOG.info("wrote %s (%d rows: ok=%d, not_implemented=%d, error=%d)",
              out, manifest["n_rows"],
              manifest["n_metrics_ok"],
              manifest["n_metrics_not_implemented"],
              manifest["n_metrics_error"])
    return out


__all__ = ["EvalConfig", "evaluate", "DEFAULT_EQUAL_AREA_EPSG"]
