"""bfbench evaluation harness.

Runs the metric functions in `bfbench.metrics` over the sampled tiles
produced by `bfbench prepare`, and writes a tidy long-format Parquet
file with one row per (aoi, tile, metric).

Metrics implemented in Phase 3:
    pixel_iou, count_error   -- populated with real values
    per-tile enrichment      -- median_ref_area_m2, ref_density_per_ha
                                at window_m in {100, 200, 500}

Metrics reserved (still stubs):
    match_instances, instance_prf, merge_rate, split_rate,
    boundary_distance       -- rows have value=NaN, status="not_implemented"

Prerequisites (produced by `bfbench prepare` + `bfbench run`):

    <work_dir>/sampled_tiles.json      seed + tile IDs (from prepare)
    <work_dir>/reference.geojson       merged, deduped ref polygons
    <work_dir>/predicted.geojson       from `bfbench run`  (vector output)

Optional (used by pixel_iou when present):
    <work_dir>/masks/<z>/<x>/<y>.png   raw prediction rasters

If per-tile masks exist, pixel_iou reprojects each into the equal-area
CRS at the tile's grid and rasterises the reference there. If not,
pixel_iou falls back to rasterising the predicted vectors -- less
accurate (roundtrip discretisation) but sufficient for a smoke run.
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
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import rasterize as _rasterize
from rasterio.transform import from_bounds as _rio_transform_from_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

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

# Windows in metres used for the density enrichment. The stratify helper
# expects a `window_m` column so downstream figures can pick one.
DENSITY_WINDOWS_M = (100.0, 200.0, 500.0)
DEFAULT_DENSITY_WINDOW_M = 200.0


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
    b = mercantile.bounds(tile)
    return (b.west, b.south, b.east, b.north)


def _clip_by_centroid(gdf: gpd.GeoDataFrame, tile_polygon_ea) -> gpd.GeoDataFrame:
    """Return subset of gdf whose centroid lies inside `tile_polygon_ea`.
    `gdf` must already be in the equal-area CRS."""
    if len(gdf) == 0:
        return gdf
    mask = gdf.geometry.centroid.within(tile_polygon_ea)
    return gdf[mask].reset_index(drop=True)


def _tile_footprint_ea(tile: mercantile.Tile, equal_area_epsg: int) -> Any:
    w, s, e, n = _tile_bounds_wgs84(tile)
    box_gdf = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326")
    return box_gdf.to_crs(f"EPSG:{equal_area_epsg}").geometry.iloc[0]


def _safe_call(fn, *args, metric_name: str, **kwargs) -> tuple[Any, str, str]:
    try:
        val = fn(*args, **kwargs)
        return val, "ok", ""
    except NotImplementedError as e:
        return None, "not_implemented", f"{metric_name}: {e!s}"
    except Exception as e:  # noqa: BLE001
        return None, "error", (
            f"{metric_name}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def _row(base: dict, metric: str, value: Any, status: str, note: str) -> dict:
    row = dict(base)
    row["metric"] = metric
    if value is None:
        row["value"] = float("nan")
    elif isinstance(value, (int | float)):
        v = float(value)
        row["value"] = float("nan") if math.isinf(v) else v
    else:
        row["value"] = float("nan")
    row["status"] = status
    row["note"] = note
    return row


def _dict_rows(base: dict, prefix: str, val: Any, status: str, note: str) -> list[dict]:
    if isinstance(val, dict):
        return [_row(base, f"{prefix}.{k}", v, status, note) for k, v in val.items()]
    return [_row(base, prefix, val, status, note)]


def _pred_mask_for_tile(cfg: EvalConfig,
                       tile: mercantile.Tile,
                       tile_poly_ea,
                       pred_gdf_ea: gpd.GeoDataFrame) -> np.ndarray:
    """Return a boolean HxW array on the equal-area grid derived from
    the tile bounds + cfg.pixel_resolution_m.

    Preference order:
        1. Raw model mask at <work_dir>/masks/<z>/<x>/<y>.png (or .tif),
           reprojected via rasterio into the equal-area grid.
        2. Rasterisation of `pred_gdf_ea` clipped to the tile footprint
           (centre-only snapping, so pixel_iou remains consistent with
           the reference rasterisation done inside the metric).
    """
    left, bottom, right, top = tile_poly_ea.bounds
    p = cfg.pixel_resolution_m
    width  = int(round((right - left) / p))
    height = int(round((top - bottom) / p))
    if width <= 0 or height <= 0:
        return np.zeros((max(1, height), max(1, width)), dtype=bool)

    mask_dir = cfg.work_dir / "masks"
    mask_paths = [
        mask_dir / str(tile.z) / str(tile.x) / f"{tile.y}.png",
        mask_dir / str(tile.z) / str(tile.x) / f"{tile.y}.tif",
    ]
    src_path = next((p for p in mask_paths if p.is_file()), None)
    if src_path is not None:
        return _reproject_mask_to_ea(src_path, cfg, left, bottom, right, top,
                                     height, width)

    # Fallback: rasterise predicted vectors clipped to the tile.
    dst_transform = _rio_transform_from_bounds(left, bottom, right, top,
                                               width, height)
    if len(pred_gdf_ea) == 0:
        return np.zeros((height, width), dtype=bool)
    pred_clip = _clip_by_centroid(pred_gdf_ea, tile_poly_ea)
    if len(pred_clip) == 0:
        return np.zeros((height, width), dtype=bool)
    shapes = [(g, 1) for g in pred_clip.geometry if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=bool)
    mask = _rasterize(shapes, out_shape=(height, width),
                      transform=dst_transform, fill=0,
                      all_touched=False, dtype="uint8")
    return mask.astype(bool)


def _reproject_mask_to_ea(src_path: Path, cfg: EvalConfig,
                          left: float, bottom: float, right: float, top: float,
                          height: int, width: int) -> np.ndarray:
    ea_crs = f"EPSG:{cfg.equal_area_epsg}"
    dst = np.zeros((height, width), dtype=np.uint8)
    dst_transform = _rio_transform_from_bounds(left, bottom, right, top,
                                               width, height)
    with rasterio.open(src_path) as src:
        src_band = src.read(1)
        # Some upstream tile writers emit a paletted PNG; treat any
        # nonzero value as "building".
        reproject(
            source=(src_band != 0).astype(np.uint8),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=ea_crs,
            resampling=Resampling.nearest,
        )
    return dst.astype(bool)


def _density_and_area_columns(tile_poly_ea, ref_gdf_ea: gpd.GeoDataFrame,
                              ref_centroids_ea: gpd.GeoSeries) -> dict[str, float]:
    """Per-tile ref-area and multi-window density columns."""
    # Reference polygons whose centroid is inside the tile:
    inside_mask = ref_centroids_ea.within(tile_poly_ea)
    inside = ref_gdf_ea[inside_mask]
    if len(inside) > 0:
        med_area = float(inside.geometry.area.median())
    else:
        med_area = float("nan")

    out = {"median_ref_area_m2": med_area}
    tc = tile_poly_ea.centroid
    for w in DENSITY_WINDOWS_M:
        window = tc.buffer(w)
        n = int(ref_centroids_ea.within(window).sum())
        area_ha = float(window.area / 1e4)
        density = n / area_ha if area_ha > 0 else float("nan")
        out[f"ref_density_per_ha_w{int(w)}"] = float(density)
    # Convenience alias for the default window used by figures:
    out["ref_density_per_ha"] = out[f"ref_density_per_ha_w{int(DEFAULT_DENSITY_WINDOW_M)}"]
    return out


def _evaluate_tile(tile: mercantile.Tile,
                   pred_gdf_ea: gpd.GeoDataFrame,
                   ref_gdf_ea:  gpd.GeoDataFrame,
                   ref_centroids_ea: gpd.GeoSeries,
                   aoi_name: str,
                   cfg: EvalConfig) -> list[dict]:
    tile_poly_ea = _tile_footprint_ea(tile, cfg.equal_area_epsg)
    pred_tile = _clip_by_centroid(pred_gdf_ea, tile_poly_ea)
    ref_tile  = _clip_by_centroid(ref_gdf_ea,  tile_poly_ea)

    enrich = _density_and_area_columns(tile_poly_ea, ref_gdf_ea, ref_centroids_ea)

    base = {
        "aoi": aoi_name,
        "tile_z": tile.z, "tile_x": tile.x, "tile_y": tile.y,
        "n_pred": len(pred_tile),
        "n_ref":  len(ref_tile),
        **enrich,
    }

    rows: list[dict] = []

    # -- reserved stubs: just record NotImplementedError
    for name, fn in (
        ("match_instances", lambda: match_instances(pred_tile, ref_tile, cfg.iou_threshold)),
    ):
        _, s, note = _safe_call(lambda: fn(), metric_name=name)  # noqa: B023
        rows.append(_row(base, f"{name}.status", None, s, note))

    from bfbench.metrics.instance import MatchResult  # noqa: PLC0415
    mr = MatchResult()
    for name, fn in (
        ("instance_prf", lambda: instance_prf(mr)),
        ("merge_rate",   lambda: merge_rate(mr)),
        ("split_rate",   lambda: split_rate(mr)),
        ("boundary_distance", lambda: boundary_distance(mr)),
    ):
        val, s, note = _safe_call(lambda: fn(), metric_name=name)  # noqa: B023
        rows += _dict_rows(base, name, val, s, note)

    # -- implemented: count_error
    ce_val, ce_s, ce_note = _safe_call(count_error, pred_tile, ref_tile,
                                       metric_name="count_error")
    rows += _dict_rows(base, "count_error", ce_val, ce_s, ce_note)

    # -- implemented: pixel_iou at both snappings (default centre-only + sensitivity)
    pred_mask = _pred_mask_for_tile(cfg, tile, tile_poly_ea, pred_gdf_ea)
    for label, all_touched in (("pixel_iou", False),
                               ("pixel_iou_all_touched", True)):
        iou_val, iou_s, iou_note = _safe_call(
            pixel_iou, pred_mask, ref_tile,
            bounds=tile_poly_ea.bounds,
            resolution=cfg.pixel_resolution_m,
            all_touched=all_touched,
            metric_name=label,
        )
        rows.append(_row(base, label, iou_val, iou_s, iou_note))

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
    """Cheap smoke-mode: duplicate the reference but shift 2 m east."""
    shifted = ref_gdf_ea.copy()
    shifted["geometry"] = shifted.geometry.translate(xoff=2.0, yoff=0.0)
    return shifted


def evaluate(cfg_path: Path,
             *, synthetic_predictions: bool = False) -> Path:
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
    ref_centroids_ea = ref_gdf_ea.geometry.centroid
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
                                  ref_centroids_ea, cfg.aoi_name, cfg)
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
        "density_windows_m": list(DENSITY_WINDOWS_M),
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
              out, manifest["n_rows"], manifest["n_metrics_ok"],
              manifest["n_metrics_not_implemented"], manifest["n_metrics_error"])
    return out


__all__ = ["EvalConfig", "evaluate", "DENSITY_WINDOWS_M"]
