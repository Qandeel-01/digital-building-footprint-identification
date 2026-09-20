"""Phase 3 GSD sweep on FULLY-COVERED mosaicked tiles (not chip-scale).

The Phase 1 / Phase 2 sweep ran on a single SpaceNet chip whose diagnostic
tile was 73 % nodata (see docs/phase1-diagnosis.md). Comparing zoom
levels on a mostly-empty tile compares padding ratios, not resolution.
This script redoes the sweep on 20 tiles sampled from the mosaic
(which by construction have 0 % nodata) and reports:

    * pred polygon count vs reference polygon count per tile
      (count ratio, median + IQR).
    * probability-mask statistics: mean, and fraction of pixels in
      the [0.3, 0.7] uncertain band.
    * median predicted polygon area vs median reference polygon area.
    * a visual panel per zoom for the MEDIAN tile.

Only the ADAPTER GSD changes between runs; everything else (model
checkpoint, filter thresholds, reference set) is fixed. The reference
polygons come from bfbench prepare's merged + deduped set, subset by
centroid inside each tile's geographic footprint (equal-area CRS for
containment; see docs/reference-data.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Bootstrap GDAL / PROJ before rasterio import.
if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))
    _lib_bin = _prefix / "Library" / "bin"
    if _lib_bin.is_dir() and str(_lib_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_lib_bin) + os.pathsep + os.environ.get("PATH", "")

import geopandas as gpd
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import rasterio
import shapely
import yaml
from PIL import Image
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

from bfbench.data.adapter import (
    WEBMERCATOR_EPSG,
    tile_bounds_webmercator,
    webmercator_gsd_at,
)
from bfbench.run_prepared import _resolve_bin

REPO = Path(__file__).resolve().parent.parent
_LOG = logging.getLogger("bfbench.gsd_sweep")

TILE_SIZE = 512
EA_EPSG = 6933


def _load_cfg(cfg_path: Path) -> dict:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _neighbours(t: mercantile.Tile) -> list[mercantile.Tile]:
    return [mercantile.Tile(t.x + dx, t.y + dy, t.z)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def _write_tile_from_mosaic(vrt_path: Path, tile: mercantile.Tile,
                             out_dir: Path) -> Path | None:
    """Reproject the mosaic window for one XYZ tile to WebMerc at
    TILE_SIZE resolution, stretch 16-bit -> 8-bit, and write a PNG."""
    w, s, e, n = tile_bounds_webmercator(tile)
    dst_transform = rasterio.transform.from_bounds(w, s, e, n, TILE_SIZE, TILE_SIZE)
    src_bands = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    with rasterio.open(vrt_path) as src:
        for i in range(3):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=src_bands[i],
                dst_transform=dst_transform,
                dst_crs=f"EPSG:{WEBMERCATOR_EPSG}",
                dst_nodata=0,
                resampling=Resampling.cubic,
            )
    out = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    for i in range(3):
        band = src_bands[i]
        valid = band[band > 0]
        if valid.size == 0:
            out[i] = 0
            continue
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo:
            hi = lo + 1
        out[i] = np.clip((band - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if not (out != 0).any():
        return None
    tile_dir = out_dir / str(tile.z) / str(tile.x)
    tile_dir.mkdir(parents=True, exist_ok=True)
    path = tile_dir / f"{tile.y}.png"
    Image.fromarray(np.transpose(out, (1, 2, 0)), mode="RGB").save(path)
    return path


def _write_tree_and_cover(vrt: Path, core_tiles: list[mercantile.Tile],
                           out_root: Path, channel_name: str) -> Path:
    channel_root = out_root / channel_name
    channel_root.mkdir(parents=True, exist_ok=True)
    all_tiles: dict[tuple[int, int, int], mercantile.Tile] = {}
    for t in core_tiles:
        all_tiles[(t.z, t.x, t.y)] = t
        for n in _neighbours(t):
            all_tiles.setdefault((n.z, n.x, n.y), n)
    for t in all_tiles.values():
        _write_tile_from_mosaic(vrt, t, channel_root)
    cover_path = out_root / "cover.csv"
    with cover_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        for t in core_tiles:
            w.writerow([t.x, t.y, t.z])
    return cover_path


def _run_subprocess(argv: list[str]) -> None:
    _LOG.info("+ %s", " ".join(str(a) for a in argv))
    r = subprocess.run(argv, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(str(a) for a in argv)}")


def _predict_vectorize_filter(zoom_dir: Path, checkpoint: Path,
                              abd_config: Path) -> Path:
    tiles_root = zoom_dir / "tiles"
    masks_dir  = zoom_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    prob_dir   = zoom_dir / "probs"
    prob_dir.mkdir(parents=True, exist_ok=True)
    predict_local = REPO / "scripts" / "predict_local.py"
    _run_subprocess([
        sys.executable, str(predict_local),
        "--dataset", str(tiles_root),
        "--checkpoint", str(checkpoint),
        "--config", str(abd_config),
        "--cover", str(tiles_root / "cover.csv"),
        "--out", str(masks_dir),
        "--metatiles", "--keep_borders",
    ])
    raw = zoom_dir / "predicted_raw.geojson"
    _run_subprocess([_resolve_bin("abd"), "vectorize",
                     "--masks", str(masks_dir), "--type", "Building",
                     "--config", str(abd_config), "--out", str(raw)])
    final = zoom_dir / "predicted.geojson"
    _run_subprocess([_resolve_bin("filter-buildings"),
                     "--data", str(raw), "--dest", str(final)])
    return final


def _pixels_in_uncertain_band(mask_paths: list[Path]) -> dict[str, float]:
    """The upstream tile writer rounds probabilities to 0/1 before saving;
    we can't recover the raw sigmoid from the PNGs. What we CAN measure
    is the fraction of tiles that are pure background (mean = 0), pure
    foreground (mean = 1), and mixed -- as a proxy for how confident
    the model was.

    Return dict with keys:
        mean_positive_frac       -- average across tiles of (positive pixels / all pixels)
        pure_bg_tile_frac        -- fraction of tiles that are entirely background
        pure_fg_tile_frac        -- fraction of tiles entirely foreground
        mixed_tile_frac          -- 1 - pure_bg - pure_fg
    """
    fracs = []
    pure_bg = 0
    pure_fg = 0
    for p in mask_paths:
        arr = np.asarray(Image.open(p).convert("L"))
        pos = (arr != 0).sum()
        n = arr.size
        f = pos / n
        fracs.append(f)
        if f == 0.0:
            pure_bg += 1
        elif f == 1.0:
            pure_fg += 1
    n_tiles = len(fracs)
    return {
        "mean_positive_frac": float(np.mean(fracs)) if fracs else float("nan"),
        "pure_bg_tile_frac":  pure_bg / n_tiles if n_tiles else float("nan"),
        "pure_fg_tile_frac":  pure_fg / n_tiles if n_tiles else float("nan"),
        "mixed_tile_frac":    (n_tiles - pure_bg - pure_fg) / n_tiles if n_tiles else float("nan"),
        "n_tiles":            n_tiles,
    }


def _per_tile_stats(pred_geojson: Path, ref_gdf_ea: gpd.GeoDataFrame,
                    z17_footprints: list[tuple[mercantile.Tile, shapely.geometry.base.BaseGeometry]]
                    ) -> list[dict]:
    """For each of the z=17 footprints, count predicted / reference polys
    whose centroid falls inside. Returns one dict per footprint.
    """
    pred = gpd.read_file(pred_geojson) if pred_geojson.is_file() else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if pred.crs is None:
        pred = pred.set_crs("EPSG:4326")
    pred_ea = pred.to_crs(f"EPSG:{EA_EPSG}")
    pred_cent = pred_ea.geometry.centroid
    ref_cent  = ref_gdf_ea.geometry.centroid

    stats = []
    for tile, poly_ea in z17_footprints:
        p_mask = pred_cent.within(poly_ea)
        r_mask = ref_cent.within(poly_ea)
        p_in = pred_ea[p_mask]
        r_in = ref_gdf_ea[r_mask]
        stats.append({
            "tile_z": tile.z, "tile_x": tile.x, "tile_y": tile.y,
            "n_pred": int(len(p_in)),
            "n_ref":  int(len(r_in)),
            "median_pred_area_m2": float(p_in.geometry.area.median()) if len(p_in) else float("nan"),
            "median_ref_area_m2":  float(r_in.geometry.area.median()) if len(r_in) else float("nan"),
        })
    return stats


def _panel_figure(z17_tiles: list[mercantile.Tile],
                  ref_gdf_wgs: gpd.GeoDataFrame,
                  results_per_zoom: dict[int, dict],
                  out_root: Path, median_idx: int) -> Path:
    """3-column figure: one row per zoom, showing imagery / prediction /
    reference for the MEDIAN tile (by count ratio) at each zoom."""
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12),
                             constrained_layout=True)
    median_tile = z17_tiles[median_idx]
    b = mercantile.bounds(median_tile)
    xlim = (b.west, b.east); ylim = (b.south, b.north)
    ref_in_view = ref_gdf_wgs.cx[b.west:b.east, b.south:b.north]
    for row, z in enumerate((16, 17, 18)):
        r = results_per_zoom[z]
        tiles_root = out_root / f"z{z}" / "tiles" / "images"
        # Imagery panel: read the median z=17 tile's neighbourhood at
        # this zoom. Simpler: just pick the tile at zoom z whose
        # footprint contains the median z=17 tile's centroid.
        target_ll = ((b.west + b.east) / 2.0, (b.south + b.north) / 2.0)
        tile_z = mercantile.tile(target_ll[0], target_ll[1], z)
        img_path = tiles_root / str(z) / str(tile_z.x) / f"{tile_z.y}.png"
        ax = axes[row, 0]
        ax.set_title(f"z={z}  imagery")
        if img_path.is_file():
            img = np.asarray(Image.open(img_path))
            tb = mercantile.bounds(tile_z)
            ax.imshow(img, extent=(tb.west, tb.east, tb.south, tb.north),
                      origin="upper")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        # Predictions
        pred_path = out_root / f"z{z}" / "predicted.geojson"
        pgdf = (gpd.read_file(pred_path).to_crs("EPSG:4326")
                if pred_path.is_file() and pred_path.stat().st_size > 100
                else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        ax = axes[row, 1]
        ax.set_title(f"z={z}  prediction ({len(pgdf)} polys total)")
        if img_path.is_file():
            ax.imshow(img, extent=(tb.west, tb.east, tb.south, tb.north),
                      origin="upper", alpha=0.5)
        pgdf.plot(ax=ax, facecolor=(1, 0.08, 0.58, 0.4),
                  edgecolor="deeppink", linewidth=1.0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        # Reference (same in all rows; drawn per zoom's imagery)
        ax = axes[row, 2]
        ax.set_title(f"z={z}  reference ({len(ref_in_view)} polys in view)")
        if img_path.is_file():
            ax.imshow(img, extent=(tb.west, tb.east, tb.south, tb.north),
                      origin="upper", alpha=0.5)
        ref_in_view.plot(ax=ax, facecolor=(0, 1, 1, 0.35),
                         edgecolor="cyan", linewidth=1.0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"GSD sweep -- median tile z=17 ({median_tile.x}, {median_tile.y})",
                 y=1.01)
    out = out_root / "gsd_sweep_panel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=REPO / "configs" / "spacenet_vegas.yaml")
    p.add_argument("--n-tiles", type=int, default=20,
                   help="Number of z=17 tiles to include in the sweep.")
    p.add_argument("--out", type=Path,
                   default=REPO / "runs" / "gsd_sweep_v2")
    args = p.parse_args(argv)

    cfg = _load_cfg(args.config)
    work_dir = Path(cfg["run"]["work_dir"])
    args.out.mkdir(parents=True, exist_ok=True)

    payload = json.loads((work_dir / "sampled_tiles.json").read_text(encoding="utf-8"))
    tiles_z17 = [mercantile.Tile(x, y, z) for z, x, y in payload["tiles"]]
    tiles_z17 = tiles_z17[: args.n_tiles]
    _LOG.info("using %d z=17 tiles", len(tiles_z17))

    # Parent (z=16) and children (z=18) sets
    tiles_z16: list[mercantile.Tile] = []
    tiles_z18: list[mercantile.Tile] = []
    seen16, seen18 = set(), set()
    for t in tiles_z17:
        p16 = mercantile.parent(t)
        if (p16.z, p16.x, p16.y) not in seen16:
            tiles_z16.append(p16); seen16.add((p16.z, p16.x, p16.y))
        for c in mercantile.children(t):
            if (c.z, c.x, c.y) not in seen18:
                tiles_z18.append(c); seen18.add((c.z, c.x, c.y))
    _LOG.info("z=16 parents=%d, z=18 children=%d", len(tiles_z16), len(tiles_z18))

    zoom_tiles = {16: tiles_z16, 17: tiles_z17, 18: tiles_z18}
    vrt = work_dir / "mosaic.vrt"
    checkpoint = Path(cfg["abd"]["checkpoint"])
    abd_config = Path(cfg["abd"]["config"])

    # z=17 tile footprints in equal-area for per-tile counting
    ea_crs = f"EPSG:{EA_EPSG}"
    footprints: list[tuple[mercantile.Tile, shapely.geometry.base.BaseGeometry]] = []
    for t in tiles_z17:
        bb = mercantile.bounds(t)
        gs = gpd.GeoSeries([box(bb.west, bb.south, bb.east, bb.north)],
                           crs="EPSG:4326").to_crs(ea_crs)
        footprints.append((t, gs.iloc[0]))

    # Reference set (already deduped by prepare)
    ref_wgs = gpd.read_file(work_dir / "reference.geojson")
    if ref_wgs.crs is None:
        ref_wgs = ref_wgs.set_crs("EPSG:4326")
    ref_ea = ref_wgs.to_crs(ea_crs)

    results_per_zoom: dict[int, dict] = {}
    per_tile_records: list[dict] = []
    for z in (16, 17, 18):
        zoom_dir = args.out / f"z{z}"
        (zoom_dir / "tiles").mkdir(parents=True, exist_ok=True)
        cover = _write_tree_and_cover(vrt, zoom_tiles[z], zoom_dir / "tiles",
                                      channel_name="images")
        _LOG.info("z=%d: wrote %d core tiles (+ neighbours), cover=%s",
                  z, len(zoom_tiles[z]), cover)
        pred_geojson = _predict_vectorize_filter(zoom_dir, checkpoint, abd_config)

        # Mask uncertainty proxy
        mask_paths = list((zoom_dir / "masks" / "images" / str(z)).rglob("*.png"))
        prob_stats = _pixels_in_uncertain_band(mask_paths)

        stats = _per_tile_stats(pred_geojson, ref_ea, footprints)
        for r in stats:
            r["zoom"] = z
            r["effective_gsd_m"] = webmercator_gsd_at(
                0.5 * (mercantile.bounds(tiles_z17[0]).south +
                       mercantile.bounds(tiles_z17[0]).north),
                z, tile_size=TILE_SIZE)
        per_tile_records.extend(stats)
        results_per_zoom[z] = {
            "prob_stats": prob_stats,
            "n_tiles": len(stats),
        }

    # Aggregate per zoom
    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(per_tile_records)
    df["count_ratio"] = df["n_pred"] / df["n_ref"].replace(0, np.nan)
    summary = {}
    for z in (16, 17, 18):
        sub = df[df["zoom"] == z]
        summary[str(z)] = {
            "effective_gsd_m": float(sub["effective_gsd_m"].iloc[0]) if len(sub) else float("nan"),
            "count_ratio_median": float(sub["count_ratio"].median()),
            "count_ratio_iqr":    [float(sub["count_ratio"].quantile(0.25)),
                                   float(sub["count_ratio"].quantile(0.75))],
            "median_pred_area_m2": float(sub["median_pred_area_m2"].median()),
            "median_ref_area_m2":  float(sub["median_ref_area_m2"].median()),
            "n_pred_median": float(sub["n_pred"].median()),
            "n_ref_median":  float(sub["n_ref"].median()),
            "prob_mask": results_per_zoom[z]["prob_stats"],
        }

    (args.out / "results.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")
    df.to_csv(args.out / "per_tile.csv", index=False)
    _LOG.info("wrote %s", args.out / "results.json")

    # Panel: median tile by z=17 count ratio
    sub17 = df[df["zoom"] == 17].reset_index(drop=True)
    order = sub17["count_ratio"].fillna(0.0).sort_values().index.tolist()
    median_idx = order[len(order) // 2]
    _panel_figure(tiles_z17, ref_wgs, results_per_zoom, args.out, median_idx)


if __name__ == "__main__":
    main()
