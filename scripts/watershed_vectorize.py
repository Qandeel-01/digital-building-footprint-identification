"""Phase 2b: watershed instance separation + vectorisation of a
masks directory produced by `abd predict` / `scripts/predict_local.py`.

Equivalent in interface to `abd vectorize`, but each connected mask
component is first split by distance-transform watershed before being
polygonised. The output is one GeoJSON (EPSG:4326) with one polygon
per instance; drop-in replacement for `<work_dir>/predicted_raw.geojson`
in the Phase 1 / 3 pipeline.

Usage
-----
    python scripts/watershed_vectorize.py \\
        --masks <work_dir>/masks \\
        --out   <work_dir>/predicted_raw.geojson \\
        --min-marker-distance-m 5.0

Caveat
------
Upstream `tile_label_to_file` writes binary label PNGs, not
probability maps. The watershed elevation surface therefore defaults
to the Euclidean distance transform of the binary foreground, which
is the standard pattern for splitting touching objects via watershed.
If probability maps are available on disk alongside the labels (naming
convention: `<z>/<x>/<y>.prob.npy`), they are used in preference.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from rasterio.features import shapes as _shapes

if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))

import geopandas as gpd  # noqa: E402
from shapely.geometry import shape as _shape  # noqa: E402

from bfbench.data.adapter import tile_bounds_webmercator  # noqa: E402
from bfbench.postprocess.watershed import (  # noqa: E402
    WatershedConfig,
    watershed_split_mask,
)

_LOG = logging.getLogger("bfbench.watershed_vectorize")


def _iter_tile_masks(masks_root: Path) -> list[tuple[mercantile.Tile, Path, Path | None]]:
    """Yield (tile, mask_path, prob_path_or_None) for every tile under
    `masks_root`.

    Expected layout: `masks_root/<z>/<x>/<y>.png` (what upstream writes).
    Optional sibling probability map: `masks_root/<z>/<x>/<y>.prob.npy`.
    """
    out: list[tuple[mercantile.Tile, Path, Path | None]] = []
    for z_dir in sorted(masks_root.iterdir()):
        if not z_dir.is_dir() or not z_dir.name.isdigit():
            continue
        z = int(z_dir.name)
        for x_dir in sorted(z_dir.iterdir()):
            if not x_dir.is_dir() or not x_dir.name.isdigit():
                continue
            x = int(x_dir.name)
            for png in sorted(x_dir.glob("*.png")):
                stem = png.stem
                if not stem.isdigit():
                    continue
                y = int(stem)
                prob = png.with_suffix(".prob.npy")
                out.append((
                    mercantile.Tile(x=x, y=y, z=z),
                    png,
                    prob if prob.is_file() else None,
                ))
    return out


def _tile_pixel_resolution_m(tile: mercantile.Tile, mask_shape: tuple[int, int]
                             ) -> float:
    """Metres per pixel for this tile at its native Web-Mercator extent."""
    w, _s, e, _n = tile_bounds_webmercator(tile)
    width_m = e - w
    return float(width_m) / float(mask_shape[1])


def _label_polygons_wgs84(labels: np.ndarray,
                          tile: mercantile.Tile) -> list[dict]:
    """Polygonise each label independently so touching instances stay
    separated after vector export."""
    w, s, e, n = tile_bounds_webmercator(tile)
    tile_h, tile_w = labels.shape
    px_w = (e - w) / tile_w
    px_h = (n - s) / tile_h
    # rasterio.features.shapes needs an affine transform; build one from
    # the tile bounds (Web-Mercator).
    from rasterio.transform import from_bounds as _from_bounds  # noqa: PLC0415
    transform = _from_bounds(w, s, e, n, tile_w, tile_h)

    unique = np.unique(labels)
    unique = unique[unique > 0]
    rows: list[dict] = []
    for lbl in unique:
        mask = (labels == lbl).astype(np.uint8)
        for geom, _val in _shapes(mask, mask=mask > 0, transform=transform):
            rows.append({
                "tile_z": tile.z,
                "tile_x": tile.x,
                "tile_y": tile.y,
                "label":  int(lbl),
                "geometry_wm": _shape(geom),
            })
    # px_w / px_h unused except to keep the resolution handy in logs.
    _ = (px_w, px_h)
    return rows


def vectorize(masks_root: Path, out_geojson: Path,
              cfg: WatershedConfig | None = None) -> Path:
    """Run watershed + vectorise over every tile under `masks_root` and
    write a single GeoJSON (EPSG:4326) to `out_geojson`."""
    cfg = cfg or WatershedConfig()
    tiles = _iter_tile_masks(masks_root)
    _LOG.info("found %d tile masks under %s", len(tiles), masks_root)
    rows_all: list[dict] = []
    for tile, png, prob_path in tiles:
        with rasterio.open(png) as src:
            mask = src.read(1)
        prob = None
        if prob_path is not None:
            prob = np.load(prob_path)
            if prob.ndim == 3 and prob.shape[0] >= 2:
                # (classes, H, W) -> use the "building" probability
                # (last channel for binary, index 1 for 3-class).
                prob = prob[1] if prob.shape[0] >= 2 else prob[0]
        px_m = _tile_pixel_resolution_m(tile, mask.shape)
        labels = watershed_split_mask(mask, pixel_resolution_m=px_m,
                                       cfg=cfg, prob_map=prob)
        rows_all.extend(_label_polygons_wgs84(labels, tile))

    if not rows_all:
        _LOG.warning("no polygons produced; writing empty GeoJSON")
        gpd.GeoDataFrame(
            columns=["tile_z", "tile_x", "tile_y", "label", "geometry"],
            geometry=[], crs="EPSG:4326",
        ).to_file(out_geojson, driver="GeoJSON")
        return out_geojson

    gdf = gpd.GeoDataFrame(
        {k: [r[k] for r in rows_all] for k in ("tile_z", "tile_x",
                                               "tile_y", "label")},
        geometry=[r["geometry_wm"] for r in rows_all],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    _LOG.info("wrote %d polygons -> %s", len(gdf), out_geojson)
    return out_geojson


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--masks", type=Path, required=True,
                   help="Root of the per-tile mask tree (<z>/<x>/<y>.png).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output GeoJSON (EPSG:4326).")
    p.add_argument("--min-marker-distance-m", type=float, default=5.0,
                   help="Minimum metric separation between two watershed markers.")
    p.add_argument("--prob-threshold", type=float, default=0.5,
                   help="(Reserved) threshold applied to probability maps.")
    args = p.parse_args(argv)
    cfg = WatershedConfig(
        min_marker_distance_m=args.min_marker_distance_m,
        prob_threshold=args.prob_threshold,
    )
    vectorize(args.masks, args.out, cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
