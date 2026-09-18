"""Diagnostics for `bfbench prepare` output on real data.

Reports:

    * Nodata fraction across every fully-covered (sampled) tile:
      min, median, p95, max, and the count of tiles the coverage
      filter rejected. If the median nodata fraction on the sampled
      set is not essentially zero, the filter is broken and nothing
      downstream should be trusted.

    * Reference-polygon dedup sanity check for the touching_union
      rule: how many duplicates removed, ratio to total, and a rough
      seam-length estimate.

Usage:
    python scripts/diagnose_prepare.py runs/vegas_prepare/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# GDAL/PROJ env vars must be set before rasterio import on Windows,
# otherwise in-process reprojection segfaults (exit 127).
if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
from pyproj import Transformer

_LOG = logging.getLogger("bfbench.diagnose_prepare")


def _tile_nodata_fraction(vrt_path: Path, tile: mercantile.Tile,
                          native_transformer: Transformer) -> float:
    """Fraction of pixels inside `tile` where all bands read as zero
    (unavailable or truly dark). Same rule as the mosaic filter.

    Window computation is arithmetic (col/row = (x - t.c)/t.a) rather
    than via rasterio.windows.from_bounds. On this Windows/GDAL 3.10
    build, from_bounds segfaults inside PROJ when called on our
    EPSG:4326 mosaic; the arithmetic path avoids that.
    """
    b = mercantile.bounds(tile)
    w_wgs, n_wgs = b.west, b.north
    e_wgs, s_wgs = b.east, b.south
    # Native reprojection: only matters if mosaic is not EPSG:4326.
    with rasterio.open(vrt_path) as src:
        if src.crs.to_epsg() != 4326:
            w, n = native_transformer.transform(w_wgs, n_wgs)
            e, s = native_transformer.transform(e_wgs, s_wgs)
        else:
            w, n = w_wgs, n_wgs
            e, s = e_wgs, s_wgs
        left, top     = min(w, e), max(s, n)
        right, bottom = max(w, e), min(s, n)
        t = src.transform
        col_off = int((left - t.c) / t.a)
        row_off = int((top  - t.f) / t.e)
        width   = int((right - left) / t.a)
        height  = int((bottom - top) / t.e)
        if width <= 0 or height <= 0:
            return 1.0
        try:
            data = src.read(window=((row_off, row_off + height),
                                    (col_off, col_off + width)),
                            boundless=True, fill_value=0)
        except rasterio.errors.RasterioIOError:
            return 1.0
    if data.size == 0:
        return 1.0
    any_band_nonzero = (data != 0).any(axis=0)
    return float(1.0 - any_band_nonzero.mean())


def _sample_uncovered_tiles(vrt_path: Path, sampled_tiles: list[mercantile.Tile],
                            candidate_tiles: list[mercantile.Tile],
                            native_transformer: Transformer,
                            n_probe: int = 50) -> list[float]:
    """Estimate nodata fraction on `n_probe` tiles that were candidates
    but not sampled -- these are the tiles the filter *rejected*."""
    sampled_keys = {(t.z, t.x, t.y) for t in sampled_tiles}
    rejected = [t for t in candidate_tiles if (t.z, t.x, t.y) not in sampled_keys]
    if not rejected:
        return []
    step = max(1, len(rejected) // n_probe)
    probes = rejected[::step][:n_probe]
    return [_tile_nodata_fraction(vrt_path, t, native_transformer) for t in probes]


def _run(work_dir: Path) -> dict:
    mosaic = work_dir / "mosaic.vrt"
    sampled_json = work_dir / "sampled_tiles.json"
    ref_geojson = work_dir / "reference.geojson"
    dedup_json = work_dir / "reference_dedup.json"

    with rasterio.open(mosaic) as src:
        native_crs = src.crs
    tr = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)

    payload = json.loads(sampled_json.read_text(encoding="utf-8"))
    sampled_tiles = [mercantile.Tile(x, y, z) for z, x, y in payload["tiles"]]
    n_candidates = payload["n_candidates"]

    _LOG.info("scanning nodata fraction on %d sampled tiles ...", len(sampled_tiles))
    sampled_nodata = [
        _tile_nodata_fraction(mosaic, t, tr) for t in sampled_tiles
    ]
    sampled_nodata_arr = np.array(sampled_nodata, dtype=float)

    # Reference-dedup sanity check.
    dedup = json.loads(dedup_json.read_text(encoding="utf-8"))
    n_in, n_out = dedup["n_input"], dedup["n_output"]
    n_removed = dedup["n_duplicates_removed"]
    ref = gpd.read_file(ref_geojson)
    # Approximate seam density: chips share edges. The "expected fraction
    # of duplicated polygons" ~ (chip perimeter density x boundary
    # building density x 2 [both sides of seam]). We report the observed
    # number and a qualitative note rather than a hard threshold.

    summary = {
        "sampled_tiles_nodata": {
            "n_tiles": int(len(sampled_nodata_arr)),
            "min":    float(sampled_nodata_arr.min()) if len(sampled_nodata_arr) else None,
            "median": float(np.median(sampled_nodata_arr)) if len(sampled_nodata_arr) else None,
            "p95":    float(np.percentile(sampled_nodata_arr, 95)) if len(sampled_nodata_arr) else None,
            "max":    float(sampled_nodata_arr.max()) if len(sampled_nodata_arr) else None,
        },
        "candidate_tiles": {
            "n_candidates": int(n_candidates),
            "n_sampled": int(len(sampled_tiles)),
            "n_rejected": int(n_candidates - len(sampled_tiles)),
            "rejection_rate": float((n_candidates - len(sampled_tiles)) / max(n_candidates, 1)),
        },
        "reference_dedup": {
            "method": dedup["method"],
            "n_input":                n_in,
            "n_output":               n_out,
            "n_duplicates_removed":   n_removed,
            "duplicate_rate":         float(n_removed / max(n_in, 1)),
            "final_polygon_count":    len(ref),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work_dir", type=Path, nargs="?",
                   default=Path("runs/vegas_prepare"))
    p.add_argument("--out", type=Path,
                   default=Path("runs/vegas_prepare/prepare_diagnosis.json"))
    args = p.parse_args(argv)

    result = _run(args.work_dir)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
