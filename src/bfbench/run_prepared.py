"""End-to-end pipeline on prepared Vegas mosaic (Phase 3 vertical slice).

`bfbench run configs/spacenet_vegas.yaml` in the Phase 1 shape adapts
per-chip. Phase 3 requires the run to operate on the per-AOI MOSAIC
and only on the SAMPLED tiles from `bfbench prepare`. This module
implements that path:

    1. Read <work_dir>/sampled_tiles.json (from prepare) and
       <work_dir>/mosaic.vrt (from prepare).

    2. For each sampled tile: read the tile's geographic bbox from
       the mosaic, reproject that window to Web Mercator at the
       tile's effective GSD, write the tile to
       <work_dir>/tiles/<channel_name>/<z>/<x>/<y>.tiff. Also emit
       the metatile neighbours (8-ring) so upstream's `--metatiles`
       has neighbours to read.

    3. Write <work_dir>/tiles/cover.csv listing the CORE sampled
       tiles (not the neighbours; upstream reads neighbours by
       existence, not by cover).

    4. Invoke predict_local.py -> masks/<z>/<x>/<y>.png.

    5. Invoke abd vectorize -> predicted_raw.geojson.

    6. Invoke filter-buildings -> predicted.geojson.

This is independent of `bfbench run`'s per-chip path (which stays
around for Phase 1's single-chip smoke config).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mercantile
import numpy as np
import rasterio
import yaml
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject as _rio_reproject
from rasterio.warp import transform_bounds

from bfbench.data.adapter import (
    WEBMERCATOR_EPSG,
    pick_zoom_for_gsd,
    tile_bounds_webmercator,
    webmercator_gsd_at,
)

_LOG = logging.getLogger("bfbench.run_prepared")


@dataclass(frozen=True)
class PreparedRunConfig:
    aoi_name: str
    work_dir: Path
    target_gsd_m: float
    tile_size: int
    zoom_tolerance: float
    keep_borders: bool
    channel_name: str
    abd_config: Path
    checkpoint: Path
    vectorize_type: str
    metatiles: bool


def _load(cfg_path: Path) -> PreparedRunConfig:
    y = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ad = y["adapter"]; abd = y["abd"]; run = y["run"]
    return PreparedRunConfig(
        aoi_name=y["aoi"]["name"],
        work_dir=Path(run["work_dir"]),
        target_gsd_m=float(ad["target_gsd_m"]),
        tile_size=int(ad["tile_size"]),
        zoom_tolerance=float(ad["zoom_tolerance"]),
        keep_borders=bool(ad.get("keep_borders", True)),
        channel_name=str(ad.get("channel_name", "images")),
        abd_config=Path(abd["config"]).expanduser(),
        checkpoint=Path(abd["checkpoint"]).expanduser(),
        vectorize_type=str(abd.get("vectorize_type", "Building")),
        metatiles=bool(abd.get("metatiles", True)),
    )


def _read_tile_from_mosaic(vrt_path: Path,
                           tile: mercantile.Tile,
                           tile_size: int,
                           channel_root: Path) -> Path | None:
    """Read one XYZ tile from the mosaic, reproject to Web Mercator at
    the tile's native GSD (tile_size px on a side), and write a TIFF
    into the slippy-map tree.
    """
    w3857, s3857, e3857, n3857 = tile_bounds_webmercator(tile)
    dst_transform = rasterio.transform.from_bounds(
        w3857, s3857, e3857, n3857, tile_size, tile_size,
    )
    dst = np.zeros((3, tile_size, tile_size), dtype=np.uint8)

    with rasterio.open(vrt_path) as src:
        # Only reproject the bands the abd loader expects (RGB).
        n_src_bands = src.count
        if n_src_bands < 3:
            raise RuntimeError(
                f"mosaic has {n_src_bands} bands; need >= 3 for RGB"
            )
        # SpaceNet PS-RGB is 16-bit; stretch by p2/p98 for 8-bit output.
        # We read the source window (bbox reprojected into src CRS) and
        # let rasterio.warp.reproject handle the reprojection.
        src_bands = np.zeros((3, tile_size, tile_size), dtype=np.float32)
        for band_i in range(3):
            _rio_reproject(
                source=rasterio.band(src, band_i + 1),
                destination=src_bands[band_i],
                dst_transform=dst_transform,
                dst_crs=f"EPSG:{WEBMERCATOR_EPSG}",
                dst_nodata=0,
                resampling=Resampling.cubic,
            )
    # Percentile stretch to 8-bit, per-band, computed over the tile.
    for band_i in range(3):
        band = src_bands[band_i]
        valid = band[band > 0]
        if valid.size == 0:
            dst[band_i] = 0
            continue
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo:
            hi = lo + 1
        stretched = (band - lo) * (255.0 / (hi - lo))
        dst[band_i] = np.clip(stretched, 0, 255).astype(np.uint8)

    if not (dst != 0).any():
        return None  # empty tile

    # Write as PNG (upstream reads both PNG and TIFF via rasterio).
    out_dir = channel_root / str(tile.z) / str(tile.x)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tile.y}.png"
    # PIL writes RGB from (H, W, C).
    img = np.transpose(dst, (1, 2, 0))
    Image.fromarray(img, mode="RGB").save(out_path, format="PNG", optimize=False)
    return out_path


def _neighbour_tiles(t: mercantile.Tile) -> list[mercantile.Tile]:
    return [mercantile.Tile(t.x + dx, t.y + dy, t.z)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)]


def adapt_from_mosaic(cfg: PreparedRunConfig,
                     sampled_tiles: list[mercantile.Tile]) -> Path:
    """Write the slippy-map tree for `sampled_tiles` + their neighbours.

    Returns the tiles-root directory.
    """
    tiles_root = cfg.work_dir / "tiles"
    tiles_root.mkdir(parents=True, exist_ok=True)
    channel_root = tiles_root / cfg.channel_name
    channel_root.mkdir(parents=True, exist_ok=True)

    vrt = cfg.work_dir / "mosaic.vrt"
    if not vrt.is_file():
        raise FileNotFoundError(f"mosaic.vrt missing at {vrt}; run bfbench prepare first")

    # All tiles to write: sampled + one-ring neighbours (dedup).
    all_tiles: dict[tuple[int, int, int], mercantile.Tile] = {}
    for t in sampled_tiles:
        all_tiles[(t.z, t.x, t.y)] = t
        for n in _neighbour_tiles(t):
            all_tiles.setdefault((n.z, n.x, n.y), n)

    _LOG.info("adapter: writing %d tiles (%d core + %d neighbours) to %s",
              len(all_tiles), len(sampled_tiles),
              len(all_tiles) - len(sampled_tiles), channel_root)

    written = 0
    for i, ((_, _, _), t) in enumerate(all_tiles.items(), start=1):
        # Skip if already written (resumable).
        out_path = channel_root / str(t.z) / str(t.x) / f"{t.y}.png"
        if out_path.is_file():
            continue
        if _read_tile_from_mosaic(vrt, t, cfg.tile_size, channel_root) is not None:
            written += 1
        if i % 25 == 0:
            _LOG.info("  adapter progress: %d/%d tiles (new=%d)",
                      i, len(all_tiles), written)

    # cover.csv: sampled tiles only (upstream's --metatiles reads
    # neighbours by existence).
    cover_path = tiles_root / "cover.csv"
    with cover_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        for t in sampled_tiles:
            w.writerow([t.x, t.y, t.z])
    return tiles_root


def _resolve_bin(name: str) -> str:
    """Find `name` on PATH or in the running env's Scripts/Library/bin.

    Windows subprocess doesn't inherit conda-env PATH modifications
    unless we made them ourselves, so `abd` / `filter-buildings`
    silently WinError 2. Fall back to interpreter-relative locations.
    """
    import shutil  # noqa: PLC0415
    exe = shutil.which(name)
    if exe:
        return exe
    for suffix in (
        f"Scripts/{name}.exe",
        f"Scripts/{name}",
        f"Library/bin/{name}.exe",
        f"Library/bin/{name}",
    ):
        cand = Path(sys.prefix) / suffix
        if cand.is_file():
            return str(cand)
    raise RuntimeError(f"{name!r} not on PATH and not in {sys.prefix}")


def _run(argv: list[str]) -> None:
    if not Path(argv[0]).is_file() and not argv[0].endswith((".py",)):
        # First arg is a bare command name; resolve to full path.
        argv = [_resolve_bin(argv[0]), *argv[1:]]
    _LOG.info("+ %s", " ".join(str(a) for a in argv))
    # Ensure GDAL/PROJ data files are found in the subprocess env.
    env = os.environ.copy()
    env.setdefault("GDAL_DATA", str(Path(sys.prefix) / "Library" / "share" / "gdal"))
    env.setdefault("PROJ_LIB", str(Path(sys.prefix) / "Library" / "share" / "proj"))
    _lib_bin = str(Path(sys.prefix) / "Library" / "bin")
    if _lib_bin not in env.get("PATH", ""):
        env["PATH"] = _lib_bin + os.pathsep + env.get("PATH", "")
    r = subprocess.run(argv, check=False, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (exit={r.returncode}): {' '.join(str(a) for a in argv)}")


def predict_vectorize_filter(cfg: PreparedRunConfig, tiles_root: Path) -> Path:
    cover = tiles_root / "cover.csv"
    masks_dir = cfg.work_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    predict_local = Path(__file__).resolve().parent.parent.parent / \
                    "scripts" / "predict_local.py"

    argv = [sys.executable, str(predict_local),
            "--dataset", str(tiles_root),
            "--checkpoint", str(cfg.checkpoint),
            "--config", str(cfg.abd_config),
            "--cover", str(cover),
            "--out", str(masks_dir)]
    if cfg.metatiles:
        argv.append("--metatiles")
        if cfg.keep_borders:
            argv.append("--keep_borders")
    _run(argv)

    raw_geojson = cfg.work_dir / "predicted_raw.geojson"
    _run(["abd", "vectorize",
          "--masks", str(masks_dir),
          "--type", cfg.vectorize_type,
          "--config", str(cfg.abd_config),
          "--out", str(raw_geojson)])

    final = cfg.work_dir / "predicted.geojson"
    _run(["filter-buildings",
          "--data", str(raw_geojson),
          "--dest", str(final)])
    return final


def run_prepared(cfg_path: Path) -> Path:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = _load(cfg_path)

    sampled_json = cfg.work_dir / "sampled_tiles.json"
    if not sampled_json.is_file():
        raise FileNotFoundError(
            f"sampled_tiles.json missing at {sampled_json}. Run "
            "`bfbench prepare` first."
        )
    payload = json.loads(sampled_json.read_text(encoding="utf-8"))
    tiles = [mercantile.Tile(x, y, z) for z, x, y in payload["tiles"]]
    _LOG.info("running on %d sampled tiles (zoom=%d, effective_gsd=%.3fm)",
              len(tiles), payload["zoom"], payload["effective_gsd_m"])

    tiles_root = adapt_from_mosaic(cfg, tiles)
    return predict_vectorize_filter(cfg, tiles_root)


__all__ = ["PreparedRunConfig", "run_prepared", "adapt_from_mosaic"]
