"""Per-AOI mosaic of SpaceNet chips as a virtual raster (GDAL VRT).

Phase 1 tiled chip-by-chip. On SpaceNet 2 chips are ~200 m per side
and z=17 tiles at tile_size=512 span ~247 m, so every chip-tile
overlap included a zero-padded ring around the chip footprint. That
propagates into any tile-boundary metric (edge buildings, coverage,
IoU per tile).

Phase 2 fixes this by building a single per-AOI virtual raster
(gdalbuildvrt), then tiling that. Tiles are emitted only when their
footprint is fully covered by real imagery -- the `min_valid_fraction`
knob is exposed but for the benchmark it should stay at 1.0 (fully
covered).

The VRT is preferred over an in-memory rasterio VRT because:
    * gdalbuildvrt handles per-chip nodata + non-matching CRSs at build
      time and produces a plain XML on disk we can commit to a run
      manifest.
    * It streams from the individual chips at read time, so the mosaic
      is O(1) disk regardless of AOI size.
    * rasterio can `open` a VRT directly -- no extra plumbing needed.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MosaicResult:
    vrt_path: Path
    n_chips: int
    bounds_wgs84: tuple[float, float, float, float]
    native_crs: str


def build_vrt(chip_paths: list[Path], out_vrt: Path,
              resampling: str = "cubic") -> MosaicResult:
    """Build a GDAL VRT covering all chip paths.

    Args:
        chip_paths: individual GeoTIFF chip paths in native SpaceNet CRS.
        out_vrt: destination .vrt file. Parent dir must exist.
        resampling: gdalbuildvrt -r flag.

    Raises:
        RuntimeError: if gdalbuildvrt exits nonzero, or if chip CRSs differ
        (mixing CRSs in one VRT is legal only if they match; SpaceNet
        chips inside one AOI share a UTM zone by construction).
    """
    if not chip_paths:
        raise ValueError("build_vrt: no chip paths provided")
    out_vrt = Path(out_vrt)
    out_vrt.parent.mkdir(parents=True, exist_ok=True)

    # Sanity: all chips must share one CRS. gdalbuildvrt does not silently
    # reproject; mismatched CRSs produce a corrupt VRT.
    crss = set()
    for cp in chip_paths:
        with rasterio.open(cp) as src:
            crss.add(str(src.crs))
    if len(crss) != 1:
        raise RuntimeError(
            f"build_vrt: chips span multiple CRSs {sorted(crss)}. "
            "SpaceNet AOIs are single-UTM-zone by construction; verify "
            "the chip list belongs to one AOI."
        )
    native_crs = next(iter(crss))

    args = ["gdalbuildvrt", "-r", resampling, "-overwrite", str(out_vrt)]
    args += [str(p) for p in chip_paths]
    _LOG.info("gdalbuildvrt %s -> %s (%d chips)", resampling, out_vrt, len(chip_paths))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gdalbuildvrt failed (exit {proc.returncode}):\n"
            f"{proc.stderr}\n{proc.stdout}"
        )

    with rasterio.open(out_vrt) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return MosaicResult(
        vrt_path=out_vrt,
        n_chips=len(chip_paths),
        bounds_wgs84=tuple(bounds_wgs84),
        native_crs=native_crs,
    )


def tile_is_fully_covered(vrt_path: Path,
                          bbox_native: tuple[float, float, float, float],
                          min_valid_fraction: float = 1.0) -> bool:
    """True iff the fraction of imagery-covered pixels within
    `bbox_native` (in the mosaic's native CRS) is >= `min_valid_fraction`.

    A pixel is considered "covered" iff at least one band reads a nonzero
    value at that pixel. Rationale: SpaceNet 2 PS-RGB chips don't set a
    `nodata` tag, so `rasterio.read` on a VRT returns 0 for pixels
    outside any source (that's the boundless `fill_value`) AND for truly
    dark pixels. The "any band nonzero" test both cases as invalid,
    which is the correct behaviour for a coverage question (a pixel
    that reads (0, 0, 0) is unusable regardless of why).
    """
    with rasterio.open(vrt_path) as src:
        left, bottom, right, top = bbox_native
        window = rasterio.windows.from_bounds(left, bottom, right, top,
                                              transform=src.transform)
        try:
            data = src.read(window=window, boundless=True, fill_value=0)
        except rasterio.errors.RasterioIOError:
            return False
        if data.size == 0 or data.shape[-1] == 0 or data.shape[-2] == 0:
            return False
        any_band_nonzero = (data != 0).any(axis=0)
        frac = float(any_band_nonzero.mean())
        return frac >= min_valid_fraction
