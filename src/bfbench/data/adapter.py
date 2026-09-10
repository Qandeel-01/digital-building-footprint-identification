"""SpaceNet PS-RGB GeoTIFFs -> abd tile layout.

Upstream abd_model.tools.predict consumes tiles laid out as a slippy-map
directory tree:

    <root>/<channel_name>/<z>/<x>/<y>.<ext>

where (x, y, z) are XYZ Web-Mercator tile coordinates (mercantile.Tile),
the default channel_name is 'images', and ext is anything rasterio can
open (png works, tiff works). The neural net expects 512x512 RGB per tile
(from the shape_in of the checkpoint).

Two decisions are made explicit and configurable because they will confound
any downstream benchmark:

1. **GSD resampling.** SpaceNet 2 PS-RGB is ~30 cm/pixel. The xBD-trained
   AlbuNet was trained on ~50 cm/pixel imagery. Predicting on native 30 cm
   without resampling silently changes the model's effective receptive
   field. We resample to a configurable target GSD (default: 0.5 m/pixel)
   using cubic resampling and record the choice in the run manifest.
2. **Tile size and border handling.** The AlbuNet checkpoint expects 512
   pixel tiles. With `--metatiles --keep_borders`, upstream stitches the
   8 neighbouring tiles at inference time to reduce edge artifacts. Our
   adapter picks a slippy-map zoom whose native tile GSD is close to the
   target GSD, tiles the imagery WITH one tile of slack on each side, and
   writes the surrounding neighbours to disk so that `--metatiles`
   works. See docs/adapter-decisions.md.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

_LOG = logging.getLogger(__name__)

# EPSG:3857 Web Mercator (the CRS mercantile.xy_bounds() uses)
WEBMERCATOR_EPSG = 3857
WGS84_EPSG = 4326

# Circumference of Earth at the equator, in meters, in EPSG:3857 units.
# 2 * pi * 6378137 = 40075016.6855784
_EARTH_CIRC_M = 2.0 * math.pi * 6378137.0


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for the SpaceNet -> abd tile adapter.

    Attributes:
        target_gsd_m: metres/pixel that imagery is resampled to before
            tiling. Default 0.5 matches the xBD training GSD of the
            upstream AlbuNet checkpoint.
        tile_size: pixel side length of one abd tile. Default 512 matches
            the checkpoint's shape_in. Do not change without a matching
            checkpoint.
        zoom: XYZ zoom level for the output tile pyramid. If None it is
            picked so the native tile GSD (at the AOI's latitude) is
            within `zoom_tolerance` of target_gsd_m.
        zoom_tolerance: acceptable |native_gsd - target_gsd| when auto-
            picking a zoom, in metres.
        keep_borders: if True, always emit every tile that intersects the
            AOI (matches `abd predict --metatiles --keep_borders`). If
            False, only emit interior tiles that have all 8 neighbours
            (matches `abd predict --metatiles` without --keep_borders).
        channel_name: name of the abd 'channel' subdirectory. Upstream's
            default config uses 'images'.
        resampling: rasterio Resampling enum member used when reprojecting.
    """

    target_gsd_m: float = 0.5
    tile_size: int = 512
    zoom: int | None = None
    zoom_tolerance: float = 0.15
    keep_borders: bool = True
    channel_name: str = "images"
    resampling: Resampling = Resampling.cubic

    def __post_init__(self) -> None:
        if self.target_gsd_m <= 0:
            raise ValueError("target_gsd_m must be positive")
        if self.tile_size <= 0 or self.tile_size % 2:
            raise ValueError("tile_size must be a positive even integer")
        if self.zoom is not None and not (0 <= self.zoom <= 22):
            raise ValueError("zoom must be in [0, 22] if specified")


# ---------------------------------------------------------------------------
# Zoom / GSD math
# ---------------------------------------------------------------------------

def webmercator_gsd_at(lat_deg: float, zoom: int, tile_size: int = 256) -> float:
    """Metres per pixel at the given latitude, if a single XYZ tile at
    `zoom` is rendered at `tile_size` pixels on a side.

    The GEOGRAPHIC extent of an (x, y, z) tile is a property of z only
    (mercantile.xy_bounds). The PIXEL size we choose to write to disk
    then determines the per-pixel GSD: at tile_size=256 (canonical slippy
    map) this yields the well-known ~156543 m/px at the equator z=0.
    """
    lat_rad = math.radians(lat_deg)
    return _EARTH_CIRC_M * math.cos(lat_rad) / (tile_size * (2 ** zoom))


def pick_zoom_for_gsd(
    lat_deg: float,
    target_gsd_m: float,
    tolerance_m: float = 0.15,
    tile_size: int = 256,
) -> int:
    """Pick the zoom whose per-pixel GSD, when rendering tiles at `tile_size`
    pixels, is closest to `target_gsd_m` at latitude `lat_deg`. Raises if
    the best match is further than `tolerance_m` from the target.

    Passing tile_size=512 (the AlbuNet checkpoint's shape_in) selects a
    zoom one level LOWER than the canonical-256 case: the same z-tile
    covers the same geographic extent, but at 512 pixels the effective
    per-pixel GSD is halved.
    """
    lat_rad = math.radians(lat_deg)
    ideal = math.log2(_EARTH_CIRC_M * math.cos(lat_rad) / (tile_size * target_gsd_m))
    z_lo, z_hi = math.floor(ideal), math.ceil(ideal)
    candidates = {
        z: abs(webmercator_gsd_at(lat_deg, z, tile_size=tile_size) - target_gsd_m)
        for z in (z_lo, z_hi)
    }
    best = min(candidates, key=candidates.get)
    err = candidates[best]
    if err > tolerance_m:
        raise ValueError(
            f"no XYZ zoom yields GSD within {tolerance_m} m of target "
            f"{target_gsd_m} m at lat={lat_deg:.4f} with tile_size={tile_size}; "
            f"best is z={best} with per_pixel_gsd="
            f"{webmercator_gsd_at(lat_deg, best, tile_size=tile_size):.3f} m "
            f"(err={err:.3f} m). Widen zoom_tolerance to accept."
        )
    return best


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

def tile_bounds_webmercator(tile: mercantile.Tile) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) of `tile` in EPSG:3857 metres."""
    b = mercantile.xy_bounds(tile)
    return (b.left, b.bottom, b.right, b.top)


def tiles_covering(
    raster_bounds_wgs84: tuple[float, float, float, float], zoom: int
) -> list[mercantile.Tile]:
    """List XYZ tiles at `zoom` that intersect the WGS84 bbox of a raster.

    mercantile.tiles handles crossing dateline / high latitudes correctly.
    """
    west, south, east, north = raster_bounds_wgs84
    return list(mercantile.tiles(west, south, east, north, [zoom]))


def _reproject_to_webmercator(src_path: Path, target_gsd_m: float, resampling: Resampling):
    """Reproject a source raster to EPSG:3857 at the requested GSD.

    Returns an in-memory (dest_data, dest_transform, crs) triple.
    dest_data has shape (bands, H, W).
    """
    with rasterio.open(src_path) as src:
        dst_crs = f"EPSG:{WEBMERCATOR_EPSG}"
        # Compute the natural transform + shape, then override resolution.
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_gsd_m
        )
        dst = np.zeros((src.count, height, width), dtype=src.dtypes[0])
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=dst[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )
        return dst, transform, dst_crs


def _read_window_from_reprojected(
    reprojected: np.ndarray,
    src_transform,
    bbox_3857: tuple[float, float, float, float],
    tile_size: int,
) -> np.ndarray | None:
    """Extract a `tile_size` x `tile_size` window from an already-reprojected
    numpy array (bands, H, W) whose transform is `src_transform` and CRS is
    EPSG:3857. Returns (tile_size, tile_size, 3) uint8, or None if the tile
    window falls outside the reprojected extent.
    """
    left, bottom, right, top = bbox_3857
    src_height, src_width = reprojected.shape[1], reprojected.shape[2]
    win = window_from_bounds(left, bottom, right, top, transform=src_transform)

    col_off = int(round(win.col_off))
    row_off = int(round(win.row_off))
    if (
        col_off < 0
        or row_off < 0
        or col_off + tile_size > src_width
        or row_off + tile_size > src_height
    ):
        return None

    patch = reprojected[:, row_off : row_off + tile_size, col_off : col_off + tile_size]
    # bands-first (C,H,W) -> H,W,C
    patch = np.moveaxis(patch, 0, -1)
    # Take the first 3 bands as RGB and clamp to uint8 for the writer.
    if patch.shape[-1] >= 3:
        patch = patch[..., :3]
    if patch.dtype != np.uint8:
        # SpaceNet PS-RGB is uint16 in native form; scale by percentile
        # stretch to avoid a dark image. This is a benchmarking harness, not
        # a training pipeline, so a fixed p2/p98 stretch is deliberate.
        lo, hi = np.percentile(patch, (2, 98))
        if hi <= lo:
            hi = lo + 1
        patch = np.clip((patch.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(
            np.uint8
        )
    return patch


def adapt_spacenet_image(
    src_image: Path,
    out_root: Path,
    aoi_bbox_wgs84: tuple[float, float, float, float] | None = None,
    config: AdapterConfig | None = None,
) -> AdapterResult:
    """Convert one SpaceNet PS-RGB GeoTIFF into abd's slippy-map tile tree.

    Args:
        src_image: path to a SpaceNet PS-RGB .tif file.
        out_root: directory to write the abd tile tree into. Created if
            absent. The resulting layout is
            <out_root>/<config.channel_name>/<z>/<x>/<y>.tiff and
            <out_root>/cover.csv.
        aoi_bbox_wgs84: optional (west, south, east, north) bounding box in
            WGS84 to restrict tiling to. If None, uses the raster's own
            bounds reprojected to WGS84.
        config: AdapterConfig instance; defaults if None.

    Returns:
        AdapterResult summarizing what was written.
    """
    config = config or AdapterConfig()
    src_image = Path(src_image)
    out_root = Path(out_root)

    with rasterio.open(src_image) as src:
        src_bounds_4326 = transform_bounds(src.crs, f"EPSG:{WGS84_EPSG}", *src.bounds)
    aoi_bbox_wgs84 = aoi_bbox_wgs84 or src_bounds_4326
    center_lat = 0.5 * (aoi_bbox_wgs84[1] + aoi_bbox_wgs84[3])

    zoom = config.zoom if config.zoom is not None else pick_zoom_for_gsd(
        center_lat,
        config.target_gsd_m,
        tolerance_m=config.zoom_tolerance,
        tile_size=config.tile_size,
    )
    # Per-output-pixel GSD implied by writing `tile_size` px into each
    # (x, y, z)-tile's geographic extent.
    tile_gsd = webmercator_gsd_at(center_lat, zoom, tile_size=config.tile_size)
    _LOG.info(
        "adapter: lat=%.4f target_gsd=%.3fm tile_size=%d chosen_zoom=%d effective_gsd=%.3fm",
        center_lat, config.target_gsd_m, config.tile_size, zoom, tile_gsd,
    )

    # Reproject once to Web Mercator at the effective per-pixel GSD.
    # Every tile is then a `tile_size` x `tile_size` slice of this array.
    reprojected, transform, _ = _reproject_to_webmercator(
        src_image, tile_gsd, config.resampling
    )

    tiles = tiles_covering(aoi_bbox_wgs84, zoom)
    channel_root = out_root / config.channel_name
    channel_root.mkdir(parents=True, exist_ok=True)

    tiles_written: list[mercantile.Tile] = []
    tiles_skipped_out_of_extent: list[mercantile.Tile] = []
    for t in tiles:
        bbox_3857 = tile_bounds_webmercator(t)
        patch = _read_window_from_reprojected(
            reprojected, transform, bbox_3857, config.tile_size
        )
        if patch is None:
            tiles_skipped_out_of_extent.append(t)
            continue
        _write_tile(channel_root, t, patch)
        tiles_written.append(t)

    if not config.keep_borders:
        # Drop tiles whose 8-neighbourhood isn't fully present on disk.
        written_set = {(t.x, t.y, t.z) for t in tiles_written}
        interior = [
            t
            for t in tiles_written
            if all(
                (t.x + dx, t.y + dy, t.z) in written_set
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if not (dx == 0 and dy == 0)
            )
        ]
        _LOG.info("adapter: keep_borders=False; %d -> %d tiles", len(tiles_written), len(interior))
        tiles_written = interior

    cover_path = out_root / "cover.csv"
    with cover_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for t in tiles_written:
            writer.writerow([t.x, t.y, t.z])

    return AdapterResult(
        source_image=src_image,
        out_root=out_root,
        zoom=zoom,
        effective_gsd_m=tile_gsd,
        target_gsd_m=config.target_gsd_m,
        tile_size=config.tile_size,
        tiles_written=len(tiles_written),
        tiles_skipped_out_of_extent=len(tiles_skipped_out_of_extent),
        keep_borders=config.keep_borders,
        cover_csv=cover_path,
    )


def _write_tile(channel_root: Path, tile: mercantile.Tile, patch: np.ndarray) -> None:
    """Write a single tile at <channel_root>/<z>/<x>/<y>.tiff."""
    d = channel_root / str(tile.z) / str(tile.x)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{tile.y}.tiff"
    Image.fromarray(patch).save(path, format="TIFF")


@dataclass(frozen=True)
class AdapterResult:
    """Return value of adapt_spacenet_image."""

    source_image: Path
    out_root: Path
    zoom: int
    effective_gsd_m: float
    target_gsd_m: float
    tile_size: int
    tiles_written: int
    tiles_skipped_out_of_extent: int
    keep_borders: bool
    cover_csv: Path
