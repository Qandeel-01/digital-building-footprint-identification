"""3-class target masks for Phase 2a (touching-building separation).

Each pixel is labelled as one of:

    0  background        (outside every reference polygon)
    1  interior          (inside a reference polygon)
    2  border contact    (inside a reference polygon AND within
                          `border_distance_m` of SOME OTHER reference
                          polygon's boundary)

The "border contact" definition is deliberately *not* a uniform ring
around every building. For isolated buildings (nearest neighbour
farther than `border_distance_m`) no pixel becomes class 2 — the
model's job is to learn edges that separate two touching buildings,
not edges in general. The classic U-Net "border weighting" trick
would mark every edge; that is a reasonable alternative, but the
brief for Phase 2 is specifically to **separate touching buildings**,
so we scope class 2 to touching-pair edges only.

CRS: masks are rasterised in the tile's EPSG:3857 (Web-Mercator)
grid, matching `rasterise_reference_to_tile` in train/dataset.py so
3-class and binary datasets agree pixel-for-pixel on class 1.

See docs/phase2-decisions.md for the rationale behind the 3 m default.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import mercantile
import numpy as np
from rasterio.features import rasterize as _rasterize
from rasterio.transform import from_bounds as _from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

from bfbench.data.adapter import WEBMERCATOR_EPSG, tile_bounds_webmercator

# Class IDs used across train/dataset.py, postprocess/three_class.py,
# and metric reporting. Changing these breaks every saved 3-class
# checkpoint; do not.
CLS_BG = 0
CLS_INTERIOR = 1
CLS_BORDER = 2
N_CLASSES = 3

DEFAULT_BORDER_DISTANCE_M = 3.0


@dataclass(frozen=True)
class ThreeClassConfig:
    """Config for the 3-class mask builder.

    Attributes:
        border_distance_m: maximum distance (in metres, measured in the
            tile's EPSG:3857 grid) from a pixel inside building A to
            the boundary of SOME OTHER building B for the pixel to
            become class 2. Default 3.0 m -- small enough to leave the
            interior of mid-sized buildings alone, large enough to
            cover row-house / terraced-house gaps on SpaceNet 2.
    """

    border_distance_m: float = DEFAULT_BORDER_DISTANCE_M

    def __post_init__(self) -> None:
        if self.border_distance_m <= 0:
            raise ValueError(
                f"border_distance_m must be > 0, got {self.border_distance_m}"
            )


def _ref_in_mercator(reference_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject the reference into EPSG:3857 so the per-tile
    rasterisation uses the same CRS as the tile's bounds."""
    if reference_gdf.crs is None:
        reference_gdf = reference_gdf.set_crs("EPSG:4326")
    return reference_gdf.to_crs(f"EPSG:{WEBMERCATOR_EPSG}")


def rasterise_three_class(reference_gdf_mercator: gpd.GeoDataFrame,
                          tile: mercantile.Tile,
                          tile_size: int,
                          cfg: ThreeClassConfig | None = None) -> np.ndarray:
    """Return a {0, 1, 2} HxW uint8 mask for `tile`.

    `reference_gdf_mercator` must already be in EPSG:3857; we do not
    reproject here to keep the per-tile cost down inside a training
    loop. Use `_ref_in_mercator` (or `BuildingTileDataset`, which does
    it at construction time) to prepare the GeoDataFrame.

    The algorithm:

      1. Clip the reference to the tile's bbox.
      2. Rasterise each clipped polygon INDIVIDUALLY with a per-poly
         integer id, producing an `(H, W)` id-raster where every
         pixel either carries a polygon id (1..N) or 0 (bg).
      3. For each polygon id k, compute the Euclidean distance
         transform of "pixels NOT in polygon k". Pixels inside
         polygon k whose EDT value is <= border_distance_m / px_m are
         pixels that are both inside k and within distance of SOME
         OTHER polygon -- they become class 2.
      4. Everything else inside any polygon is class 1.

    Complexity: O(N * H * W) EDTs. In practice N is small per tile
    (<< 100) and this runs in a few ms per 512x512 tile on CPU.
    """
    cfg = cfg or ThreeClassConfig()
    out = np.zeros((tile_size, tile_size), dtype=np.uint8)
    w, s, e, n = tile_bounds_webmercator(tile)
    tile_poly = box(w, s, e, n)
    clipped = reference_gdf_mercator[reference_gdf_mercator.intersects(tile_poly)]
    if len(clipped) == 0:
        return out
    shapes = [(g, i + 1) for i, g in enumerate(clipped.geometry)
              if g is not None and not g.is_empty]
    if not shapes:
        return out

    transform = _from_bounds(w, s, e, n, tile_size, tile_size)
    id_raster = _rasterize(shapes, out_shape=(tile_size, tile_size),
                           transform=transform, fill=0,
                           all_touched=False, dtype="int32")

    # Pixel size in metres (same in x and y for a Web-Mercator tile).
    px_m = (e - w) / float(tile_size)
    border_px = cfg.border_distance_m / px_m

    unique_ids = np.unique(id_raster)
    unique_ids = unique_ids[unique_ids != 0]
    if len(unique_ids) < 2:
        # Only one polygon in the tile; nothing to "touch".
        out[id_raster != 0] = CLS_INTERIOR
        return out

    # For each polygon id k, compute the distance from "every pixel"
    # to the nearest pixel NOT in polygon k. Inside k this is the
    # distance to the nearest polygon boundary; because we rasterised
    # each polygon with its own id, "not in k" INCLUDES bg AND other
    # polygons, so this is the distance to the nearest OTHER feature
    # (any boundary), measured in pixels.
    #
    # Rule: pixels inside k whose "distance to nearest non-k pixel"
    # is <= border_px are candidates for the border class. But we
    # must restrict further to pixels where the nearest non-k pixel
    # is itself inside SOME OTHER polygon (not bg), otherwise we'd
    # mark the exterior boundary of isolated buildings.
    #
    # Implementation uses `return_indices=True` to recover the nearest
    # non-k pixel for each pixel; we then check that pixel's class.
    out[id_raster != 0] = CLS_INTERIOR
    for k in unique_ids:
        mask_k = (id_raster == k)
        # Distance from every pixel to the nearest "not-in-k" pixel,
        # plus the (y, x) indices of that nearest pixel.
        dist_px, (iy, ix) = distance_transform_edt(mask_k, return_indices=True)
        # Pixels inside k that are close enough to SOME non-k pixel:
        candidate = mask_k & (dist_px <= border_px)
        if not candidate.any():
            continue
        # The nearest non-k pixel for each candidate -- is it inside
        # another polygon (not bg)? If yes, that's a touching-pair.
        nearest_id = id_raster[iy[candidate], ix[candidate]]
        is_other_polygon = nearest_id != 0  # 0 = bg
        # Promote those pixels to border class.
        cand_coords = np.argwhere(candidate)
        promote = cand_coords[is_other_polygon]
        if len(promote):
            out[promote[:, 0], promote[:, 1]] = CLS_BORDER
    return out


def three_class_to_binary(mask: np.ndarray) -> np.ndarray:
    """Collapse a 3-class mask back to a binary interior mask.

    Used at inference time: `argmax(logits) == CLS_INTERIOR`. The
    border class is NOT merged into the interior -- that is the whole
    point: predicted-border pixels become gaps between touching
    buildings in the vectorised output.
    """
    return (mask == CLS_INTERIOR).astype(np.uint8)


__all__ = [
    "CLS_BG",
    "CLS_BORDER",
    "CLS_INTERIOR",
    "DEFAULT_BORDER_DISTANCE_M",
    "N_CLASSES",
    "ThreeClassConfig",
    "rasterise_three_class",
    "three_class_to_binary",
]
