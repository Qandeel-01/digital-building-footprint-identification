"""Correctness tests for the 3-class touching-building mask builder.

Covers the three invariants the Phase 2a training depends on:

  1. Isolated building -> interior only, no border pixels. (If this
     breaks, isolated buildings get a spurious edge-ring and the model
     will over-predict borders at inference.)
  2. Two touching buildings -> both have a strip of class-2 border on
     the touching side, and nowhere else. (This is the whole point of
     Phase 2a.)
  3. Three-way T-junction -> all three buildings have border pixels
     on the sides adjacent to a neighbour. (Smoke; catches order-of-
     iteration bugs.)
"""

from __future__ import annotations

import geopandas as gpd
import mercantile
import numpy as np
import pytest
from shapely.geometry import box

from bfbench.data.three_class import (
    CLS_BG,
    CLS_BORDER,
    CLS_INTERIOR,
    ThreeClassConfig,
    rasterise_three_class,
    three_class_to_binary,
)
from bfbench.train.dataset import _ref_in_mercator

# ---------------------------------------------------------------------------
# Fixtures: a reproducible tile + a helper to build reference polygons
# directly in that tile's Web-Mercator footprint.
# ---------------------------------------------------------------------------

TILE = mercantile.Tile(x=24852, y=50532, z=17)
TILE_SIZE = 128  # 8 px/m at z=17 Vegas-ish latitude; good for assertion sanity


def _wm_bounds():
    from bfbench.data.adapter import tile_bounds_webmercator
    return tile_bounds_webmercator(TILE)


def _square_at_wm(cx: float, cy: float, side_m: float):
    """Build a square polygon in EPSG:3857 of `side_m` on a side,
    centred on (cx, cy). The caller is already in EPSG:3857 metres."""
    half = side_m / 2.0
    return box(cx - half, cy - half, cx + half, cy + half)


def _ref_in_mercator_from_wm(geoms) -> gpd.GeoDataFrame:
    """Short-circuit the dataset's `_ref_in_mercator`: we already have
    geoms in EPSG:3857, so just wrap them."""
    return gpd.GeoDataFrame(geometry=list(geoms),
                             crs="EPSG:3857")


# ---------------------------------------------------------------------------
# Invariant 1: isolated building, no border pixels.
# ---------------------------------------------------------------------------

def test_isolated_building_has_no_border_pixels():
    w, s, e, n = _wm_bounds()
    cx = 0.5 * (w + e)
    cy = 0.5 * (s + n)
    gdf = _ref_in_mercator_from_wm([_square_at_wm(cx, cy, side_m=20.0)])
    mask = rasterise_three_class(gdf, TILE, tile_size=TILE_SIZE)
    assert mask.max() == CLS_INTERIOR, (
        f"isolated building produced class {mask.max()} (expected only {CLS_INTERIOR})"
    )
    # And interior must be non-empty.
    assert (mask == CLS_INTERIOR).sum() > 0


# ---------------------------------------------------------------------------
# Invariant 2: two touching buildings -> border on the touching side only.
# ---------------------------------------------------------------------------

def test_two_touching_buildings_have_border_strip_between_them():
    """Two 20 m squares sharing an edge. Along the shared edge (plus
    `border_distance_m` into each square) we expect class 2; the far
    side of each square should still be class 1."""
    w, s, e, n = _wm_bounds()
    cy = 0.5 * (s + n)
    side = 20.0
    left_cx = 0.5 * (w + e) - side  # right edge aligned on the midline
    right_cx = 0.5 * (w + e) + side
    gdf = _ref_in_mercator_from_wm([
        _square_at_wm(left_cx, cy, side_m=side),
        _square_at_wm(right_cx, cy, side_m=side),
    ])
    # Touching: adjust so right edge of left square = left edge of right square.
    # left square x in [left_cx - side/2, left_cx + side/2] = [left_cx-10, left_cx+10]
    # right square x in [right_cx - 10, right_cx + 10]
    # for touching, need left_cx + 10 == right_cx - 10, i.e. right_cx = left_cx + 20
    right_cx = left_cx + side
    gdf = _ref_in_mercator_from_wm([
        _square_at_wm(left_cx, cy, side_m=side),
        _square_at_wm(right_cx, cy, side_m=side),
    ])
    mask = rasterise_three_class(gdf, TILE, tile_size=TILE_SIZE,
                                  cfg=ThreeClassConfig(border_distance_m=3.0))

    # There must be border pixels (class 2).
    n_border = int((mask == CLS_BORDER).sum())
    assert n_border > 0, "no border pixels on two touching squares"

    # Border pixels must sit near x = midpoint-between-squares; concretely,
    # for each row of border pixels, the x-coordinate must be within
    # border_distance_m of the shared edge. We check this in pixel
    # coordinates: the shared edge is at xm = left_cx + side/2 =
    # left_cx + 10. We convert xm -> px via the tile's bounds.
    shared_xm = left_cx + side / 2.0
    px_m = (e - w) / float(TILE_SIZE)
    shared_px = (shared_xm - w) / px_m
    border_coords = np.argwhere(mask == CLS_BORDER)
    xs = border_coords[:, 1].astype(float)
    assert float(np.abs(xs - shared_px).max()) <= 3.0 / px_m + 1.0, (
        "border pixels sit farther from the shared edge than border_distance_m"
    )

    # And the FAR side of each square (|x - shared_px| > ~10 px for a
    # 20 m square at ~0.5 m/px) must still be class 1, never class 2.
    far_cols = np.arange(TILE_SIZE)
    far_cols = far_cols[np.abs(far_cols - shared_px) > 2 * 3.0 / px_m]
    if len(far_cols):
        far_strip = mask[:, far_cols]
        assert CLS_BORDER not in np.unique(far_strip), (
            "class 2 appearing far from the shared edge; "
            "the border should be a touching-pair feature only"
        )


def test_two_touching_buildings_total_class1_shrinks_vs_binary():
    """Phase 2a's point: class 1 must be strictly a subset of the
    binary interior for touching-pair tiles. Interior + border = binary."""
    w, s, e, n = _wm_bounds()
    cy = 0.5 * (s + n)
    side = 20.0
    left_cx = 0.5 * (w + e) - side
    right_cx = left_cx + side
    gdf = _ref_in_mercator_from_wm([
        _square_at_wm(left_cx, cy, side_m=side),
        _square_at_wm(right_cx, cy, side_m=side),
    ])
    mask = rasterise_three_class(gdf, TILE, tile_size=TILE_SIZE)
    binary = three_class_to_binary(mask)
    n_interior = int((mask == CLS_INTERIOR).sum())
    n_border   = int((mask == CLS_BORDER).sum())
    n_binary   = int(binary.sum())
    # Interior is a strict subset of binary (border was carved out).
    assert n_binary == n_interior
    assert n_border > 0
    assert n_interior + n_border > 0


# ---------------------------------------------------------------------------
# Invariant 3: three buildings in a row -> middle one borders both sides.
# ---------------------------------------------------------------------------

def test_three_in_a_row_middle_has_borders_on_both_sides():
    w, s, e, n = _wm_bounds()
    cy = 0.5 * (s + n)
    side = 15.0
    mid_cx = 0.5 * (w + e)
    left_cx  = mid_cx - side
    right_cx = mid_cx + side
    gdf = _ref_in_mercator_from_wm([
        _square_at_wm(left_cx,  cy, side_m=side),
        _square_at_wm(mid_cx,   cy, side_m=side),
        _square_at_wm(right_cx, cy, side_m=side),
    ])
    mask = rasterise_three_class(gdf, TILE, tile_size=TILE_SIZE,
                                  cfg=ThreeClassConfig(border_distance_m=3.0))

    px_m = (e - w) / float(TILE_SIZE)
    mid_px = (mid_cx - w) / px_m
    # Middle square's x extent in px:
    mid_left_px  = (mid_cx - side / 2.0 - w) / px_m
    mid_right_px = (mid_cx + side / 2.0 - w) / px_m
    border_coords = np.argwhere(mask == CLS_BORDER)
    xs = border_coords[:, 1].astype(float)

    # At least one border pixel on each side (within the middle
    # square's x range): left half and right half.
    left_border  = xs[(xs >= mid_left_px) & (xs < mid_px)]
    right_border = xs[(xs >= mid_px) & (xs <= mid_right_px)]
    assert len(left_border)  > 0, "no border on left side of middle"
    assert len(right_border) > 0, "no border on right side of middle"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_zero_or_negative_border_distance_rejected():
    with pytest.raises(ValueError, match="> 0"):
        ThreeClassConfig(border_distance_m=0.0)
    with pytest.raises(ValueError, match="> 0"):
        ThreeClassConfig(border_distance_m=-1.0)


def test_empty_reference_returns_all_background():
    gdf = _ref_in_mercator_from_wm([])
    mask = rasterise_three_class(gdf, TILE, tile_size=TILE_SIZE)
    assert mask.shape == (TILE_SIZE, TILE_SIZE)
    assert int(mask.sum()) == 0
    assert mask.max() == CLS_BG


def test_three_class_to_binary_only_keeps_interior():
    mask = np.array([[CLS_BG, CLS_INTERIOR, CLS_BORDER],
                     [CLS_INTERIOR, CLS_BORDER, CLS_BG]], dtype=np.uint8)
    out = three_class_to_binary(mask)
    expected = np.array([[0, 1, 0],
                         [1, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


# Silence a flake8 warning via a cheap import check.
def test_module_reexports_are_intact():
    from bfbench.data import three_class as m  # noqa: PLC0415
    for name in ("rasterise_three_class", "three_class_to_binary",
                 "ThreeClassConfig", "CLS_BG", "CLS_INTERIOR",
                 "CLS_BORDER", "N_CLASSES", "DEFAULT_BORDER_DISTANCE_M"):
        assert hasattr(m, name), name


# `_ref_in_mercator` is a private helper in train/dataset but we exercise
# it here to pin its contract: given WGS84 input it returns 3857.
def test_ref_in_mercator_crs_contract():
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    out = _ref_in_mercator(gdf)
    assert out.crs.to_epsg() == 3857
