"""Tests for the per-AOI descriptive summary."""

from __future__ import annotations

import math

import geopandas as gpd
import mercantile
import pytest
from shapely.geometry import Point

from bfbench.data.summary import summarize_aoi


def _one_tile_and_ref(n_buildings: int, tile: mercantile.Tile,
                      building_side_m: float = 10.0):
    """Put `n_buildings` little squares inside `tile` bounds, in EPSG:4326."""
    b = mercantile.bounds(tile)
    # Buildings placed on a grid inside the tile
    cols = int(math.sqrt(n_buildings))
    rows = math.ceil(n_buildings / cols)
    dx = (b.east - b.west) / (cols + 1)
    dy = (b.north - b.south) / (rows + 1)
    polys = []
    r = 1e-6  # tiny geographic-degree half-side just for centroid placement
    for i in range(n_buildings):
        col, row = i % cols, i // cols
        cx = b.west + (col + 1) * dx
        cy = b.south + (row + 1) * dy
        polys.append(Point(cx, cy).buffer(r))
    return gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")


def test_no_tiles_gives_nan_metrics():
    ref = _one_tile_and_ref(5, mercantile.Tile(1, 1, 17))
    s = summarize_aoi("empty", [], ref)
    assert s.tile_count == 0
    assert s.total_reference_buildings == 0
    assert math.isnan(s.median_building_area_m2)
    assert math.isnan(s.buildings_per_hectare)


def test_counts_only_refs_inside_sampled_tiles():
    """Buildings outside the sampled tiles must not count."""
    inside_tile = mercantile.Tile(x=23554, y=51377, z=17)  # the Vegas sample tile
    outside_tile = mercantile.Tile(x=99999, y=99999, z=17)
    inside = _one_tile_and_ref(6, inside_tile)
    outside = _one_tile_and_ref(3, outside_tile)
    ref = gpd.GeoDataFrame(
        gpd.pd.concat([inside, outside], ignore_index=True),
        crs="EPSG:4326",
    )
    s = summarize_aoi("mix", [inside_tile], ref)
    assert s.total_reference_buildings == 6


def test_refuses_reference_without_crs():
    ref = gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1e-6)])  # no crs
    with pytest.raises(ValueError, match="must have a CRS"):
        summarize_aoi("bad", [mercantile.Tile(1, 1, 17)], ref)


def test_density_scales_with_building_count():
    tile = mercantile.Tile(x=23554, y=51377, z=17)
    a = summarize_aoi("a", [tile], _one_tile_and_ref(4, tile))
    b = summarize_aoi("b", [tile], _one_tile_and_ref(16, tile))
    assert b.buildings_per_hectare == pytest.approx(4 * a.buildings_per_hectare, rel=0.1)
