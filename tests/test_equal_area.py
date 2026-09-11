"""Tests for the equal-area CRS assertion helper."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from bfbench.data.equal_area import (
    DEFAULT_EQUAL_AREA_EPSG,
    NonEqualAreaCRSError,
    assert_equal_area_crs,
    is_equal_area,
)


def test_default_is_equal_area():
    assert is_equal_area(f"EPSG:{DEFAULT_EQUAL_AREA_EPSG}")


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857", "EPSG:32611", "EPSG:4087"])
def test_common_not_equal_area(crs):
    """Lat/lon, Web Mercator, UTM, and equidistant cylindrical are NOT
    equal-area, and the checker must say so."""
    assert not is_equal_area(crs), f"{crs} incorrectly flagged as equal-area"


def test_assert_raises_on_wgs84():
    with pytest.raises(NonEqualAreaCRSError, match="not equal-area"):
        assert_equal_area_crs("EPSG:4326", what="test")


def test_assert_raises_on_none_crs():
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    # Deliberately don't set a CRS
    with pytest.raises(NonEqualAreaCRSError, match="has no CRS"):
        assert_equal_area_crs(gdf, what="test")


def test_assert_ok_on_default():
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=f"EPSG:{DEFAULT_EQUAL_AREA_EPSG}")
    assert_equal_area_crs(gdf, what="ok")  # no raise
