"""Per-AOI descriptive statistics of the INPUT to the benchmark.

Emphasis on `INPUT`: this is not a metric of the model. These numbers
describe the sampled tiles + the reference polygons before any
prediction runs, so they can appear in the README (per Task 1e) without
violating the no-performance-numbers rule.

Every area is computed in an equal-area CRS; the `equal_area_epsg`
argument controls which one (default: EPSG:6933; see bfbench.data.equal_area).
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import mercantile
import pandas as pd
from shapely.geometry import box

from bfbench.data.equal_area import DEFAULT_EQUAL_AREA_EPSG, assert_equal_area_crs


@dataclass(frozen=True)
class AOISummary:
    aoi: str
    tile_count: int
    total_reference_buildings: int
    median_building_area_m2: float
    buildings_per_hectare: float
    equal_area_epsg: int


def summarize_aoi(aoi_name: str,
                  sampled_tiles: list[mercantile.Tile],
                  reference_gdf: gpd.GeoDataFrame,
                  equal_area_epsg: int = DEFAULT_EQUAL_AREA_EPSG) -> AOISummary:
    """Compute the per-AOI numbers documented in Task 1e.

    reference_gdf is expected in EPSG:4326; it is reprojected to
    equal-area for area computation. Only buildings whose centroid falls
    inside the sampled-tile union are counted -- otherwise per-AOI
    building density conflates model input and reference labels outside
    the sampled area.
    """
    if reference_gdf.crs is None:
        raise ValueError("reference_gdf must have a CRS set (expected EPSG:4326)")
    ea_crs = f"EPSG:{equal_area_epsg}"
    assert_equal_area_crs(ea_crs, what=f"summarize_aoi({aoi_name})")

    # Union of sampled tile footprints in WGS84
    if not sampled_tiles:
        return AOISummary(
            aoi=aoi_name, tile_count=0, total_reference_buildings=0,
            median_building_area_m2=float("nan"),
            buildings_per_hectare=float("nan"),
            equal_area_epsg=equal_area_epsg,
        )

    tile_polys_wgs = []
    for t in sampled_tiles:
        b = mercantile.bounds(t)
        tile_polys_wgs.append(box(b.west, b.south, b.east, b.north))
    tiles_gdf_wgs = gpd.GeoDataFrame(geometry=tile_polys_wgs, crs="EPSG:4326")
    tiles_union_wgs = tiles_gdf_wgs.geometry.union_all()

    ref_wgs = reference_gdf.to_crs("EPSG:4326") if reference_gdf.crs.to_epsg() != 4326 \
        else reference_gdf
    # Centroid in an equal-area CRS avoids the "geographic centroid is
    # approximate" warning and keeps the centroid-in-polygon test
    # geometrically correct.
    ref_ea_for_cent = ref_wgs.to_crs(ea_crs)
    tiles_ea_for_cent = tiles_gdf_wgs.to_crs(ea_crs)
    tiles_union_ea = tiles_ea_for_cent.geometry.union_all()
    inside_mask = ref_ea_for_cent.geometry.centroid.within(tiles_union_ea)
    inside = ref_wgs[inside_mask].reset_index(drop=True)

    ref_ea = inside.to_crs(ea_crs)
    tiles_ea = tiles_gdf_wgs.to_crs(ea_crs)
    total_tile_area_m2 = float(tiles_ea.geometry.area.sum())
    if total_tile_area_m2 <= 0:
        return AOISummary(
            aoi=aoi_name, tile_count=len(sampled_tiles),
            total_reference_buildings=int(len(ref_ea)),
            median_building_area_m2=float("nan"),
            buildings_per_hectare=float("nan"),
            equal_area_epsg=equal_area_epsg,
        )

    total_area_ha = total_tile_area_m2 / 1e4
    per_ha = len(ref_ea) / total_area_ha if total_area_ha > 0 else float("nan")
    med_area = float(ref_ea.geometry.area.median()) if len(ref_ea) else float("nan")

    return AOISummary(
        aoi=aoi_name,
        tile_count=len(sampled_tiles),
        total_reference_buildings=int(len(ref_ea)),
        median_building_area_m2=med_area,
        buildings_per_hectare=per_ha,
        equal_area_epsg=equal_area_epsg,
    )


def summaries_to_dataframe(summaries: list[AOISummary]) -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in summaries])
