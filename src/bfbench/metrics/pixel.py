"""Pixel-level IoU (soft segmentation IoU).

STUB ONLY. See module docstring in bfbench.metrics.
"""

from __future__ import annotations

import geopandas as gpd

from bfbench.data.equal_area import assert_equal_area_crs


def pixel_iou(pred_gdf: gpd.GeoDataFrame,
              ref_gdf: gpd.GeoDataFrame,
              bounds: tuple[float, float, float, float],
              resolution: float) -> float:
    """Pixel-level IoU inside `bounds`, at a given `resolution` in metres.

    Both GeoDataFrames must be in the SAME equal-area CRS (asserted below);
    `bounds` is (left, bottom, right, top) in that CRS's units (metres);
    `resolution` is metres per pixel.

    Returns:
        IoU as a float in [0, 1]. NaN if both rasters are empty.

    OPEN DECISIONS:

    1. **Rasterisation snapping.** The output raster is a discrete grid;
       polygon edges rarely fall on pixel centres. Do we snap using
       "all-touched" rasterisation (any pixel intersected -> 1) or
       "centre-only" (pixel centre inside polygon -> 1)?
       - "all-touched" inflates areas; "centre-only" bites off edges.
       - Both are legitimate; the implementer must pick one and be
         explicit about the pixel-size sensitivity of that choice.

    2. **Resolution.** Passing `resolution=0.5` uses ~xBD training GSD
       and is the default across the benchmark. Change with care: pixel
       IoU converges to shape IoU as resolution -> 0, so results at
       different resolutions are not directly comparable.

    3. **Empty-vs-empty.** If both rasters are entirely background,
       pixel IoU is 0/0. Return 1.0 (perfect vacuous agreement), 0.0
       (nothing found on either side), or NaN? Pick one; document why.

    Raises:
        NotImplementedError.
    """
    assert_equal_area_crs(pred_gdf, what="pixel_iou(pred_gdf)")
    assert_equal_area_crs(ref_gdf,  what="pixel_iou(ref_gdf)")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    raise NotImplementedError(
        "pixel_iou: rasterisation snapping and empty-vs-empty convention "
        "not chosen. See docstring."
    )
