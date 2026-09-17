"""Count-error metrics.

STUB ONLY. See module docstring in bfbench.metrics.
"""

from __future__ import annotations

import geopandas as gpd


def count_error(pred_gdf: gpd.GeoDataFrame,
                ref_gdf: gpd.GeoDataFrame) -> dict[str, float]:
    """Signed and absolute count error, plus relative error.

    Inputs:
        pred_gdf, ref_gdf: any CRS -- these are counts, so no area math.
            Empty inputs allowed.

    Returns a dict with keys:
        "abs_error":      |n_pred - n_ref|
        "signed_error":   n_pred - n_ref   (positive = over-count)
        "relative_error": (n_pred - n_ref) / max(n_ref, 1)  OR
                          NaN if n_ref == 0 -- see OPEN DECISION 1.

    OPEN DECISIONS:

    1. **Relative error when n_ref == 0.** Divide by 0 -> NaN, or
       return inf, or return |n_pred|? A benign-looking convention
       here changes what an "empty tile" contributes when averaged
       across an AOI.

    2. **Count of what.** Every reference polygon -- or only those
       whose centroid falls inside the sampled-tile union? For
       Phase 2's tile-level analysis the latter is correct; for
       AOI-level it depends. Should the count be documented as
       taking only in-scope polygons?

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "count_error: relative-error-at-zero convention + scope not chosen. "
        "See docstring."
    )
