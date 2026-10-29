"""Count-error metrics.

Implemented in Phase 3.

Per-tile:  abs_error, signed_error only. Per-tile relative error is
deliberately omitted -- see docs/metric-decisions.md section 5.

AOI-level: count_error_aoi aggregates tile-level errors and reports
the relative error at that level, where the divisor is never zero
for any run that has at least one reference polygon.
"""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd


def count_error(pred_gdf: gpd.GeoDataFrame,
                ref_gdf: gpd.GeoDataFrame) -> dict[str, float]:
    """Signed and absolute count error for one tile.

    Returns:
        {"abs_error": |n_pred - n_ref|,
         "signed_error": n_pred - n_ref}

    Both are ints returned as float for JSON/Parquet friendliness.
    """
    n_pred = int(len(pred_gdf))
    n_ref  = int(len(ref_gdf))
    return {
        "abs_error":    float(abs(n_pred - n_ref)),
        "signed_error": float(n_pred - n_ref),
    }


def count_error_aoi(rows: pd.DataFrame) -> dict[str, float]:
    """AOI-level count error aggregated across tiles.

    Args:
        rows: a DataFrame with per-tile columns "n_pred" and "n_ref".
            Extra columns are ignored.

    Returns:
        {"abs_error": sum |n_pred - n_ref|,
         "signed_error": sum n_pred - sum n_ref,
         "relative_error": signed / max(sum n_ref, 1)  (NaN iff sum n_ref == 0)}

    The AOI-level relative error is meaningful because sum(n_ref) is
    never zero for any real AOI. Per-tile relative error is not
    provided (see docs/metric-decisions.md section 5).
    """
    if "n_pred" not in rows.columns or "n_ref" not in rows.columns:
        raise ValueError("count_error_aoi: rows must have n_pred and n_ref columns")
    n_pred = int(rows["n_pred"].sum())
    n_ref  = int(rows["n_ref"].sum())
    abs_e  = float((rows["n_pred"] - rows["n_ref"]).abs().sum())
    signed = float(n_pred - n_ref)
    rel = math.nan if n_ref == 0 else signed / n_ref
    return {
        "abs_error":      abs_e,
        "signed_error":   signed,
        "relative_error": rel,
    }
