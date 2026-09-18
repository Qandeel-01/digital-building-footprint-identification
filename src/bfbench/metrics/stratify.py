"""Stratified metrics: by building size (per-tile median area) and by
local building density (buildings per hectare, moving window).

Implemented in Phase 3.

Bin definition:
    Quantile bins (equal count per bin) at the AOI level. Ties on the
    bin edges are broken by putting equal values in the LOWER bin
    (pandas.qcut default with duplicates='drop').

Confidence bands:
    Non-parametric bootstrap, n_boot = 500, 95 % percentile interval.
    Deterministic: the bootstrap uses `numpy.random.default_rng(seed)`
    with seed = 0 by default.

Density definition:
    Moving window of radius `window_m` around each tile centroid.
    Density = (number of REFERENCE buildings whose centroid is within
    window_m of the tile centroid) / (window area in ha).
    Reference-not-predicted is used deliberately (density is a
    property of the ground truth, not the model). See
    docs/metric-decisions.md section 4 for why there is no default
    window size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_BOOT_N = 500
_BOOT_SEED = 0


def _bootstrap_ci(values: np.ndarray,
                  n_boot: int = _BOOT_N,
                  seed: int = _BOOT_SEED,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean. Deterministic given `seed`."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def _required_columns(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: missing required columns {missing}")


def _stratify(df: pd.DataFrame, x_col: str, y_col: str, bin_col: str,
              n_bins: int) -> pd.DataFrame:
    """Common per-AOI quantile-bin + aggregate + bootstrap CI."""
    out_rows = []
    for aoi, sub in df.groupby("aoi", sort=True):
        if len(sub) == 0:
            continue
        try:
            bins = pd.qcut(sub[x_col], q=n_bins, duplicates="drop")
        except ValueError:
            # e.g. all-NaN or too few unique values
            continue
        for interval, group in sub.groupby(bins, observed=True):
            values = group[y_col].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            n = int(len(values))
            if n == 0:
                continue
            lo, hi = _bootstrap_ci(values)
            out_rows.append({
                "aoi":         aoi,
                bin_col:       str(interval),
                "bin_left":    float(interval.left),
                "bin_right":   float(interval.right),
                "metric":      group["metric"].iloc[0] if "metric" in group.columns else "value",
                "value_mean":  float(values.mean()),
                "value_ci_lo": lo,
                "value_ci_hi": hi,
                "n_tiles":     n,
            })
    return pd.DataFrame(out_rows)


def stratify_by_area(results: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Bin per-tile results by median-reference-building-area quantile.

    Input columns required:
        aoi, metric, value, median_ref_area_m2

    Returns columns:
        aoi, area_bin, bin_left, bin_right, metric, value_mean,
        value_ci_lo, value_ci_hi, n_tiles
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    _required_columns(results, ["aoi", "value", "median_ref_area_m2"],
                      what="stratify_by_area")
    return _stratify(results, "median_ref_area_m2", "value", "area_bin", n_bins)


def stratify_by_density(results: pd.DataFrame, window_m: float,
                        n_bins: int = 5) -> pd.DataFrame:
    """Bin per-tile results by local building density.

    `window_m` is REQUIRED (no default; see docs/metric-decisions.md
    section 4). The harness runs the sweep at 100, 200 and 500 m and
    emits results at each window into the Parquet with a `window_m`
    column, so robustness to window choice is measurable rather than
    assumed.

    Input columns required:
        aoi, metric, value, ref_density_per_ha
    Optional:
        window_m -- if present the results are filtered to matching
        window_m rows before stratifying.

    Returns columns:
        aoi, density_bin, bin_left, bin_right, metric, value_mean,
        value_ci_lo, value_ci_hi, n_tiles
    """
    if window_m is None:
        raise TypeError("stratify_by_density: window_m is required (no default).")
    if window_m <= 0:
        raise ValueError(f"window_m must be positive, got {window_m}")
    _required_columns(results, ["aoi", "value", "ref_density_per_ha"],
                      what="stratify_by_density")
    if "window_m" in results.columns:
        results = results[results["window_m"].astype(float) == float(window_m)]
    return _stratify(results, "ref_density_per_ha", "value", "density_bin", n_bins)
