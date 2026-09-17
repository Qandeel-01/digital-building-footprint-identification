"""Stratified metrics: by building size and by local density.

STUB ONLY. See module docstring in bfbench.metrics.
"""

from __future__ import annotations

import pandas as pd


def stratify_by_area(results: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Bucket per-tile results by median-reference-building-area quantile
    bin and re-aggregate.

    Inputs:
        results: tidy long-format DataFrame with (at minimum) columns
            "aoi", "tile_z", "tile_x", "tile_y", "metric", "value", and
            "median_ref_area_m2". Every row is one (tile, metric).
        n_bins: number of quantile bins (equal-count buckets).

    Returns a DataFrame with columns
        ("aoi", "area_bin", "metric", "value_mean", "value_ci_lo",
         "value_ci_hi", "n_tiles").

    OPEN DECISIONS:

    1. **Bin definition.** Quantile bins (equal count per bin) OR
       fixed size cuts in m2 (e.g. 0-50, 50-150, 150-500, 500+)? The
       former makes AOIs comparable at equal n; the latter makes bin
       labels physically meaningful. Pick one, document it.

    2. **Which area?** Area of the REFERENCE polygon (matched), or the
       PREDICTED polygon, or per-tile median reference area? The
       column in the input is the tile-level median already, but the
       question "F1 on tiles where the typical building is <50 m2"
       and "F1 on individual buildings with area <50 m2" are different
       aggregates. Pick one; the harness in `bfbench.eval` writes the
       per-tile median.

    3. **Confidence bands.** Bootstrap (recommended, n_boot >= 500) or
       normal approx? For small AOIs the bootstrap is essential.

    Raises:
        NotImplementedError.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    raise NotImplementedError(
        "stratify_by_area: bin definition + which area + CI method not chosen. "
        "See docstring."
    )


def stratify_by_density(results: pd.DataFrame, window_m: float) -> pd.DataFrame:
    """Bucket per-tile results by local building density (buildings per
    hectare in a moving window of radius `window_m` metres) and re-aggregate.

    Inputs:
        results: same tidy long-format DataFrame as stratify_by_area.
        window_m: moving-window radius in metres. Density at a tile is
            "buildings whose centroid lies within `window_m` of the tile
            centroid, per hectare".

    Returns a DataFrame with columns
        ("aoi", "density_bin", "metric", "value_mean", "value_ci_lo",
         "value_ci_hi", "n_tiles").

    OPEN DECISIONS:

    1. **Definition of local density.**
       - Moving window (radius `window_m`) around the tile centroid,
         counting reference polygons per hectare.
       - Or: per-tile density (buildings in tile / tile area). Simpler,
         but noisy for small tiles.
       - Or: kernel density estimate at the tile centroid.
       Different definitions change what "sparse" and "dense" mean.
       Pick one, document it.

    2. **Window size default.** `window_m = 500` yields a 78.5 ha
       window at any latitude; `window_m = 1000` -> 314 ha. Larger
       windows smooth noise, smaller windows resolve neighbourhood-scale
       heterogeneity. Pick a default; the caller can override.

    3. **Which "buildings" for density.** Reference or predicted?
       Reference is the honest denominator (density is a property of
       the ground truth, not the model). Explicitly document.

    4. **Bin definition.** Same choice as stratify_by_area (quantile
       vs fixed cuts).

    Raises:
        NotImplementedError.
    """
    if window_m <= 0:
        raise ValueError(f"window_m must be positive, got {window_m}")
    raise NotImplementedError(
        "stratify_by_density: definition of density + window size + which "
        "polys + bin definition not chosen. See docstring."
    )
