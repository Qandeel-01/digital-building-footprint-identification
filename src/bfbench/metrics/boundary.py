"""Boundary distance metrics: Chamfer and Hausdorff.

RESERVED STUB. Depends on MatchResult, which may change when
match_instances is implemented. See docs/metric-decisions.md section 6
for the sampling convention (densified at `sample_step_m` intervals,
default 0.5 m -- NOT vertex-only).
"""

from __future__ import annotations

from bfbench.metrics.instance import MatchResult


def boundary_distance(match_result: MatchResult,
                      sample_step_m: float = 0.5) -> dict[str, float]:
    """Mean Chamfer and mean Hausdorff distance across matched pairs.

    Args:
        match_result: from match_instances (in an equal-area CRS).
        sample_step_m: point-set sampling interval along polygon
            boundaries, in metres. Default 0.5 m. Vertex-only sampling
            is deliberately not supported -- it measures vectoriser
            output, not geometry (see docs/metric-decisions.md section 6).

    Returns:
        {"mean_chamfer": ..., "mean_hausdorff": ...} in metres.

    OPEN DECISIONS (implementer's call):
        1. Hausdorff: directed vs symmetric.
        2. What to do with unmatched polygons (exclude vs +inf vs
           fallback).

    Raises:
        NotImplementedError.
    """
    if sample_step_m <= 0:
        raise ValueError(f"sample_step_m must be positive, got {sample_step_m}")
    raise NotImplementedError(
        "boundary_distance is stubbed pending final MatchResult design "
        "and the Hausdorff-variant + unmatched-handling decisions."
    )
