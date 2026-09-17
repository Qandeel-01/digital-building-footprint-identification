"""Boundary distance metrics: Chamfer and Hausdorff.

STUB ONLY. See module docstring in bfbench.metrics.
"""

from __future__ import annotations

from bfbench.metrics.instance import MatchResult


def boundary_distance(match_result: MatchResult) -> dict[str, float]:
    """Mean Chamfer and mean Hausdorff distance across matched pairs.

    Inputs:
        match_result: a MatchResult produced by match_instances, in an
            equal-area CRS.

    Returns a dict with keys:
        "mean_chamfer": mean over matched pairs of the symmetric
            Chamfer distance between prediction boundary and reference
            boundary (metres).
        "mean_hausdorff": mean over matched pairs of the directed max
            distance between the two boundaries (metres).

    OPEN DECISIONS:

    1. **Boundary discretisation.** Chamfer sums distances between
       point sets. For a polygon do we sample the boundary at a fixed
       spacing (e.g. every 0.5 m) or use vertex coordinates directly?
       Vertex-only underestimates the metric on long straight edges.
       Fixed-spacing pays a runtime cost; make the spacing configurable.

    2. **Hausdorff variants.** Directed (max distance from A to B) vs
       symmetric (max of both directions). Symmetric is standard; be
       explicit and pick one.

    3. **Unmatched polygons.** Pure boundary-distance is defined only
       for matched pairs. Do we exclude unmatched entirely from the
       mean (recommended and consistent with match-level analysis),
       or count them as +infinity, or use a fallback like the diagonal
       of the tile bbox? Pick one; a false consensus here poisons
       AOI-averaged plots.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "boundary_distance: boundary sampling + hausdorff variant + "
        "unmatched handling not chosen. See docstring."
    )
