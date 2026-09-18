"""Instance-level matching + precision/recall/F1 + merge/split rates.

RESERVED STUBS. `match_instances` and `instance_prf` are reserved for
the project owner because the whole benchmark rests on the design of
these two functions. Do not implement here. `merge_rate` and
`split_rate` consume MatchResult and are also stubbed pending the
final MatchResult design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd

from bfbench.data.equal_area import assert_equal_area_crs


@dataclass(frozen=True)
class MatchResult:
    """Result of matching predicted polygons to reference polygons.

    Attributes:
        matched: list of (pred_idx, ref_idx, iou) triples for polygons
            that satisfied the matching rule. IoU is a float in [0, 1]
            computed in the equal-area CRS the inputs were in.
        unmatched_pred: predicted indices that did not match any
            reference under the rule (false positives, ignoring merges).
        unmatched_ref: reference indices that did not match any
            prediction (false negatives, ignoring splits).
        merged_groups: list of (pred_idx, [ref_idx, ...]) tuples for
            predictions that covered >= 2 references. The definition
            of "cover" is the implementer's call.
        split_groups: list of (ref_idx, [pred_idx, ...]) tuples for
            references covered by >= 2 predictions.
        equal_area_epsg: the equal-area CRS the geometries were
            projected into for area/IoU computation.

    NOTE: this dataclass may change when the implementer designs
    match_instances. Downstream metrics (merge_rate, split_rate,
    boundary_distance) are stubbed for the same reason.
    """

    matched: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_ref: list[int] = field(default_factory=list)
    merged_groups: list[tuple[int, list[int]]] = field(default_factory=list)
    split_groups: list[tuple[int, list[int]]] = field(default_factory=list)
    equal_area_epsg: int = 6933


def match_instances(pred_gdf: gpd.GeoDataFrame,
                    ref_gdf: gpd.GeoDataFrame,
                    iou_threshold: float) -> MatchResult:
    """Match predicted polygons to reference polygons.

    Inputs must be in an equal-area CRS (precondition enforced below).

    OPEN DECISIONS (implementer's call):

    1. **Assignment algorithm.** Greedy-by-descending-IoU vs Hungarian
       optimal assignment. Both are valid; they can disagree when >=3
       polygons compete. Write a test that pins the disagreement case.

    2. **What counts as a merge / split.** Multiple defensible
       definitions -- intersection IoU >= threshold, union IoU >=
       threshold, or intersection/pred_area >= 0.5. Pick one, document
       it, add a test.

    INVARIANT the implementer MUST preserve (Phase 3 correction to the
    Phase 2 recommendation):

        the ASSIGNMENT is order-independent -- match_instances(A, B)
        and match_instances(B, A) produce the same set of matched pairs
        (modulo swapping the two indices).

        The metrics DERIVED from the assignment (precision, recall,
        merge_rate, split_rate) are NOT symmetric on purpose. Forcing
        metric-level symmetry would delete the pred-vs-ref asymmetry
        that defines merge and split events, which is the phenomenon
        this benchmark exists to study.

    See docs/metric-decisions.md section 1 for the full rationale.

    Raises:
        NotImplementedError until an implementer picks and codes the
        above.
    """
    assert_equal_area_crs(pred_gdf, what="match_instances(pred_gdf)")
    assert_equal_area_crs(ref_gdf,  what="match_instances(ref_gdf)")
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    raise NotImplementedError(
        "match_instances is reserved for the project owner. See "
        "src/bfbench/metrics/instance.py and docs/metric-decisions.md "
        "section 1 for the invariant it must preserve."
    )


def instance_prf(match_result: MatchResult) -> dict[str, float]:
    """Precision, recall, F1 from a MatchResult.

    OPEN DECISIONS (implementer's call):

    1. **How to score a 1-to-3 merge.** (1 TP + 2 FN + 0 FP) vs
       (0 TP + 1 FP + 3 FN) vs (3 TP + 0 FP + 0 FN). Different
       conventions -> different F1. Pick one, document it, add a
       parametrised test.

    2. **Precision on empty predictions.** 0/0 = NaN vs 1.0 vs 0.0.
       Pick one; write a test.

    3. **F1 when precision + recall = 0.** 0.0 (common) vs NaN.

    Metric-level symmetry is deliberately NOT required. See
    docs/metric-decisions.md section 1 and the test file for why.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "instance_prf is reserved for the project owner. Depends on "
        "match_instances first."
    )


def merge_rate(match_result: MatchResult) -> float:
    """Fraction of predictions that cover >= 2 reference buildings.

    Stubbed pending the final MatchResult design. The MatchResult
    structure may change when match_instances is implemented, so
    coding merge_rate against the current draft would risk needing to
    be redone.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "merge_rate is stubbed pending final MatchResult design."
    )


def split_rate(match_result: MatchResult) -> float:
    """Fraction of reference buildings covered by >= 2 predictions.

    Stubbed pending the final MatchResult design.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "split_rate is stubbed pending final MatchResult design."
    )
