"""Instance-level matching + precision/recall/F1 + merge/split rates.

STUBS ONLY. Every function raises NotImplementedError with a docstring
that names the open decisions the implementer must make. Do NOT
implement here.
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
            predictions that intersected >= 2 references at IoU >=
            threshold (a "1 prediction covers many references" event).
            The scoring convention for these events is the implementer's
            call -- see the merge_rate docstring.
        split_groups: list of (ref_idx, [pred_idx, ...]) tuples for
            references that were covered by >= 2 predictions (a "1 ref
            was fragmented into many predictions" event).
        equal_area_epsg: the equal-area CRS the geometries were
            projected into for area/IoU computation.
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

    Inputs:
        pred_gdf: predicted building polygons. Must be in an equal-area
            CRS (assertion enforced below); if it isn't, reproject
            BEFORE calling this function -- doing it inside the metric
            would hide the fact that the caller passed the wrong units.
        ref_gdf: reference building polygons, same CRS requirement.
        iou_threshold: float in (0, 1]. Pairs with IoU >= threshold are
            eligible for matching. 0.5 is the common convention (Pascal
            VOC); a lower threshold rewards near-misses.

    Returns:
        MatchResult (see the dataclass docstring for field semantics).

    Units:
        IoU is unitless; internally uses areas in m2 (equal-area CRS).

    OPEN DECISIONS the implementer must make and record:

    1. **Assignment algorithm.**
       - Greedy-by-descending-IoU: sort candidate pairs by IoU
         descending, greedily consume the highest available IoU pair
         first, remove both from further consideration, repeat. O(P R
         log(PR)). Yields a valid matching but NOT necessarily the
         globally maximum total IoU when three or more polygons
         compete for the same partners.
       - Hungarian (optimal assignment): build a P x R cost matrix
         (cost = 1 - IoU or +inf when IoU < threshold), solve the
         linear assignment problem for the minimum total cost. O(P R
         min(P, R)) with scipy.optimize.linear_sum_assignment. Yields
         the maximum total IoU across all valid assignments.
       - Decision required: pick one, and record which. Write a test
         where the two disagree (e.g. one prediction is a good match
         for either of two references but greedy picks the wrong pair).

    2. **Symmetry.**
       - Is match_instances(A, B) == match_instances(B, A) up to a swap
         of matched pair indices? For greedy assignment the answer
         depends on tie-breaking; for Hungarian it depends on whether
         cost is 1 - IoU (symmetric) or e.g. -log(IoU) (still symmetric)
         or something with a pred-vs-ref asymmetry.
       - Recommend: symmetric. Assert it in a test.

    3. **What counts as a merge / split.**
       - Definition of a "merge": one prediction whose intersection
         with each of >= 2 distinct references has IoU >= threshold,
         OR IoU with the union >= threshold, OR each single-ref
         intersection has area / prediction area >= 0.5? Different
         definitions give different rates; the implementer must pick
         one and document it in this docstring.
       - Same question for splits, mirrored.

    Raises:
        NotImplementedError until an implementer picks and codes the
        above.
    """
    assert_equal_area_crs(pred_gdf, what="match_instances(pred_gdf)")
    assert_equal_area_crs(ref_gdf,  what="match_instances(ref_gdf)")
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    raise NotImplementedError(
        "match_instances: assignment algorithm (greedy vs Hungarian) and "
        "the merge/split definition are not chosen. See docstring."
    )


def instance_prf(match_result: MatchResult) -> dict[str, float]:
    """Precision, recall, F1 from a MatchResult.

    Returns a dict with keys "precision", "recall", "f1", each a float.

    OPEN DECISIONS:

    1. **How to score a 1-to-3 merge.** With one prediction covering
       three references, is that:
       - 1 TP + 2 FN + 0 FP  ("we found the neighbourhood, missed 2/3"),
       - 0 TP + 3 FN + 1 FP  ("we didn't find any of the individual
         buildings; that big blob is a false positive"),
       - 3 TP + 0 FN + 0 FP  ("we found each reference at least
         partly"),
       or something else? Recall depends STRONGLY on this. Pick a rule
       and write both parametrised test cases (the tests file already
       does that -- verify the parametrisation).

    2. **Precision when there are 0 predictions.** Convention:
       precision = 1.0 (vacuously; no false positives) OR NaN (undefined)
       OR 0.0 (nothing found -> no precision to speak of). Pick one,
       write a test.

    3. **F1 when precision + recall = 0.** Return 0.0 (common) or NaN?
       Consequences for stratified plots (bins with no predictions).

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "instance_prf: merge scoring convention + empty-set precision + "
        "F1 degenerate case not chosen. See docstring."
    )


def merge_rate(match_result: MatchResult) -> float:
    """Fraction of predictions that cover >= 2 reference buildings.

    Definition of "cover": as chosen in match_instances (see its
    OPEN DECISIONS block, item 3). Returned value is in [0, 1].

    OPEN DECISIONS:
    1. **Denominator.** Fraction of ALL predictions, or fraction of
       matched predictions? The former is more comparable across
       recall levels; the latter is what practitioners usually mean
       by "merge rate".
    2. **How to weight merges of 3, 4, ... references.** Same weight
       as a 2-way merge, or scaled by (n_refs - 1)? Different
       conventions -> different intercomparability across AOIs.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "merge_rate: denominator + weighting convention not chosen. See docstring."
    )


def split_rate(match_result: MatchResult) -> float:
    """Fraction of reference buildings covered by >= 2 predictions.

    Same OPEN DECISIONS structure as merge_rate, mirrored on the
    reference side.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError(
        "split_rate: denominator + weighting convention not chosen. See docstring."
    )
