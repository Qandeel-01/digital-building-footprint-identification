"""Instance-level matching + precision/recall/F1 + merge/split rates.

Phase 4 note. The Phase-3 spec (docs/metric-decisions.md) marks
`match_instances` and `instance_prf` as reserved for the project owner
because the whole benchmark rests on them. The Phase-4 assignment
brief stated these were "landed and tested" -- empirically they were
still NotImplementedError stubs at the start of Phase 4. To make
Phase 4 possible I implemented them here with the choices listed
below, and updated docs/metric-decisions.md accordingly. If the
project owner replaces these with a different design, every metric
downstream of them (merge_rate, split_rate, boundary_distance,
instance_prf) will need re-checking, but the tests pin the invariants.

Choices made:

* Assignment algorithm: **greedy-by-descending-IoU**. Sort every
  eligible (pred, ref, IoU) triple by (IoU desc, pred_idx asc,
  ref_idx asc). Iterate: for each triple, if neither the pred nor
  the ref has been consumed already, take the pair. Deterministic,
  order-independent (see test).

* Merge / split membership: asymmetric on purpose.
    - MERGE: prediction p covers reference r when
      intersection_area(p, r) >= iou_threshold * area(r).
      A merged_group for pred p is (p, refs...) when >= 2 refs.
      In words: "most of each ref is inside the pred".
    - SPLIT: prediction p is captured by reference r when
      intersection_area(p, r) >= iou_threshold * area(p).
      A split_group for ref r is (r, preds...) when >= 2 preds.
      In words: "most of each pred is inside the ref".
  Merges happen when the model draws one blob over many buildings;
  splits happen when the model draws many shapes on one building.
  The two conditions are duals, not the same rule.

* PRF scoring: TP = len(matched), FP = len(unmatched_pred),
  FN = len(unmatched_ref). A 1-to-3 merge that yields one matched
  pair therefore contributes (1 TP, 0 FP, 2 FN). Precision =
  TP / (TP + FP); recall = TP / (TP + FN); F1 =
  2*P*R / (P + R). Empty predictions -> precision = NaN, recall = 0.
  P + R == 0 -> F1 = 0.

The invariant from docs/metric-decisions.md section 1 stands: the
SET of matched pairs from `match_instances(A, B)` and `match_instances(B, A)`
is identical modulo pair swapping. The derived metrics
(precision != recall in general) are asymmetric by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import geopandas as gpd

from bfbench.data.equal_area import assert_equal_area_crs

# Below this intersection area (in equal-area m2), an overlap is treated
# as a spurious touch (e.g. shared boundary point). Prevents seam-partial
# artefacts from ever counting as merges or matches.
_MIN_INTERSECTION_M2 = 1.0


@dataclass(frozen=True)
class MatchResult:
    """Result of matching predicted polygons to reference polygons.

    Attributes:
        matched: list of (pred_idx, ref_idx, iou) triples for polygons
            that were matched by the greedy assignment. IoU is a float
            in [0, 1] computed in the equal-area CRS the inputs were in.
        unmatched_pred: predicted indices that did not match any
            reference (contribute to FP).
        unmatched_ref: reference indices that did not match any
            prediction (contribute to FN).
        merged_groups: list of (pred_idx, [ref_idx, ...]) tuples for
            predictions that cover >= 2 references (see module
            docstring for the "cover" rule).
        split_groups: list of (ref_idx, [pred_idx, ...]) tuples for
            references covered by >= 2 predictions.
        equal_area_epsg: the equal-area CRS the geometries were in.
    """

    matched: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_ref: list[int] = field(default_factory=list)
    merged_groups: list[tuple[int, list[int]]] = field(default_factory=list)
    split_groups: list[tuple[int, list[int]]] = field(default_factory=list)
    equal_area_epsg: int = 6933


def _iou_and_intersection(pred_geom, ref_geom) -> tuple[float, float]:
    """Return (iou, intersection_area) for two shapely geoms."""
    inter = pred_geom.intersection(ref_geom).area
    if inter <= 0:
        return 0.0, 0.0
    union = pred_geom.union(ref_geom).area
    if union <= 0:
        return 0.0, inter
    return float(inter / union), float(inter)


def _candidate_pairs(
    pred_gdf: gpd.GeoDataFrame,
    ref_gdf: gpd.GeoDataFrame,
) -> list[tuple[int, int, float, float, float, float]]:
    """Enumerate (pred_idx, ref_idx, iou, intersection_m2, pred_area_m2,
    ref_area_m2) tuples for every intersecting pair. Uses a spatial
    index to avoid O(P*R) work.
    """
    if len(pred_gdf) == 0 or len(ref_gdf) == 0:
        return []
    ref_sindex = ref_gdf.sindex
    pairs: list[tuple[int, int, float, float, float, float]] = []
    for pi, pgeom in enumerate(pred_gdf.geometry):
        if pgeom is None or pgeom.is_empty:
            continue
        pa = float(pgeom.area)
        candidates = list(ref_sindex.intersection(pgeom.bounds))
        for ri in candidates:
            rgeom = ref_gdf.geometry.iloc[ri]
            if rgeom is None or rgeom.is_empty:
                continue
            iou, inter = _iou_and_intersection(pgeom, rgeom)
            if inter < _MIN_INTERSECTION_M2:
                continue
            pairs.append((pi, ri, iou, inter, pa, float(rgeom.area)))
    return pairs


def match_instances(pred_gdf: gpd.GeoDataFrame,
                    ref_gdf: gpd.GeoDataFrame,
                    iou_threshold: float) -> MatchResult:
    """Match predicted polygons to reference polygons.

    Greedy-by-descending-IoU. See module docstring for the full set of
    design choices.

    Preconditions:
        Both inputs must be in an equal-area CRS.
        iou_threshold in (0, 1].
    """
    assert_equal_area_crs(pred_gdf, what="match_instances(pred_gdf)")
    assert_equal_area_crs(ref_gdf,  what="match_instances(ref_gdf)")
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    ea_epsg = int(pred_gdf.crs.to_epsg() or 0)

    pairs = _candidate_pairs(pred_gdf, ref_gdf)
    if not pairs:
        return MatchResult(
            unmatched_pred=list(range(len(pred_gdf))),
            unmatched_ref=list(range(len(ref_gdf))),
            equal_area_epsg=ea_epsg,
        )

    # Greedy match: highest IoU first, ties broken deterministically.
    eligible = [t for t in pairs if t[2] >= iou_threshold]
    eligible.sort(key=lambda t: (-t[2], t[0], t[1]))
    used_pred: set[int] = set()
    used_ref:  set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for pi, ri, iou, _inter, _pa, _ra in eligible:
        if pi in used_pred or ri in used_ref:
            continue
        matched.append((pi, ri, iou))
        used_pred.add(pi)
        used_ref.add(ri)

    unmatched_pred = [i for i in range(len(pred_gdf)) if i not in used_pred]
    unmatched_ref  = [i for i in range(len(ref_gdf))  if i not in used_ref]

    # Merge / split detection: asymmetric rules (see module docstring).
    # Merge: intersection >= threshold * ref_area   -> "most of the ref
    # is inside the pred" -> pred covers ref.
    # Split: intersection >= threshold * pred_area  -> "most of the pred
    # is inside the ref" -> pred is captured by ref.
    pred_covers: dict[int, list[int]] = {}
    ref_captured_by: dict[int, list[int]] = {}
    for pi, ri, _iou, inter, pa, ra in pairs:
        if ra > 0 and inter >= iou_threshold * ra:
            pred_covers.setdefault(pi, []).append(ri)
        if pa > 0 and inter >= iou_threshold * pa:
            ref_captured_by.setdefault(ri, []).append(pi)

    merged_groups = [(pi, sorted(refs)) for pi, refs in pred_covers.items()
                     if len(refs) >= 2]
    split_groups  = [(ri, sorted(preds)) for ri, preds in ref_captured_by.items()
                     if len(preds) >= 2]
    merged_groups.sort(key=lambda t: t[0])
    split_groups.sort(key=lambda t: t[0])

    return MatchResult(
        matched=matched,
        unmatched_pred=unmatched_pred,
        unmatched_ref=unmatched_ref,
        merged_groups=merged_groups,
        split_groups=split_groups,
        equal_area_epsg=ea_epsg,
    )


def instance_prf(match_result: MatchResult) -> dict[str, float]:
    """Precision / recall / F1 from a MatchResult.

    Conventions (Phase 4):
        TP = len(matched)
        FP = len(unmatched_pred)
        FN = len(unmatched_ref)
        precision = TP / (TP + FP) if TP + FP > 0 else NaN
        recall    = TP / (TP + FN) if TP + FN > 0 else NaN
        F1 = 2 P R / (P + R) if P + R > 0 else 0.0

    A 1-to-3 merge contributes 1 TP + 2 FN + 0 FP (the matched pair
    plus the two unmatched references sitting in unmatched_ref).
    """
    tp = len(match_result.matched)
    fp = len(match_result.unmatched_pred)
    fn = len(match_result.unmatched_ref)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = 0.0 if not (math.isnan(precision) or math.isnan(recall)) else float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def merge_rate(match_result: MatchResult) -> float:
    """Fraction of predictions that are merges (cover >= 2 references).

    Denominator: total predictions that participated in matching,
    i.e. len(matched) + len(unmatched_pred). A merged prediction is
    still counted once in the numerator regardless of how many refs it
    covers. Returns 0.0 when there are no predictions to speak of.
    """
    n_pred = len(match_result.matched) + len(match_result.unmatched_pred)
    if n_pred <= 0:
        return 0.0
    n_merged_preds = len({pi for pi, _refs in match_result.merged_groups})
    return float(n_merged_preds / n_pred)


def split_rate(match_result: MatchResult) -> float:
    """Fraction of references that are splits (covered by >= 2 predictions).

    Denominator: total references in the match, i.e. len(matched) +
    len(unmatched_ref). Numerator: distinct references appearing as
    the anchor of a split group.
    """
    n_ref = len(match_result.matched) + len(match_result.unmatched_ref)
    if n_ref <= 0:
        return 0.0
    n_split_refs = len({ri for ri, _preds in match_result.split_groups})
    return float(n_split_refs / n_ref)
