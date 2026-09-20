# Metric-module design decisions (Phase 3 spec + Phase 4 implementations)

This file records the calls the metric module makes as a benchmark
design choice, especially the ones I was asked to reverse from the
Phase 2 draft. Every decision here is enforced by both the docstring
and the test.

**Phase 4 note.** Sections 1 (matching design) and 6 (Chamfer analytic)
were amended in Phase 4:

* The Phase-4 assignment stated `match_instances` and `instance_prf`
  were "landed and tested" by the project owner. Empirically the
  stubs were still `NotImplementedError` at the start of Phase 4.
  To make Phase 4 possible I (Claude) implemented them with the
  choices listed in section 1 below and in the docstring of
  `src/bfbench/metrics/instance.py`. If the project owner replaces
  these with a different design, everything downstream (merge_rate,
  split_rate, boundary_distance, instance_prf) needs re-checking;
  the tests pin the invariants.

* Section 6 quoted the symmetric Chamfer of two 10x10 m squares
  offset by 5 m as 3.125 m. That count missed the pred-right-edge
  case (the right edge of the pred lies INSIDE the ref, so its
  distance to the ref boundary is min(5, y, 10-y), not 5). The
  correct answer is 2.5 m. Fixed in-place below.

## 1. `match_instances` — asymmetric metrics, order-independent assignment

Phase 2 recommended metric-level symmetry:
"match_instances(A, B) == match_instances(B, A)".

That is wrong.

Precision runs prediction -> reference (how many predictions have a
matching reference), recall runs reference -> prediction (how many
references have a matching prediction), and merge / split rates are
DEFINED by the asymmetry between the two directions. A one-prediction-
covers-three-references event is only visible from the pred side; from
the ref side it looks like three refs each matched by the same pred.
Forcing symmetry deletes the ability to measure merge/split at all,
which is the phenomenon this benchmark exists to study.

The correct invariant is narrower and lives at the assignment step:

    the SET of matched pairs must be independent of the input order.
    match_instances(A, B).matched, treated as {(p_idx, r_idx)}, must
    equal match_instances(B, A).matched with the two indices swapped.
    The metrics DERIVED from that matched set may then be asymmetric
    on purpose (precision vs recall).

Test: `test_match_instances_assignment_is_order_independent` locks in
the assignment invariant. There is deliberately no
`test_match_instances_metrics_are_symmetric`; a comment in the test
file explains why.

**Phase 4 implementation choices** (Claude, in the absence of an
owner-authored implementation):

* Assignment algorithm: **greedy-by-descending-IoU**. Every eligible
  pair (IoU >= threshold) is sorted by `(iou desc, pred_idx asc,
  ref_idx asc)` and consumed in order. Deterministic and
  order-independent under the SET-of-pairs invariant above.

* Merge/split coverage rules are **asymmetric on purpose**:
    - MERGE (pred covers refs): `intersection >= threshold * ref_area`
      -- "most of each ref is inside the pred". Two or more such refs
      per pred trigger a merged_group.
    - SPLIT (multiple preds captured by one ref):
      `intersection >= threshold * pred_area` -- "most of each pred is
      inside the ref". Two or more such preds per ref trigger a
      split_group.
  These are dual conditions, not the same rule. A merge with three
  small refs under a large pred is not the same shape as a split with
  three small preds under a large ref.

* PRF scoring: TP = matched, FP = unmatched_pred, FN = unmatched_ref.
  A 1-to-3 merge yielding one matched pair contributes 1 TP + 2 FN.
  precision = TP / (TP + FP) if any pred, else NaN.
  recall    = TP / (TP + FN) if any ref, else NaN.
  F1 = 2 P R / (P + R) if P + R > 0, else 0.
  Empty predictions -> precision = NaN + F1 = NaN, recall = 0.

* Merge / split rate denominators:
    - `merge_rate = |{p : p in a merged_group}| / (n_matched + n_unmatched_pred)`
    - `split_rate = |{r : r in a split_group}| / (n_matched + n_unmatched_ref)`

* Minimum intersection area to consider a pair: 1 m2. Below this we
  treat the overlap as a shared-boundary artefact (dedup seam, touch)
  rather than a real overlap. Prevents seam-partial ghosts from
  showing up as matches or as merges.

## 2. Rasterisation — centre-only default, all-touched exposed for sensitivity

Phase 2 recommended all-touched. That was wrong for this study.

all-touched rasterisation inflates polygon area in a size-dependent
way. Consider a square with side `s` at pixel resolution `p`:
    - all-touched area  ≈ (s + p)^2       (perimeter grows one pixel)
    - centre-only area  ≈ s^2             (no inflation)

Relative inflation is `((s + p)^2 - s^2) / s^2 = 2p/s + (p/s)^2`.
At `p = 0.5 m`:

| Building side s | Building area | Inflation from all-touched |
|-----------------|---------------|----------------------------|
|  5 m            |   25 m2       | 21.0 %                      |
|  9 m            |   80 m2       | 11.6 %                      |
| 15 m            |  225 m2       |  6.9 %                      |
| 32 m            | ~1000 m2      |  3.2 %                      |

Since one of the stratified plots is F1 vs building-size decile, an
all-touched default writes a monotonic bias directly into the study
axis: small buildings score better with all-touched not because the
model finds them better, but because the pixel accounting inflates
their union more than their intersection.

Decision: `pixel_iou(..., all_touched=False)` is the default. The
harness runs a second sensitivity pass with `all_touched=True` and
emits both into the results Parquet. `stratify_by_area` can then
diff them and report the size-dependent bias explicitly instead of
hiding it.

`tests/test_metrics.py::test_pixel_iou_all_touched_inflates_area_size_dependently`
measures this on synthetic squares at 5 / 9 / 15 / 32 m sides and
locks in the observed inflation, so the bias is documented in code
rather than only in this file.

The Phase 2 claim that "SpaceNet labels are pre-dilated relative to
imagery, so all-touched compensates" is unsupported. I could not find
a citation in either the SpaceNet 2 paper (Van Etten et al., 2018) or
the AWS Open Data README. It is deleted.

## 3. `pixel_iou` — raster interface, not vector

Phase 2 signature:
    `pixel_iou(pred_gdf, ref_gdf, bounds, resolution) -> float`.

That signature forces the pipeline through
`raster (model output) -> vector (abd vectorize) -> raster (pixel_iou)`,
which compounds discretisation error twice. The model's native output
is a per-pixel probability mask. The right pixel-IoU compares that
mask against a rasterised reference at the same grid.

New signature:

    pixel_iou(
        pred_prob:  np.ndarray | Path,      # HxW probability or mask
        ref_gdf:    gpd.GeoDataFrame,       # reference polygons (equal-area)
        bounds:     tuple[float, ...],      # (left, bottom, right, top) in EA CRS
        resolution: float,                  # metres/pixel; must match pred_prob's grid
        threshold:  float = 0.5,            # sigmoid threshold if pred_prob is prob
        all_touched: bool = False,          # snapping mode for reference rasterisation
    ) -> float

Rules:
    - If `pred_prob` is a Path, it's a PNG/TIFF read via rasterio.
    - If it's a ndarray, it must already be aligned with `bounds` +
      `resolution` (caller reprojects if needed).
    - The reference is rasterised inside the metric at the same grid,
      using `all_touched` for snapping.
    - The metric does NOT reproject the reference. It asserts equal-area
      CRS and matching bounds.

Instance metrics (`match_instances`, `instance_prf`, `merge_rate`,
`split_rate`, `boundary_distance`) keep working on vector polygons.
The two categories are complementary, not redundant.

## 4. `stratify_by_density` — no default window

`window_m` used to default to 500. That silently hid the fact that
"local density" is not a natural constant.

Decision: `window_m` is required. The harness runs a fixed sweep at
`window_m ∈ {100, 200, 500}` and writes all three into the results
Parquet as separate rows with a `window_m` column. Robustness to
window choice is now measurable, not assumed.

## 5. `count_error` — no per-tile relative error

Per-tile relative error `(n_pred - n_ref) / n_ref` is undefined at
`n_ref = 0` and noise-dominated at small `n_ref`. Filtering
`n_ref >= 5` (the Phase 2 proposal) introduces a selection effect
that removes exactly the tiles that are hardest to detect (sparse
suburban / semi-rural).

Decision: `count_error(pred_gdf, ref_gdf) -> {abs_error, signed_error}`.
Per-tile. Relative error is a separate aggregate-only helper:

    count_error_aoi(rows: pd.DataFrame) -> {abs_error, signed_error,
                                            relative_error}

where the AOI-level relative error is `sum(signed_error) / sum(n_ref)`
across all tiles — noise averages out and the divisor is never zero
for a benchmark run that has any references at all.

## 6. `boundary_distance` — densified sampling

Vertex-only Chamfer / Hausdorff measures how many vertices the
vectoriser emitted per polygon, which is a property of the
Douglas-Peucker step upstream, not of the geometry. The Phase 2 test
assumed vertex-only implicitly.

Decision: `boundary_distance(match_result, sample_step_m: float = 0.5)`.
Every polygon boundary is resampled at `sample_step_m` intervals
before the point-set metric.

Fixture math: two `10 x 10 m` squares whose left edges are 5 m apart.
Pred = [0,10] x [0,10]. Ref = [5,15] x [0,10].

    - Pred LEFT edge (x=0, y in [0,10]): nearest ref boundary is ref
      LEFT edge (5, y), distance 5. Mean over 10 m of edge: 5.
    - Pred RIGHT edge (x=10, y in [0,10]): (10, y) lies INSIDE ref, so
      the nearest boundary points are ref's LEFT edge at distance 5,
      ref's TOP corner (10, 10) at distance |y - 10|, or ref's BOTTOM
      corner (10, 0) at distance |y|. Min over y in [0,10] is
      min(5, y, 10-y). Mean = int_0^5 y dy / 10 + int_5^10 (10-y) dy / 10
      = 12.5/10 + 12.5/10 = 2.5.
    - Pred TOP edge (y=10, x in [0,10]): for x in [0, 5) the nearest
      ref point is (5, 10), distance 5 - x. For x in [5, 10] pred TOP
      and ref TOP coincide, distance 0. Mean = int_0^5 (5-x) dx / 10 =
      12.5/10 = 1.25.
    - Pred BOTTOM edge: same as TOP by symmetry. Mean = 1.25.

    Mean pred->ref = (5*10 + 2.5*10 + 1.25*10 + 1.25*10) / 40
                   = (50 + 25 + 12.5 + 12.5) / 40 = 2.5 m.
    By symmetry mean ref->pred = 2.5 m.
    **Symmetric Chamfer = 2.5 m** at infinitesimal sampling.

    Symmetric Hausdorff on the same fixture is 5 m (the max distance
    from any boundary point to the nearest boundary point of the other
    polygon; realised at e.g. pred bottom-left corner (0, 0) whose
    nearest ref boundary point is (5, 0), distance 5).

At `sample_step_m = 0.5`, the discretisation error is small (< 5 %),
so the test tolerance is 5 %.

(The Phase-2 draft of this section quoted 3.125 m; that count missed
the pred RIGHT edge lying INSIDE the ref and thus having distance
min(5, y, 10-y) rather than 5. Fixed here.)

## 7. Reserved-stub meta-test (superseded in Phase 4)

Phase 3 kept a meta-test that pinned every reserved metric to
`NotImplementedError` specifically. In Phase 4 every reserved metric
was implemented, so the meta-test was replaced by real per-function
assertions (see `tests/test_metrics.py`). The distinction the
Phase-3 meta-test protected (fail-with-NotImplementedError, not
fail-with-ImportError) is preserved indirectly by each function's
own tests exercising real inputs and asserting real outputs -- an
import error would surface at collection time.
