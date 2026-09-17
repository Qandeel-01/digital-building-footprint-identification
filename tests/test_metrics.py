"""Failing tests for the bfbench.metrics module.

Design contract (per Phase 2 Task 2):

    - Every test here MUST FAIL on a clean checkout, because every metric
      function is a stub raising NotImplementedError. The failure MUST
      be a NotImplementedError -- not an ImportError, not a fixture
      collection error, not a NameError.

    - Expected values are computed BY HAND in the fixture docstrings
      with arithmetic shown. Do not compute them by calling a reference
      library inside the test: that would move the ambiguity into the
      library, not out of it.

    - Merge / split scoring conventions are represented as separate
      parametrised tests. The implementer picks a convention by leaving
      the corresponding test's assertion unchanged (so it passes) and
      changing (or deleting) the others; the parametrisation records
      the trade-off explicitly rather than hiding it in the assertion.

    - `test_stubs_fail_with_notimplemented_not_something_else` is the
      meta-test the Task 2 brief asks for: it verifies every stub is
      importable and its failure mode is NotImplementedError specifically,
      so we know the whole suite's failure is really "not implemented"
      rather than "the tests are broken".

The suite is verified to fail on clean checkout by running
    pytest tests/test_metrics.py
in a fresh clone with `bfbench` installed. Every assertion below
raises NotImplementedError until the corresponding stub is filled in.
"""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from bfbench.data.equal_area import (
    DEFAULT_EQUAL_AREA_EPSG,
    NonEqualAreaCRSError,
)
from bfbench.metrics import (
    MatchResult,
    boundary_distance,
    count_error,
    instance_prf,
    match_instances,
    merge_rate,
    pixel_iou,
    split_rate,
    stratify_by_area,
    stratify_by_density,
)

EA = f"EPSG:{DEFAULT_EQUAL_AREA_EPSG}"


# ---------------------------------------------------------------------------
# Fixture helpers -- geometry with by-hand-computable areas.
# ---------------------------------------------------------------------------

def _square(x0: float, y0: float, side: float) -> Polygon:
    """Axis-aligned square in the equal-area CRS. Area = side*side m2."""
    return Polygon([(x0, y0), (x0 + side, y0),
                    (x0 + side, y0 + side), (x0, y0 + side)])


def _gdf(polys: list[Polygon]) -> gpd.GeoDataFrame:
    """Wrap polygons in a GeoDataFrame in the equal-area CRS."""
    return gpd.GeoDataFrame(geometry=polys, crs=EA)


# ---------------------------------------------------------------------------
# match_instances -- IoU semantics
# ---------------------------------------------------------------------------

def test_match_instances_two_squares_50pct_overlap_known_iou():
    """Two 10x10 m squares overlapping by exactly 50%.

    Arithmetic (in the equal-area CRS, units = m2):
        pred:  square at (0, 0), side 10 -> area  100 m2
        ref:   square at (5, 0), side 10 -> area  100 m2
        inter: rectangle 5 wide x 10 tall -> area   50 m2
        union: 100 + 100 - 50             -> area  150 m2
        IoU = 50 / 150 = 0.3333... exactly 1/3.

    Assertion: match_instances must return exactly one matched pair
    (0, 0) with IoU within a small tolerance of 1/3.
    """
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(5.0, 0.0, 10.0)])

    result = match_instances(pred, ref, iou_threshold=0.25)

    assert len(result.matched) == 1, "expected exactly one matched pair"
    p_idx, r_idx, iou = result.matched[0]
    assert p_idx == 0 and r_idx == 0
    assert math.isclose(iou, 1.0 / 3.0, rel_tol=1e-6, abs_tol=1e-9), (
        f"expected IoU=1/3 exactly, got {iou}"
    )
    assert result.unmatched_pred == []
    assert result.unmatched_ref == []


def test_match_instances_identical_sets_iou_one():
    """Identical single polygon on both sides.

    Arithmetic:
        pred == ref == square at (0, 0), side 10 -> area 100 m2
        inter = union = 100 m2
        IoU = 100 / 100 = 1.0
    """
    poly = _square(0.0, 0.0, 10.0)
    pred = _gdf([poly])
    ref  = _gdf([poly])

    result = match_instances(pred, ref, iou_threshold=0.5)

    assert len(result.matched) == 1
    _, _, iou = result.matched[0]
    assert math.isclose(iou, 1.0, abs_tol=1e-9)
    assert result.unmatched_pred == []
    assert result.unmatched_ref == []


def test_match_instances_disjoint_sets_no_match():
    """Two disjoint squares (100 m apart) -> no IoU, no matches.

    Arithmetic:
        intersection = 0 m2 -> IoU = 0. Below any reasonable threshold.
        The pred polygon is unmatched (FP), the ref polygon is unmatched (FN).
    """
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(100.0, 0.0, 10.0)])

    result = match_instances(pred, ref, iou_threshold=0.5)

    assert result.matched == []
    assert result.unmatched_pred == [0]
    assert result.unmatched_ref == [0]


def test_match_instances_rejects_wrong_crs():
    """A GeoDataFrame in EPSG:4326 (lat/lon) must be rejected BEFORE we
    even reach the assignment step -- this guards the equal-area
    precondition and must NOT depend on the metric's implementation.
    This test is expected to pass on clean checkout: it exercises the
    precondition, not the (unimplemented) core.
    """
    pred = _gdf([_square(0.0, 0.0, 10.0)]).to_crs("EPSG:4326")
    ref  = _gdf([_square(5.0, 0.0, 10.0)]).to_crs("EPSG:4326")
    with pytest.raises(NonEqualAreaCRSError):
        match_instances(pred, ref, iou_threshold=0.5)


def test_match_instances_rejects_threshold_out_of_range():
    """Also a precondition test that passes on clean checkout."""
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    with pytest.raises(ValueError):
        match_instances(pred, ref, iou_threshold=1.5)
    with pytest.raises(ValueError):
        match_instances(pred, ref, iou_threshold=0.0)


# ---------------------------------------------------------------------------
# instance_prf
# ---------------------------------------------------------------------------

def test_prf_identical_sets_perfect():
    """One matched pair, no unmatched.

    Arithmetic:
        TP=1, FP=0, FN=0
        precision = 1 / (1 + 0) = 1.0
        recall    = 1 / (1 + 0) = 1.0
        F1        = 2 * 1 * 1 / (1 + 1) = 1.0
    """
    mr = MatchResult(matched=[(0, 0, 1.0)],
                     unmatched_pred=[], unmatched_ref=[])
    prf = instance_prf(mr)
    assert math.isclose(prf["precision"], 1.0, abs_tol=1e-9)
    assert math.isclose(prf["recall"], 1.0, abs_tol=1e-9)
    assert math.isclose(prf["f1"], 1.0, abs_tol=1e-9)


def test_prf_disjoint_sets_zero():
    """No matches, one FP, three FN.

    Arithmetic:
        TP=0, FP=1, FN=3
        precision = 0 / (0 + 1) = 0.0
        recall    = 0 / (0 + 3) = 0.0
        F1: precision+recall = 0 -> convention (open decision) is 0.0
    """
    mr = MatchResult(matched=[],
                     unmatched_pred=[0], unmatched_ref=[0, 1, 2])
    prf = instance_prf(mr)
    assert math.isclose(prf["precision"], 0.0, abs_tol=1e-9)
    assert math.isclose(prf["recall"], 0.0, abs_tol=1e-9)
    assert math.isclose(prf["f1"], 0.0, abs_tol=1e-9)


def test_prf_empty_predictions_precision_is_open_decision():
    """Zero predictions, three references.

    Arithmetic:
        TP=0, FP=0, FN=3
        recall = 0 / (0 + 3) = 0.0 -- unambiguous.
        precision = 0 / (0 + 0) -- 0/0. See instance_prf OPEN DECISION 2.

    Convention picked here (the implementer picks one; this test
    encodes NaN as the answer). If the implementer picks 1.0 or 0.0,
    they should replace `math.isnan(...)` with the appropriate
    `isclose(...)` and delete the corresponding sibling test.
    """
    mr = MatchResult(matched=[],
                     unmatched_pred=[], unmatched_ref=[0, 1, 2])
    prf = instance_prf(mr)
    assert math.isclose(prf["recall"], 0.0, abs_tol=1e-9)
    assert math.isnan(prf["precision"]), (
        "convention: precision on empty predictions is NaN. If the "
        "implementer chose 1.0 or 0.0 instead, update this assertion "
        "and record the choice in instance_prf's docstring."
    )


# ---------------------------------------------------------------------------
# Merge / split -- parametrised so the implementer picks a convention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("convention_label", "expected_tp_fp_fn"),
    [
        # 1 pred covers 3 refs. Three defensible (TP, FP, FN) triples.
        # The implementer picks ONE and leaves that parametrisation's
        # assertion; the others will still raise NotImplementedError
        # (until deleted) so the choice is a visible code-review event.
        ("1_tp_2_fn_0_fp", (1, 0, 2)),
        ("0_tp_1_fp_3_fn", (0, 1, 3)),
        ("3_tp_0_fp_0_fn", (3, 0, 0)),
    ],
    ids=["1TP2FN", "0TP1FP3FN", "3TP0FN"],
)
def test_1to3_merge_case_multiple_defensible_scorings(convention_label,
                                                     expected_tp_fp_fn):
    """One prediction (pred 0) covers three separate reference buildings
    (refs 0, 1, 2). The canonical high-IoU match is (0, 0); the merged
    group is recorded so instance_prf can score it.

    Which of the three (TP, FP, FN) triples above is 'correct' is the
    open decision. This parametrised test records ALL three so the
    implementer picks one by keeping its assertion and dropping the
    others.
    """
    mr = MatchResult(
        matched=[(0, 0, 0.6)],
        unmatched_pred=[],
        unmatched_ref=[1, 2],
        merged_groups=[(0, [0, 1, 2])],
    )
    prf = instance_prf(mr)
    tp, fp, fn = expected_tp_fp_fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if not math.isnan(precision):
        assert math.isclose(prf["precision"], precision, abs_tol=1e-9), (
            f"convention {convention_label}: expected precision "
            f"{precision} but got {prf['precision']}"
        )
    if not math.isnan(recall):
        assert math.isclose(prf["recall"], recall, abs_tol=1e-9), (
            f"convention {convention_label}: expected recall "
            f"{recall} but got {prf['recall']}"
        )


def test_split_case_3_predictions_on_1_reference():
    """Three predictions (0, 1, 2) all covering the same reference
    building (ref 0). The canonical match is (0, 0); the other two
    predictions belong to the split group.

    split_rate is expected to be in [0, 1]. This test just asserts the
    function runs and returns a finite value in [0, 1] on this input;
    the precise value depends on the denominator convention (open
    decision 1) and cannot be pinned without picking one.
    """
    mr = MatchResult(
        matched=[(0, 0, 0.4)],
        unmatched_pred=[1, 2],
        unmatched_ref=[],
        split_groups=[(0, [0, 1, 2])],
    )
    rate = split_rate(mr)
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0, f"split_rate must be in [0, 1], got {rate}"


def test_merge_rate_zero_on_clean_1to1_matching():
    """Clean 1-to-1 matching with no merge events -> merge_rate == 0.

    (This test does not care which denominator convention was picked;
    the merge rate on a MatchResult with zero merged_groups is zero
    under every convention.)
    """
    mr = MatchResult(matched=[(0, 0, 1.0)],
                     unmatched_pred=[], unmatched_ref=[],
                     merged_groups=[])
    assert math.isclose(merge_rate(mr), 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# pixel_iou
# ---------------------------------------------------------------------------

def test_pixel_iou_two_squares_50pct_overlap_converges_to_one_third():
    """Same 50%-overlap fixture as match_instances.

    Continuous IoU is 50 / 150 = 1/3 (see match_instances test above).
    Pixel IoU at any finite resolution differs from the continuous IoU
    by a rasterisation error that depends on OPEN DECISION 1
    (all-touched vs centre-only). At `resolution=0.5` m over a
    (0, 0, 20, 10) bounding box:
        - 20 / 0.5 = 40 pixels wide
        - 10 / 0.5 = 20 pixels tall
        - pred fills the left 10 m -> 20 pixels wide x 20 tall = 400 px
        - ref  fills 5..15 m       -> 20 pixels wide x 20 tall = 400 px
        - inter fills 5..10 m      -> 10 pixels wide x 20 tall = 200 px
        - union                    -> 400 + 400 - 200            = 600 px
        - pixel IoU = 200 / 600 = 1/3 EXACTLY under either snapping
          convention because the polygon edges lie on pixel boundaries.

    So this specific fixture gives 1/3 regardless of the open decision.
    """
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(5.0, 0.0, 10.0)])
    iou = pixel_iou(pred, ref, bounds=(0.0, 0.0, 20.0, 10.0),
                    resolution=0.5)
    assert math.isclose(iou, 1.0 / 3.0, rel_tol=1e-6, abs_tol=1e-6), (
        f"expected pixel IoU=1/3 with polygon edges on pixel boundaries, "
        f"got {iou}"
    )


def test_pixel_iou_rejects_wrong_crs():
    """Precondition test -- expected to pass on clean checkout."""
    pred = _gdf([_square(0.0, 0.0, 10.0)]).to_crs("EPSG:4326")
    ref  = _gdf([_square(5.0, 0.0, 10.0)]).to_crs("EPSG:4326")
    with pytest.raises(NonEqualAreaCRSError):
        pixel_iou(pred, ref, bounds=(0.0, 0.0, 20.0, 10.0), resolution=0.5)


def test_pixel_iou_rejects_nonpositive_resolution():
    """Precondition test -- expected to pass on clean checkout."""
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    with pytest.raises(ValueError):
        pixel_iou(pred, ref, bounds=(0.0, 0.0, 20.0, 10.0), resolution=0.0)


# ---------------------------------------------------------------------------
# count_error
# ---------------------------------------------------------------------------

def test_count_error_signed_and_absolute():
    """4 predictions, 6 references.

    Arithmetic:
        abs_error      = |4 - 6|          = 2
        signed_error   = 4 - 6            = -2
        relative_error = -2 / 6           = -1/3
    """
    pred = _gdf([_square(0.0, 0.0, 1.0)] * 4)
    ref  = _gdf([_square(0.0, 0.0, 1.0)] * 6)
    result = count_error(pred, ref)
    assert math.isclose(result["abs_error"],      2.0, abs_tol=1e-9)
    assert math.isclose(result["signed_error"],  -2.0, abs_tol=1e-9)
    assert math.isclose(result["relative_error"], -1.0 / 3.0,
                        rel_tol=1e-6, abs_tol=1e-9)


def test_count_error_empty_reference_is_open_decision():
    """0 predictions, 0 references. Relative error = 0/0.

    Convention encoded here: NaN. If the implementer picks a different
    convention (0.0, inf, |n_pred|), update this assertion and record
    the decision in count_error's docstring.
    """
    pred = _gdf([])
    ref  = _gdf([])
    result = count_error(pred, ref)
    assert math.isclose(result["abs_error"], 0.0, abs_tol=1e-9)
    assert math.isclose(result["signed_error"], 0.0, abs_tol=1e-9)
    assert math.isnan(result["relative_error"]), (
        "convention: relative error is NaN when n_ref == 0. "
        "See count_error OPEN DECISION 1."
    )


# ---------------------------------------------------------------------------
# boundary_distance
# ---------------------------------------------------------------------------

def test_chamfer_between_two_offset_squares_known_value():
    """Two 10 x 10 m squares whose LEFT edges are 5 m apart -- i.e. the
    same 50%-overlap fixture from earlier tests.

    Pred boundary: perimeter of a square at (0, 0)  -> (10, 10).
    Ref  boundary: perimeter of a square at (5, 0)  -> (15, 10).

    Derivation of symmetric Chamfer at fine boundary sampling:

        - Both squares have identical size and only differ by a 5 m
          horizontal shift.
        - Each square perimeter has length 40 m.

        For a point on the LEFT edge of pred (x=0, y in [0,10]) the
        nearest point on ref's boundary is on ref's LEFT edge (x=5,
        same y) at distance exactly 5. Contribution to the pred->ref
        mean: 5.

        For a point on the RIGHT edge of pred (x=10, y in [0,10]):
        the nearest ref boundary is ref's LEFT edge (x=5, same y),
        distance 5. But wait -- ref's LEFT edge is INSIDE pred's
        interior; the boundary of ref goes THROUGH pred. So the pred
        RIGHT edge's nearest point on ref boundary is at x=5 (ref
        left edge inside pred). Distance = 10 - 5 = 5. Contribution: 5.

        Actually reconsider more carefully: the pred RIGHT edge is at
        x=10. Ref boundary includes ref LEFT (x=5), ref RIGHT (x=15),
        ref TOP/BOTTOM (y in {0,10}). Distances from (10, y):
            to ref LEFT  (5,  y): |10-5|  = 5
            to ref RIGHT (15, y): |10-15| = 5
            to ref TOP/BOT (x=10, 0 or 10): 0 if y in {0,10} else >0
        Min = 5 for interior y, 0 for the two corners.
        Contribution: essentially 5.

        For a point on the TOP edge of pred (y=10, x in [0,10]):
        nearest ref boundary is ref's TOP edge (y=10, x in [5,15]).
            If x in [5, 10]: distance 0 (both boundaries coincide there).
            If x in [0, 5):  distance = 5 - x.
        Mean over x in [0, 10]:
            integral over [0,5] of (5-x) dx = [5x - x^2/2] from 0..5
                                            = 25 - 12.5 = 12.5
            integral over [5,10] of 0 dx    = 0
            length = 10
            mean = 12.5 / 10 = 1.25
        Contribution: 1.25.

        By symmetry the BOTTOM edge gives 1.25 as well.

        Overall mean pred->ref distance (weighted by edge lengths, each
        edge is 10 m and there are 4 edges = 40 m):
            mean = (10*5 + 10*5 + 10*1.25 + 10*1.25) / 40
                 = (50 + 50 + 12.5 + 12.5) / 40
                 = 125 / 40
                 = 3.125 m

        By full symmetry (the two squares are congruent up to shift),
        the ref->pred mean is also 3.125 m. So the symmetric Chamfer
        distance is 3.125 m.

    NOTE: the arithmetic above assumes fine sampling. Coarse vertex-only
    sampling gives a different (larger) mean. The test tolerates ~5%
    error to allow the implementer to pick a boundary sampling step.
    """
    mr = MatchResult(matched=[(0, 0, 1.0 / 3.0)],
                     unmatched_pred=[], unmatched_ref=[])
    result = boundary_distance(mr)
    assert "mean_chamfer" in result and "mean_hausdorff" in result
    assert math.isclose(result["mean_chamfer"], 3.125,
                        rel_tol=0.05, abs_tol=0.25), (
        f"expected mean_chamfer near 3.125 m under fine sampling, "
        f"got {result['mean_chamfer']}"
    )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def _tidy_results() -> pd.DataFrame:
    """Small tidy long-format frame with 4 tiles across 2 AOIs."""
    return pd.DataFrame([
        {"aoi": "V", "tile_z": 17, "tile_x": 1, "tile_y": 1,
         "metric": "f1", "value": 0.6, "median_ref_area_m2": 80.0,
         "n_ref_buildings": 12},
        {"aoi": "V", "tile_z": 17, "tile_x": 2, "tile_y": 1,
         "metric": "f1", "value": 0.4, "median_ref_area_m2": 40.0,
         "n_ref_buildings": 40},
        {"aoi": "P", "tile_z": 17, "tile_x": 1, "tile_y": 1,
         "metric": "f1", "value": 0.7, "median_ref_area_m2": 120.0,
         "n_ref_buildings": 8},
        {"aoi": "P", "tile_z": 17, "tile_x": 2, "tile_y": 1,
         "metric": "f1", "value": 0.5, "median_ref_area_m2": 60.0,
         "n_ref_buildings": 20},
    ])


def test_stratify_by_area_rejects_zero_bins():
    """Precondition test -- expected to pass on clean checkout."""
    df = _tidy_results()
    with pytest.raises(ValueError):
        stratify_by_area(df, n_bins=0)


def test_stratify_by_area_returns_expected_columns():
    """stratify_by_area on the 4-row tidy fixture -> a DataFrame with
    the canonical (aoi, area_bin, metric, value_mean, value_ci_lo,
    value_ci_hi, n_tiles) columns.
    """
    df = _tidy_results()
    out = stratify_by_area(df, n_bins=2)
    expected_cols = {
        "aoi", "area_bin", "metric",
        "value_mean", "value_ci_lo", "value_ci_hi", "n_tiles",
    }
    assert expected_cols.issubset(set(out.columns)), (
        f"expected columns {expected_cols}, got {set(out.columns)}"
    )
    assert len(out) > 0


def test_stratify_by_density_rejects_nonpositive_window():
    """Precondition test -- expected to pass on clean checkout."""
    df = _tidy_results()
    with pytest.raises(ValueError):
        stratify_by_density(df, window_m=0.0)


def test_stratify_by_density_returns_expected_columns():
    df = _tidy_results()
    out = stratify_by_density(df, window_m=500.0)
    expected_cols = {
        "aoi", "density_bin", "metric",
        "value_mean", "value_ci_lo", "value_ci_hi", "n_tiles",
    }
    assert expected_cols.issubset(set(out.columns))


# ---------------------------------------------------------------------------
# Meta: fail-mode discipline
# ---------------------------------------------------------------------------

def test_stubs_fail_with_notimplemented_not_something_else():
    """Task 2 mandates: the test suite must FAIL on a clean checkout,
    and it must fail with NotImplementedError, not an ImportError or
    a fixture bug.

    This test explicitly asserts that every stub -- once handed
    correctly-typed inputs -- raises NotImplementedError specifically,
    and not something else. If this test passes but the other tests
    all raise NotImplementedError too, the failure is 'not implemented',
    which is the correct signal.
    """
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    mr   = MatchResult(matched=[(0, 0, 1.0)])
    df   = _tidy_results()

    calls = [
        ("match_instances",  lambda: match_instances(pred, ref, 0.5)),
        ("instance_prf",     lambda: instance_prf(mr)),
        ("merge_rate",       lambda: merge_rate(mr)),
        ("split_rate",       lambda: split_rate(mr)),
        ("pixel_iou",        lambda: pixel_iou(pred, ref,
                                               bounds=(0, 0, 10, 10),
                                               resolution=0.5)),
        ("count_error",      lambda: count_error(pred, ref)),
        ("boundary_distance",lambda: boundary_distance(mr)),
        ("stratify_by_area", lambda: stratify_by_area(df, n_bins=2)),
        ("stratify_by_density",
                             lambda: stratify_by_density(df, window_m=500.0)),
    ]
    for name, fn in calls:
        try:
            fn()
        except NotImplementedError:
            continue
        except Exception as e:  # noqa: BLE001
            # A real implementation would return a value and we wouldn't
            # reach either branch. Anything else (ImportError, TypeError,
            # KeyError, ...) is the fixture bug Task 2 explicitly asks
            # this test to catch.
            pytest.fail(
                f"{name}: expected NotImplementedError (or a real "
                f"implementation returning a value), got {type(e).__name__}: {e}"
            )
        else:
            # Reaching here means the stub was implemented and returned
            # a value. That is legal once the implementer fills it in;
            # the assertion tests above will start doing real work.
            continue
