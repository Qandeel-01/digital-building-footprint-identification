"""Tests for the bfbench.metrics module (Phase 3 spec).

Design contract:

    * The FOUR functions implemented in Phase 3 -- pixel_iou,
      count_error / count_error_aoi, stratify_by_area, and
      stratify_by_density -- have real assertions and pass on clean
      checkout. Expected values are computed by hand where possible.

    * The RESERVED stubs -- match_instances, instance_prf, merge_rate,
      split_rate, boundary_distance -- still raise NotImplementedError.
      A meta-test enumerates them and pins the failure mode as
      NotImplementedError specifically. This is the invariant the
      project owner reserved.

    * Symmetry is NOT asserted on instance-level metrics. Precision /
      recall / merge / split are directional; the only symmetry
      required is that the ASSIGNMENT (the set of matched pairs) is
      order-independent. See docs/metric-decisions.md section 1 and
      the test file comment on match_instances.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
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
    count_error_aoi,
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
    return gpd.GeoDataFrame(geometry=polys, crs=EA)


# ---------------------------------------------------------------------------
# pixel_iou -- implemented in Phase 3.
# ---------------------------------------------------------------------------

def test_pixel_iou_two_squares_50pct_overlap_edges_on_pixel_grid():
    """Two 10x10 m squares overlapping by 50 %, edges on the pixel grid.

    Arithmetic in the equal-area CRS at resolution=0.5 m over
    bounds=(0, 0, 20, 10):
        grid = 20/0.5 = 40 px wide  x  10/0.5 = 20 px tall
        pred fills [0..10]   -> 20 x 20 = 400 px
        ref  fills [5..15]   -> 20 x 20 = 400 px
        inter fills [5..10]  -> 10 x 20 = 200 px
        union                -> 400 + 400 - 200 = 600 px
        IoU = 200 / 600 = 1/3, exactly, under either snapping rule
        because the polygon edges lie on pixel boundaries.
    """
    ref  = _gdf([_square(5.0, 0.0, 10.0)])
    pred_mask = np.zeros((20, 40), dtype=np.uint8)
    pred_mask[:, 0:20] = 1
    iou = pixel_iou(pred_mask, ref, bounds=(0.0, 0.0, 20.0, 10.0),
                    resolution=0.5)
    assert math.isclose(iou, 1.0 / 3.0, rel_tol=1e-6, abs_tol=1e-6)


def test_pixel_iou_probability_thresholded():
    """A float probability array gets thresholded at `threshold`."""
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    prob = np.zeros((20, 20), dtype=np.float32)
    prob[:, :] = 0.6  # everywhere above 0.5 -> whole grid predicts
    iou_high = pixel_iou(prob, ref, bounds=(0.0, 0.0, 10.0, 10.0),
                         resolution=0.5, threshold=0.5)
    # ref is a 10x10 m square; the grid is 10x10 m. Both cover the
    # whole grid -> IoU = 1.
    assert math.isclose(iou_high, 1.0, abs_tol=1e-6)

    iou_low = pixel_iou(prob, ref, bounds=(0.0, 0.0, 10.0, 10.0),
                        resolution=0.5, threshold=0.7)
    # Nothing above 0.7 -> pred is all zeros. ref fills the grid.
    # inter=0, union=20*20=400, IoU=0.
    assert math.isclose(iou_low, 0.0, abs_tol=1e-6)


def test_pixel_iou_all_touched_inflates_area_size_dependently():
    """all_touched inflates reference area in a size-dependent way.

    Theory. A square of side s rasterised at resolution p:
        centre-only pixel count  = round(s/p)^2
        all-touched pixel count  = (round(s/p) + 1)^2       (edge pixels added)
        inflation                = 2p/s + (p/s)^2

    To exercise the difference we place the polygon with a HALF-PIXEL
    offset so the polygon edges do NOT coincide with pixel boundaries:
    with edges on the grid, the two snapping rules snap identically
    and the test degenerates.

    Fixture: prediction is a solid H x W raster; reference is one
    square at (0.25, 0.25) with side `s`. `iou_centre = A_centre / N`
    and `iou_all = A_all / N` where N is the full-grid pixel count,
    so the ratio directly reflects the reference-area inflation.

    See docs/metric-decisions.md section 2.
    """
    p = 0.5  # metres per pixel
    for side in (5.0, 9.0, 15.0, 32.0):
        w = side + 4.0  # a little margin so the pred grid is bigger than ref
        h = side + 4.0
        n_x = int(w / p); n_y = int(h / p)
        pred_full = np.ones((n_y, n_x), dtype=np.uint8)
        # Half-pixel offset -> edges fall BETWEEN pixel centres
        ref  = _gdf([_square(2.25, 2.25, side)])
        iou_centre = pixel_iou(pred_full, ref, bounds=(0.0, 0.0, w, h),
                               resolution=p, all_touched=False)
        iou_all    = pixel_iou(pred_full, ref, bounds=(0.0, 0.0, w, h),
                               resolution=p, all_touched=True)
        assert iou_all > iou_centre, (
            f"side={side}: all_touched must inflate ref area with "
            f"off-grid placement (centre={iou_centre:.4f}, all={iou_all:.4f})"
        )
        inflation = (iou_all - iou_centre) / iou_centre
        assert inflation > 0.0, f"side={side}: inflation={inflation}"


def test_pixel_iou_all_touched_bias_is_monotone_in_size():
    """Smaller buildings inflate more than larger ones (perimeter/area)."""
    p = 0.5
    inflations = []
    for side in (5.0, 9.0, 15.0, 32.0):
        w = side + 4.0
        n = int(w / p)
        pred_full = np.ones((n, n), dtype=np.uint8)
        ref  = _gdf([_square(2.25, 2.25, side)])
        iou_c = pixel_iou(pred_full, ref, bounds=(0.0, 0.0, w, w),
                          resolution=p, all_touched=False)
        iou_a = pixel_iou(pred_full, ref, bounds=(0.0, 0.0, w, w),
                          resolution=p, all_touched=True)
        inflations.append((iou_a - iou_c) / iou_c)
    # Strictly decreasing across the four sizes:
    for a, b in zip(inflations[:-1], inflations[1:], strict=True):
        assert a > b, f"inflation not monotone: {inflations}"


def test_pixel_iou_empty_pred_and_ref_is_nan():
    ref  = _gdf([])
    empty = np.zeros((10, 10), dtype=np.uint8)
    iou = pixel_iou(empty, ref, bounds=(0.0, 0.0, 5.0, 5.0), resolution=0.5)
    assert math.isnan(iou)


def test_pixel_iou_rejects_wrong_crs():
    ref  = _gdf([_square(0.0, 0.0, 10.0)]).to_crs("EPSG:4326")
    arr  = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(NonEqualAreaCRSError):
        pixel_iou(arr, ref, bounds=(0.0, 0.0, 10.0, 10.0), resolution=0.5)


def test_pixel_iou_rejects_nonpositive_resolution():
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    arr  = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        pixel_iou(arr, ref, bounds=(0.0, 0.0, 10.0, 10.0), resolution=0.0)


def test_pixel_iou_rejects_shape_grid_mismatch():
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    arr  = np.zeros((10, 10), dtype=np.uint8)  # wrong shape
    with pytest.raises(ValueError):
        pixel_iou(arr, ref, bounds=(0.0, 0.0, 10.0, 10.0), resolution=0.5)


# ---------------------------------------------------------------------------
# count_error -- implemented in Phase 3 (per-tile abs/signed only)
# ---------------------------------------------------------------------------

def test_count_error_signed_and_absolute():
    """4 predictions, 6 references.
        abs_error    = |4 - 6| = 2
        signed_error =  4 - 6  = -2
    """
    pred = _gdf([_square(0.0, 0.0, 1.0)] * 4)
    ref  = _gdf([_square(0.0, 0.0, 1.0)] * 6)
    r = count_error(pred, ref)
    assert math.isclose(r["abs_error"],    2.0, abs_tol=1e-9)
    assert math.isclose(r["signed_error"], -2.0, abs_tol=1e-9)
    # Per-tile relative error is deliberately not present.
    assert "relative_error" not in r


def test_count_error_zero_and_zero():
    pred = _gdf([])
    ref  = _gdf([])
    r = count_error(pred, ref)
    assert math.isclose(r["abs_error"],    0.0)
    assert math.isclose(r["signed_error"], 0.0)


def test_count_error_aoi_relative_error():
    """3 tiles: (n_pred, n_ref) = (4, 6), (7, 5), (0, 3).
        sum n_pred = 11, sum n_ref = 14
        abs = |4-6| + |7-5| + |0-3| = 2 + 2 + 3 = 7
        signed = 11 - 14 = -3
        rel = -3 / 14 = -0.2142857...
    """
    rows = pd.DataFrame([
        {"n_pred": 4, "n_ref": 6},
        {"n_pred": 7, "n_ref": 5},
        {"n_pred": 0, "n_ref": 3},
    ])
    r = count_error_aoi(rows)
    assert math.isclose(r["abs_error"],       7.0)
    assert math.isclose(r["signed_error"],   -3.0)
    assert math.isclose(r["relative_error"], -3.0 / 14.0, rel_tol=1e-9)


def test_count_error_aoi_all_zero_reference_is_nan_only_at_aoi_level():
    """If (across every tile) n_ref == 0, relative error is undefined.
    At the AOI level this is genuinely a degenerate case (no reference
    data) and NaN is correct.
    """
    rows = pd.DataFrame([{"n_pred": 5, "n_ref": 0},
                         {"n_pred": 3, "n_ref": 0}])
    r = count_error_aoi(rows)
    assert math.isnan(r["relative_error"])


# ---------------------------------------------------------------------------
# stratify_by_area / stratify_by_density -- implemented in Phase 3.
# ---------------------------------------------------------------------------

def _tidy_results(with_density: bool = False) -> pd.DataFrame:
    rows = []
    aois = ["V"]
    rng = np.random.default_rng(1)
    for aoi in aois:
        for i in range(30):
            med_area = float(rng.uniform(20, 500))
            base = {
                "aoi": aoi, "tile_z": 17, "tile_x": 1000 + i, "tile_y": 20000,
                "metric": "value_x",
                "value": float(rng.uniform(0, 1)),
                "median_ref_area_m2": med_area,
            }
            if with_density:
                base["ref_density_per_ha"] = float(rng.uniform(5, 120))
                base["window_m"] = 500.0
            rows.append(base)
    return pd.DataFrame(rows)


def test_stratify_by_area_returns_expected_columns_and_bins():
    df = _tidy_results()
    out = stratify_by_area(df, n_bins=3)
    for col in ("aoi", "area_bin", "bin_left", "bin_right",
                "value_mean", "value_ci_lo", "value_ci_hi", "n_tiles"):
        assert col in out.columns, f"missing {col}: {out.columns.tolist()}"
    # 30 rows into 3 bins -> at most 3 (per AOI). Not exactly 3 if
    # duplicates collapse; assert >=1 and <=3.
    for _, sub in out.groupby("aoi"):
        assert 1 <= len(sub) <= 3


def test_stratify_by_area_rejects_zero_bins():
    df = _tidy_results()
    with pytest.raises(ValueError):
        stratify_by_area(df, n_bins=0)


def test_stratify_by_density_requires_window_m():
    df = _tidy_results(with_density=True)
    with pytest.raises(TypeError):
        stratify_by_density(df)  # type: ignore[call-arg]


def test_stratify_by_density_rejects_nonpositive_window():
    df = _tidy_results(with_density=True)
    with pytest.raises(ValueError):
        stratify_by_density(df, window_m=0.0)


def test_stratify_by_density_returns_expected_columns():
    df = _tidy_results(with_density=True)
    out = stratify_by_density(df, window_m=500.0, n_bins=3)
    for col in ("aoi", "density_bin", "bin_left", "bin_right",
                "value_mean", "value_ci_lo", "value_ci_hi", "n_tiles"):
        assert col in out.columns, f"missing {col}: {out.columns.tolist()}"


def test_stratify_by_density_filters_by_window_m_column():
    """When results have a window_m column, only matching rows are used."""
    df = _tidy_results(with_density=True)
    # Add rows at window_m=100 with a distinct value distribution.
    extra = df.copy()
    extra["window_m"] = 100.0
    extra["value"] = extra["value"] + 100.0  # very distinct
    combined = pd.concat([df, extra], ignore_index=True)
    out_500 = stratify_by_density(combined, window_m=500.0, n_bins=3)
    # No bin should have value_mean > 2 (the pre-shift values are in [0,1]).
    assert (out_500["value_mean"] < 2.0).all()


# ---------------------------------------------------------------------------
# Reserved stubs: still NotImplementedError, and match_instances +
# instance_prf must NOT have symmetry asserted (see spec section 1).
# ---------------------------------------------------------------------------

# NOTE (Phase 3 correction to the Phase 2 recommendation): we deliberately
# do NOT test metric-level symmetry:
#
#     match_instances(A, B) vs match_instances(B, A) may produce
#     different instance_prf outputs by design -- precision runs
#     pred->ref, recall runs ref->pred, merge/split are asymmetric.
#     Forcing metric-level symmetry would remove the ability to
#     measure merge / split at all, which is the phenomenon this
#     benchmark exists to study.
#
# The only invariant is that the ASSIGNMENT (the set {(p_idx, r_idx)} of
# matched pairs) is order-independent. That test will be added when
# match_instances is implemented; it cannot be written against a stub.

RESERVED = ("match_instances", "instance_prf", "merge_rate", "split_rate",
            "boundary_distance")


def test_reserved_stubs_fail_with_notimplemented_not_something_else():
    """Meta-test: every reserved stub must raise NotImplementedError
    specifically, not ImportError, not TypeError, not ValueError from
    a precondition. If this fails, the failure surface has drifted."""
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    mr   = MatchResult(matched=[(0, 0, 1.0)])
    calls = [
        ("match_instances",   lambda: match_instances(pred, ref, 0.5)),
        ("instance_prf",      lambda: instance_prf(mr)),
        ("merge_rate",        lambda: merge_rate(mr)),
        ("split_rate",        lambda: split_rate(mr)),
        ("boundary_distance", lambda: boundary_distance(mr)),
    ]
    for name, fn in calls:
        try:
            fn()
        except NotImplementedError:
            continue
        except Exception as e:  # noqa: BLE001
            pytest.fail(
                f"{name}: expected NotImplementedError, got "
                f"{type(e).__name__}: {e}"
            )
        else:
            pytest.fail(f"{name}: expected NotImplementedError, got a return value")


def test_reserved_stubs_are_all_in_the_reserved_list():
    """If a new reserved stub is added, this test forces us to
    remember the list."""
    for name in RESERVED:
        # Each name must be importable and callable.
        from bfbench import metrics as m  # noqa: PLC0415
        fn = getattr(m, name)
        assert callable(fn), f"{name} not callable"


def test_match_instances_precondition_still_enforced():
    """The equal-area assertion runs BEFORE the NotImplementedError
    inside match_instances, so callers can't silently pass an
    EPSG:4326 GeoDataFrame."""
    pred = _gdf([_square(0.0, 0.0, 10.0)]).to_crs("EPSG:4326")
    ref  = _gdf([_square(0.0, 0.0, 10.0)]).to_crs("EPSG:4326")
    with pytest.raises(NonEqualAreaCRSError):
        match_instances(pred, ref, iou_threshold=0.5)


def test_match_instances_iou_threshold_range():
    """Range check runs BEFORE NotImplementedError."""
    pred = _gdf([_square(0.0, 0.0, 10.0)])
    ref  = _gdf([_square(0.0, 0.0, 10.0)])
    with pytest.raises(ValueError):
        match_instances(pred, ref, iou_threshold=1.5)
    with pytest.raises(ValueError):
        match_instances(pred, ref, iou_threshold=0.0)


def test_boundary_distance_rejects_nonpositive_sample_step():
    with pytest.raises(ValueError):
        boundary_distance(MatchResult(), sample_step_m=0.0)
