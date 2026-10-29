"""Bootstrap CI tests.

Hand-computed truths for the mean bootstrap, plus structural checks on
the micro-PRF ratio-of-sums bootstrap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bfbench.metrics.bootstrap import (
    BootstrapCI,
    bootstrap_mean_ci,
    bootstrap_micro_prf,
    bootstrap_statistic_ci,
    micro_prf_point,
)

# ---------------------------------------------------------------------------
# Determinism + edge cases on bootstrap_mean_ci
# ---------------------------------------------------------------------------

def test_bootstrap_mean_is_deterministic_given_seed():
    rng = np.random.default_rng(0)
    xs = rng.normal(loc=5.0, scale=1.0, size=50)
    a = bootstrap_mean_ci(xs, n_boot=500, seed=42)
    b = bootstrap_mean_ci(xs, n_boot=500, seed=42)
    assert a.point == b.point and a.lo == b.lo and a.hi == b.hi


def test_bootstrap_mean_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    xs = rng.normal(loc=5.0, scale=1.0, size=100)
    ci = bootstrap_mean_ci(xs, n_boot=500, seed=7)
    assert ci.lo <= ci.point <= ci.hi


def test_bootstrap_mean_tight_at_large_n():
    rng = np.random.default_rng(2)
    xs = rng.normal(loc=0.0, scale=1.0, size=5000)
    ci = bootstrap_mean_ci(xs, n_boot=500, seed=0)
    # standard error ~ 1 / sqrt(5000) = 0.014; 95% CI half-width ~ 0.028
    assert (ci.hi - ci.lo) < 0.1


def test_bootstrap_mean_dropna():
    xs = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
    ci = bootstrap_mean_ci(xs, n_boot=200, seed=0)
    assert ci.point == pytest.approx(2.0)


def test_bootstrap_mean_empty_returns_nan():
    ci = bootstrap_mean_ci([], n_boot=100, seed=0)
    assert math.isnan(ci.point)
    assert math.isnan(ci.lo)
    assert math.isnan(ci.hi)


def test_bootstrap_mean_single_value_is_degenerate():
    ci = bootstrap_mean_ci([3.0], n_boot=100, seed=0)
    assert ci.point == 3.0
    assert ci.lo == 3.0 and ci.hi == 3.0


# ---------------------------------------------------------------------------
# bootstrap_statistic_ci -- the general entry point
# ---------------------------------------------------------------------------

def test_statistic_ci_reduces_to_mean_for_mean_point_fn():
    rng = np.random.default_rng(3)
    xs = rng.normal(size=200).reshape(-1, 1)
    mean_ci = bootstrap_mean_ci(xs.ravel(), n_boot=400, seed=5)
    stat_ci = bootstrap_statistic_ci(xs, point_fn=lambda a: float(a.mean()),
                                    n_boot=400, seed=5)
    # Both use the same seed, same resampling scheme, same statistic
    # -> point estimate is identical, CI is numerically identical.
    assert stat_ci.point == pytest.approx(mean_ci.point)
    assert stat_ci.lo    == pytest.approx(mean_ci.lo)
    assert stat_ci.hi    == pytest.approx(mean_ci.hi)


# ---------------------------------------------------------------------------
# micro_prf_point -- ratio-of-sums arithmetic
# ---------------------------------------------------------------------------

def test_micro_prf_point_hand_computed():
    # Two tiles: (tp=4, fp=1, fn=1) and (tp=6, fp=2, fn=3).
    # Totals: TP=10, FP=3, FN=4.
    # precision = 10 / 13; recall = 10 / 14; f1 = 2 * p * r / (p + r)
    arr = np.array([[4, 1, 1], [6, 2, 3]], dtype=float)
    out = micro_prf_point(arr)
    assert out["precision"] == pytest.approx(10 / 13)
    assert out["recall"]    == pytest.approx(10 / 14)
    expected_f1 = 2 * (10 / 13) * (10 / 14) / ((10 / 13) + (10 / 14))
    assert out["f1"] == pytest.approx(expected_f1)


def test_micro_prf_point_zero_predictions():
    # No predictions anywhere -> precision undefined (0/0 -> NaN) but
    # recall is defined (0 / FN) = 0, so F1 is 0 (defined limit).
    arr = np.array([[0, 0, 5]], dtype=float)
    out = micro_prf_point(arr)
    assert math.isnan(out["precision"])
    assert out["recall"] == 0.0
    # f1 is NaN because precision is NaN
    assert math.isnan(out["f1"])


def test_micro_prf_point_rejects_bad_shape():
    with pytest.raises(ValueError, match="N, 3"):
        micro_prf_point(np.array([[1, 2]]))


# ---------------------------------------------------------------------------
# bootstrap_micro_prf -- the end-to-end call used in the comparison table
# ---------------------------------------------------------------------------

def test_bootstrap_micro_prf_returns_three_cis_with_point_in_range():
    rng = np.random.default_rng(4)
    # 50 tiles with per-tile TP ~ Poisson(10), FP ~ Poisson(2), FN ~ Poisson(2)
    tp = rng.poisson(10, size=50)
    fp = rng.poisson(2, size=50)
    fn = rng.poisson(2, size=50)
    arr = np.stack([tp, fp, fn], axis=1).astype(float)
    out = bootstrap_micro_prf(arr, n_boot=400, seed=11)
    assert set(out.keys()) == {"precision", "recall", "f1"}
    for name, ci in out.items():
        assert isinstance(ci, BootstrapCI), name
        assert ci.lo <= ci.point <= ci.hi, name


def test_bootstrap_micro_prf_deterministic_given_seed():
    arr = np.array([[4, 1, 1], [6, 2, 3], [3, 0, 2]], dtype=float)
    a = bootstrap_micro_prf(arr, n_boot=200, seed=0)
    b = bootstrap_micro_prf(arr, n_boot=200, seed=0)
    for k in ("precision", "recall", "f1"):
        assert a[k].point == b[k].point
        assert a[k].lo    == b[k].lo
        assert a[k].hi    == b[k].hi


def test_bootstrap_ci_as_dict_keys():
    ci = BootstrapCI(point=0.5, lo=0.4, hi=0.6, n_boot=2000, seed=7)
    d = ci.as_dict()
    assert set(d.keys()) == {"point", "ci_lo", "ci_hi", "n_boot", "seed"}
    assert d["point"] == 0.5
    assert d["ci_lo"] == 0.4
    assert d["ci_hi"] == 0.6
