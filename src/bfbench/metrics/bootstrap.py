"""Non-parametric percentile bootstrap for top-line metrics.

`bfbench.metrics.stratify` already has a tiny internal bootstrap used
for density/area bin confidence bands. This module exposes the same
math at module level for the Phase 1 "baseline vs fine-tuned" table,
with two differences:

1. More resamples by default (2000 vs 500). The stratified version
   runs once per (AOI, bin) and lives inside hot plotting code; the
   top-line version runs ~20 times total (metrics x AOIs) so the extra
   resamples are free and the CI is tighter.
2. A `point_fn` parameter so the caller can bootstrap any statistic,
   not just the mean. For micro-averaged precision / recall / F1 the
   point estimate is NOT the mean of per-tile values -- it is a
   ratio of sums -- and the bootstrap must mirror that.

Both the bin-level and the top-line bootstrap use
`numpy.random.default_rng(seed)` so (same data, same seed) is byte-
identical.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05  # 95 % CI


@dataclass(frozen=True)
class BootstrapCI:
    """One (point estimate, lo, hi, n_boot, seed) row."""

    point: float
    lo: float
    hi: float
    n_boot: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point":  float(self.point),
            "ci_lo":  float(self.lo),
            "ci_hi":  float(self.hi),
            "n_boot": int(self.n_boot),
            "seed":   int(self.seed),
        }


def bootstrap_mean_ci(values: np.ndarray | list[float],
                     *, n_boot: int = DEFAULT_N_BOOT,
                     seed: int = DEFAULT_SEED,
                     alpha: float = DEFAULT_ALPHA) -> BootstrapCI:
    """Percentile bootstrap on the sample mean.

    NaNs are dropped before resampling. Empty input returns an all-NaN
    CI rather than raising, so a caller aggregating many metrics can
    produce a tidy table even when one metric has no valid rows.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                           n_boot=n_boot, seed=seed)
    if n == 1:
        return BootstrapCI(point=float(arr[0]), lo=float(arr[0]), hi=float(arr[0]),
                           n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    # resamples: (n_boot, n), means along axis 1
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    return BootstrapCI(
        point=float(arr.mean()),
        lo=float(np.percentile(boots, 100 * alpha / 2.0)),
        hi=float(np.percentile(boots, 100 * (1.0 - alpha / 2.0))),
        n_boot=n_boot,
        seed=seed,
    )


def bootstrap_statistic_ci(samples: np.ndarray,
                          point_fn: Callable[[np.ndarray], float],
                          *, n_boot: int = DEFAULT_N_BOOT,
                          seed: int = DEFAULT_SEED,
                          alpha: float = DEFAULT_ALPHA) -> BootstrapCI:
    """Percentile bootstrap on an arbitrary statistic of a 2-D sample.

    `samples` is shape (N, D): N observations, D fields per observation
    (e.g. (tp, fp, fn) per tile). `point_fn` takes an (M, D) slice and
    returns a scalar point estimate. This is the entry point for
    ratio-of-sums statistics like micro-F1 where the correct bootstrap
    is NOT a bootstrap of means.

    Returns an all-NaN CI if N == 0.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    if n == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                           n_boot=n_boot, seed=seed)
    point = float(point_fn(arr))
    if n == 1:
        return BootstrapCI(point=point, lo=point, hi=point,
                           n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boots[b] = float(point_fn(arr[idx[b]]))
    return BootstrapCI(
        point=point,
        lo=float(np.percentile(boots, 100 * alpha / 2.0)),
        hi=float(np.percentile(boots, 100 * (1.0 - alpha / 2.0))),
        n_boot=n_boot,
        seed=seed,
    )


def micro_prf_point(tp_fp_fn: np.ndarray) -> dict[str, float]:
    """Micro-averaged precision/recall/F1 from a (N, 3) array of per-tile
    (tp, fp, fn) counts. The micro average is the ratio of sums,
    *not* the mean of per-tile ratios; that is what you want when tile
    sizes differ in how many objects they contain.
    """
    arr = np.asarray(tp_fp_fn, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"micro_prf_point expects (N, 3), got {arr.shape}")
    tp = float(arr[:, 0].sum())
    fp = float(arr[:, 1].sum())
    fn = float(arr[:, 2].sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def bootstrap_micro_prf(tp_fp_fn: np.ndarray,
                        *, n_boot: int = DEFAULT_N_BOOT,
                        seed: int = DEFAULT_SEED,
                        alpha: float = DEFAULT_ALPHA
                        ) -> dict[str, BootstrapCI]:
    """CIs for micro-averaged precision, recall, F1 from per-tile (tp,
    fp, fn) counts. Resamples tiles with replacement; recomputes the
    ratio of sums; takes percentiles.
    """
    arr = np.asarray(tp_fp_fn, dtype=float)
    out = {}
    for name in ("precision", "recall", "f1"):
        out[name] = bootstrap_statistic_ci(
            arr,
            point_fn=lambda a, k=name: micro_prf_point(a)[k],
            n_boot=n_boot, seed=seed, alpha=alpha,
        )
    return out


__all__ = [
    "BootstrapCI",
    "DEFAULT_ALPHA",
    "DEFAULT_N_BOOT",
    "DEFAULT_SEED",
    "bootstrap_mean_ci",
    "bootstrap_micro_prf",
    "bootstrap_statistic_ci",
    "micro_prf_point",
]
