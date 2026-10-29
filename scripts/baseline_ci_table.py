"""Compute 95% bootstrap CIs for the four baseline AOIs and emit the
README-ready markdown row for each.

Reuses:
    bfbench.metrics.bootstrap.bootstrap_micro_prf     for P / R / F1
    bfbench.metrics.bootstrap.bootstrap_mean_ci       for merge/split rate
    scripts.compare_baseline_vs_finetuned._bootstrap_median_ci
                                                      for pixel_iou + Chamfer

Point estimates are identical to scripts/aoi_summary.py; this is the same
data, just with CIs attached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bfbench.metrics.bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    BootstrapCI,
    bootstrap_mean_ci,
    bootstrap_micro_prf,
)


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    tile_cols = ["aoi", "tile_z", "tile_x", "tile_y", "n_pred", "n_ref"]
    ok = df[df["status"] == "ok"].copy()
    return ok.pivot_table(
        index=tile_cols, columns="metric", values="value", aggfunc="first",
    ).reset_index()


def _bootstrap_median_ci(values: np.ndarray, *,
                         n_boot: int, seed: int, alpha: float) -> BootstrapCI:
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
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.median(arr[idx], axis=1)
    return BootstrapCI(
        point=float(np.median(arr)),
        lo=float(np.percentile(boots, 100 * alpha / 2.0)),
        hi=float(np.percentile(boots, 100 * (1.0 - alpha / 2.0))),
        n_boot=n_boot, seed=seed,
    )


def one_aoi(parquet: Path, *, n_boot: int = DEFAULT_N_BOOT,
            seed: int = DEFAULT_SEED, alpha: float = DEFAULT_ALPHA) -> dict:
    wide = _pivot(pd.read_parquet(parquet))
    tp_fp_fn = wide[["instance_prf.tp", "instance_prf.fp",
                     "instance_prf.fn"]].to_numpy(dtype=float)
    prf = bootstrap_micro_prf(tp_fp_fn, n_boot=n_boot, seed=seed, alpha=alpha)

    merge_ci  = bootstrap_mean_ci(wide["merge_rate"].to_numpy(dtype=float),
                                 n_boot=n_boot, seed=seed, alpha=alpha)
    split_ci  = bootstrap_mean_ci(wide["split_rate"].to_numpy(dtype=float),
                                 n_boot=n_boot, seed=seed, alpha=alpha)
    pixel_ci  = _bootstrap_median_ci(wide["pixel_iou"].to_numpy(dtype=float),
                                    n_boot=n_boot, seed=seed, alpha=alpha)
    chamfer_ci = _bootstrap_median_ci(
        wide["boundary_distance.mean_chamfer"].to_numpy(dtype=float),
        n_boot=n_boot, seed=seed, alpha=alpha,
    )
    return {
        "aoi":        wide["aoi"].iloc[0] if len(wide) else "UNKNOWN",
        "n_tiles":    int(len(wide)),
        "precision":  prf["precision"],
        "recall":     prf["recall"],
        "f1":         prf["f1"],
        "merge":      merge_ci,
        "split":      split_ci,
        "pixel_iou":  pixel_ci,
        "chamfer":    chamfer_ci,
    }


def _fmt(ci: BootstrapCI, decimals: int = 3) -> str:
    if ci.point != ci.point:  # NaN
        return "n/a"
    return f"{ci.point:.{decimals}f} [{ci.lo:.{decimals}f}, {ci.hi:.{decimals}f}]"


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| AOI | n_tiles | precision [CI] | recall [CI] | F1 [CI] | merge_rate [CI] | pixel_IoU_median [CI] | Chamfer_median_m [CI] |",
        "|---|--:|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['aoi']} | {r['n_tiles']} | "
            f"{_fmt(r['precision'])} | {_fmt(r['recall'])} | "
            f"{_fmt(r['f1'])} | {_fmt(r['merge'])} | "
            f"{_fmt(r['pixel_iou'])} | {_fmt(r['chamfer'], decimals=2)} |"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquets", nargs="+", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    rows = [one_aoi(pq, n_boot=args.n_boot, seed=args.seed) for pq in args.parquets]
    print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
