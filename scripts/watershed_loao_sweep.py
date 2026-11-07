"""Leave-one-AOI-out sweep over `min_marker_distance_m` for Phase 2b
watershed.

For each held-out AOI:
    1. Average F1 across the OTHER three AOIs for each candidate marker
       distance in {3, 5, 8, 12} m.
    2. Pick the marker distance that maximises that three-AOI mean.
    3. Report the F1 that choice yields on the held-out AOI, alongside
       the baseline F1 on the same tiles (so you can see whether
       watershed is worth running at all).

This mimics a clean "tune on three regions, test on the fourth" split
*at the AOI level*. It does NOT use the Phase 1 geographic block split
(which does not exist yet).

Every sweep parquet already exists on disk from
`scripts/watershed_vectorize.py` + `bfbench eval`; this is pure
aggregation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bfbench.metrics.bootstrap import bootstrap_mean_ci, bootstrap_micro_prf

_LOG = logging.getLogger("bfbench.watershed_loao_sweep")

AOIS = ("vegas", "paris", "shanghai", "khartoum")
AOI_LABEL = {
    "vegas":    "AOI_2_Vegas",
    "paris":    "AOI_3_Paris",
    "shanghai": "AOI_4_Shanghai",
    "khartoum": "AOI_5_Khartoum",
}


def _f1_point(parquet: Path) -> float:
    """Micro-F1 from a bfbench eval parquet (ratio of sums, same
    definition as scripts/baseline_ci_table.py)."""
    df = pd.read_parquet(parquet)
    ok = df[df.status == "ok"]
    wide = ok.pivot_table(
        index=["tile_z", "tile_x", "tile_y"],
        columns="metric", values="value", aggfunc="first",
    ).reset_index()
    tp = float(wide["instance_prf.tp"].sum())
    fp = float(wide["instance_prf.fp"].sum())
    fn = float(wide["instance_prf.fn"].sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        return float("nan")
    return 2 * prec * rec / (prec + rec)


def _f1_ci(parquet: Path) -> tuple[float, float, float]:
    """95% bootstrap CI on micro-F1 (ratio of sums), resampling TILES
    with replacement. Matches _f1_point's point estimate exactly.

    Consistent with scripts/baseline_ci_table.py: that script uses
    bootstrap_micro_prf on the same (tp, fp, fn) triples.
    """
    df = pd.read_parquet(parquet)
    ok = df[df.status == "ok"]
    wide = ok.pivot_table(
        index=["tile_z", "tile_x", "tile_y"],
        columns="metric", values="value", aggfunc="first",
    ).reset_index()
    tp_fp_fn = wide[["instance_prf.tp", "instance_prf.fp",
                     "instance_prf.fn"]].to_numpy(dtype=float)
    if len(tp_fp_fn) == 0:
        return (float("nan"), float("nan"), float("nan"))
    cis = bootstrap_micro_prf(tp_fp_fn, n_boot=2000, seed=0)
    f1_ci = cis["f1"]
    return (f1_ci.point, f1_ci.lo, f1_ci.hi)


def _mean_metric_ci(parquet: Path, metric: str) -> tuple[float, float, float]:
    """95% bootstrap CI on the mean per-tile value of `metric`
    (e.g. 'merge_rate', 'split_rate')."""
    df = pd.read_parquet(parquet)
    arr = df[df.metric == metric].value.to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    ci = bootstrap_mean_ci(arr, n_boot=2000, seed=0)
    return (ci.point, ci.lo, ci.hi)


def _sweep_parquets(aoi: str, distances_m: list[int]) -> dict[int, Path]:
    return {
        d: Path(f"runs/{aoi}_prepare_sweep_m{d}/eval/results.parquet")
        if d != 5
        else Path(f"runs/{aoi}_prepare_phase2b/eval/results.parquet")
        for d in distances_m
    }


def _baseline_parquet(aoi: str) -> Path:
    return Path(f"runs/{aoi}_prepare/eval/results.parquet")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distances", type=int, nargs="+", default=[3, 5, 8, 12],
                   help="Candidate min_marker_distance_m values to sweep.")
    args = p.parse_args(argv)
    distances = sorted(set(args.distances))

    # Verify every parquet exists before we start aggregating.
    missing: list[Path] = []
    for aoi in AOIS:
        if not _baseline_parquet(aoi).is_file():
            missing.append(_baseline_parquet(aoi))
        for pq in _sweep_parquets(aoi, distances).values():
            if not pq.is_file():
                missing.append(pq)
    if missing:
        for m in missing:
            print(f"MISSING: {m}", file=sys.stderr)
        print("Run scripts/watershed_vectorize.py + bfbench eval for the "
              "distances above before invoking this script.", file=sys.stderr)
        return 2

    # Build the AOI x distance F1 matrix (point estimates).
    rows = []
    for aoi in AOIS:
        f1_base = _f1_point(_baseline_parquet(aoi))
        rows.append({"aoi": aoi, "distance_m": "baseline", "f1": f1_base})
        for d, pq in _sweep_parquets(aoi, distances).items():
            rows.append({"aoi": aoi, "distance_m": d, "f1": _f1_point(pq)})
    f1_df = pd.DataFrame(rows)

    print("\n### Micro-F1 by AOI and min_marker_distance_m\n")
    pivot = f1_df.pivot_table(index="aoi", columns="distance_m",
                               values="f1", aggfunc="first")
    col_order = ["baseline"] + distances
    pivot = pivot[col_order]
    headers = ["AOI"] + ["baseline" if c == "baseline" else f"{c} m"
                        for c in col_order]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for aoi, row in pivot.iterrows():
        cells = [AOI_LABEL[aoi]] + [f"{row[c]:.3f}" for c in col_order]
        print("| " + " | ".join(cells) + " |")

    # LOAO: for each held-out AOI, pick the distance that maximises
    # mean F1 across the other three AOIs. Report F1, merge_rate,
    # split_rate side-by-side with CIs. CI-disjoint flag indicates
    # whether the F1 improvement is statistically clean.
    print("\n### Leave-one-AOI-out sweep\n")
    print("| held-out AOI | chosen distance | metric | baseline [CI] | "
          "watershed [CI] | delta | CIs disjoint? |")
    print("|---|---|---|---|---|---|---|")
    loao_choices: list[int] = []
    for held_out in AOIS:
        others = [a for a in AOIS if a != held_out]
        means = {}
        for d in distances:
            vals = [
                _f1_point(_sweep_parquets(aoi, [d])[d])
                for aoi in others
            ]
            means[d] = float(np.nanmean(vals))
        best_d = max(means, key=lambda d: means[d])
        loao_choices.append(best_d)
        chosen_pq = _sweep_parquets(held_out, [best_d])[best_d]
        base_pq   = _baseline_parquet(held_out)
        for metric_name, label in (
            ("f1",         "F1 (micro)"),
            ("merge_rate", "merge_rate"),
            ("split_rate", "split_rate"),
        ):
            if metric_name == "f1":
                b_p, b_lo, b_hi = _f1_ci(base_pq)
                c_p, c_lo, c_hi = _f1_ci(chosen_pq)
            else:
                b_p, b_lo, b_hi = _mean_metric_ci(base_pq, metric_name)
                c_p, c_lo, c_hi = _mean_metric_ci(chosen_pq, metric_name)
            delta = c_p - b_p
            # For merge/split, lower is better by design. Mark CI-disjoint
            # when the intervals don't overlap, regardless of direction.
            disjoint = c_lo > b_hi or c_hi < b_lo
            print(f"| {AOI_LABEL[held_out]} | {best_d} m | {label} | "
                  f"{b_p:.3f} [{b_lo:.3f}, {b_hi:.3f}] | "
                  f"{c_p:.3f} [{c_lo:.3f}, {c_hi:.3f}] | "
                  f"{delta:+.3f} | "
                  f"{'yes' if disjoint else 'no'} |")

    # Topline: does ANY setting beat the baseline on ANY AOI, by a
    # margin whose CIs clearly separate?
    print("\n### Does any (AOI, distance) combination beat the baseline?\n")
    print("| AOI | best distance in sweep | best F1 [CI] | baseline F1 [CI] "
          "| delta | CIs disjoint? |")
    print("|---|---|---|---|---|---|")
    clear_wins = 0
    beat_any_point = False
    for aoi in AOIS:
        best_d = max(distances, key=lambda d: _f1_point(_sweep_parquets(aoi, [d])[d]))
        best_point, best_lo, best_hi = _f1_ci(_sweep_parquets(aoi, [best_d])[best_d])
        base_point, base_lo, base_hi = _f1_ci(_baseline_parquet(aoi))
        delta = best_point - base_point
        if delta > 0:
            beat_any_point = True
        # CIs disjoint iff best_lo > base_hi (strict improvement) or
        # best_hi < base_lo (strict regression).
        disjoint = best_lo > base_hi or best_hi < base_lo
        if disjoint and delta > 0:
            clear_wins += 1
        print(f"| {AOI_LABEL[aoi]} | {best_d} m | "
              f"{best_point:.3f} [{best_lo:.3f}, {best_hi:.3f}] | "
              f"{base_point:.3f} [{base_lo:.3f}, {base_hi:.3f}] | "
              f"{delta:+.3f} | "
              f"{'yes' if disjoint else 'no'} |")

    print()
    if clear_wins > 0:
        print(f"**{clear_wins} / {len(AOIS)} AOIs: watershed beats baseline "
              f"with non-overlapping 95% CIs at some sweep distance.**")
    elif beat_any_point:
        print("**Some AOIs show a positive point-estimate delta, but no CI "
              "cleanly separates from the baseline.**")
    else:
        print("**No sweep setting beats baseline F1 on any AOI.**")

    if all(c == max(distances) for c in loao_choices):
        print(
            f"\n**NB: every LOAO choice is at {max(distances)} m, the "
            f"largest swept value. The sweep is probably not at the "
            f"peak -- rerun with larger --distances to find the "
            f"actual maximum.**"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
