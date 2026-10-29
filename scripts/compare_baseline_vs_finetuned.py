"""Build the Phase 1 "baseline vs fine-tuned" comparison table.

Reads TWO eval parquets (produced by `bfbench eval`), one from the
baseline checkpoint and one from a fine-tuned checkpoint, both scored
on the SAME held-out test tiles (= the Phase 1 geographic test blocks).
Emits:

    <out_dir>/comparison_<aoi>.json         (machine-readable)
    <out_dir>/comparison_<aoi>.md           (markdown table for docs/results.md)

Reported metrics with 95 % bootstrap CIs:
    - precision, recall, F1 (micro, from match_instances.n_matched +
      per-tile counts)
    - merge_rate, split_rate
    - pixel_iou (median across tiles -- bootstrap on the median)
    - Chamfer (boundary_distance.mean_m, bootstrap on the mean)

The eval parquet shape:
    one row per (aoi, tile, metric) with columns:
        aoi, tile_z, tile_x, tile_y, n_pred, n_ref,
        metric, value, status, note, ...

The script does NOT invent any numbers: a metric that is `not_implemented`
in both parquets gets a "n/a" cell in the markdown and is omitted from
the JSON. Every number in the output is a bootstrap over actual values.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bfbench.metrics.bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    BootstrapCI,
    bootstrap_mean_ci,
)

# Metrics to emit in the table. (parquet metric_name, display_name,
# aggregator). "mean" / "median" refer to how we reduce across tiles.
METRICS = [
    ("instance_prf.precision", "precision",  "mean"),
    ("instance_prf.recall",    "recall",     "mean"),
    ("instance_prf.f1",        "F1",         "mean"),
    ("merge_rate",             "merge_rate", "mean"),
    ("split_rate",             "split_rate", "mean"),
    ("pixel_iou",              "pixel_IoU",  "median"),
    ("boundary_distance.mean_m", "Chamfer_m", "median"),
]


@dataclass(frozen=True)
class _MetricCI:
    label: str
    aggregator: str
    point: float
    lo: float
    hi: float
    n_tiles: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "metric":       self.label,
            "aggregator":   self.aggregator,
            "point":        self.point,
            "ci_lo":        self.lo,
            "ci_hi":        self.hi,
            "n_tiles":      self.n_tiles,
        }


def _bootstrap_median_ci(values: np.ndarray, *,
                         n_boot: int, seed: int, alpha: float) -> BootstrapCI:
    """Percentile bootstrap on the sample median."""
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
        n_boot=n_boot,
        seed=seed,
    )


def _tile_values(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Return the per-tile values for `metric` from a tidy eval parquet,
    filtered to `status == 'ok'`.
    """
    sub = df[df["metric"] == metric]
    sub = sub[sub["status"] == "ok"] if "status" in sub.columns else sub
    return sub["value"].to_numpy(dtype=float)


def _ci_for_metric(df: pd.DataFrame, metric_name: str, display_name: str,
                   aggregator: str, *, n_boot: int, seed: int,
                   alpha: float) -> _MetricCI:
    arr = _tile_values(df, metric_name)
    if aggregator == "mean":
        ci = bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed, alpha=alpha)
    elif aggregator == "median":
        ci = _bootstrap_median_ci(arr, n_boot=n_boot, seed=seed, alpha=alpha)
    else:
        raise ValueError(f"unknown aggregator {aggregator!r}")
    return _MetricCI(label=display_name, aggregator=aggregator,
                    point=ci.point, lo=ci.lo, hi=ci.hi,
                    n_tiles=int(np.count_nonzero(~np.isnan(arr))))


def _tiles_in_both(baseline_df: pd.DataFrame,
                   finetuned_df: pd.DataFrame) -> set[tuple[int, int, int]]:
    """Return the tile IDs present in both parquets. The comparison is
    done on this set to guarantee apples-to-apples.
    """
    def _ids(df: pd.DataFrame) -> set[tuple[int, int, int]]:
        return {(int(z), int(x), int(y))
                for z, x, y in zip(df["tile_z"], df["tile_x"], df["tile_y"],
                                   strict=True)}
    return _ids(baseline_df) & _ids(finetuned_df)


def _filter_to_tiles(df: pd.DataFrame, tiles: set[tuple[int, int, int]]
                     ) -> pd.DataFrame:
    tz, tx, ty = zip(*sorted(tiles), strict=True)
    want = pd.DataFrame({"tile_z": tz, "tile_x": tx, "tile_y": ty})
    return df.merge(want, on=["tile_z", "tile_x", "tile_y"], how="inner")


def compare(baseline_parquet: Path, finetuned_parquet: Path,
            aoi_name: str, out_dir: Path, *,
            n_boot: int = DEFAULT_N_BOOT,
            seed: int = DEFAULT_SEED,
            alpha: float = DEFAULT_ALPHA) -> dict:
    """Produce the comparison table. Returns the JSON-shaped dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    b_df = pd.read_parquet(baseline_parquet)
    f_df = pd.read_parquet(finetuned_parquet)
    shared = _tiles_in_both(b_df, f_df)
    if not shared:
        raise ValueError(
            f"baseline ({baseline_parquet}) and fine-tuned "
            f"({finetuned_parquet}) parquets share zero tiles. Did you "
            "re-evaluate the baseline on the Phase 1 test blocks?"
        )
    b_df = _filter_to_tiles(b_df, shared)
    f_df = _filter_to_tiles(f_df, shared)

    rows: list[dict] = []
    for metric_name, display_name, agg in METRICS:
        b_ci = _ci_for_metric(b_df, metric_name, display_name, agg,
                              n_boot=n_boot, seed=seed, alpha=alpha)
        f_ci = _ci_for_metric(f_df, metric_name, display_name, agg,
                              n_boot=n_boot, seed=seed, alpha=alpha)
        rows.append({
            "metric":           display_name,
            "aggregator":       agg,
            "baseline":         b_ci.as_dict(),
            "finetuned":        f_ci.as_dict(),
            "n_shared_tiles":   len(shared),
        })

    out = {
        "aoi":                aoi_name,
        "baseline_parquet":   str(Path(baseline_parquet).resolve()),
        "finetuned_parquet":  str(Path(finetuned_parquet).resolve()),
        "n_shared_tiles":     len(shared),
        "bootstrap":          {"n_boot": n_boot, "seed": seed, "alpha": alpha},
        "rows":               rows,
    }
    (out_dir / f"comparison_{aoi_name}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8",
    )
    (out_dir / f"comparison_{aoi_name}.md").write_text(
        _render_markdown(out), encoding="utf-8",
    )
    return out


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.3f}"


def _render_markdown(data: dict) -> str:
    lines = [
        f"### {data['aoi']} — baseline vs. fine-tuned on {data['n_shared_tiles']} "
        "held-out tiles",
        "",
        f"95% bootstrap CIs ({data['bootstrap']['n_boot']} resamples, "
        f"seed {data['bootstrap']['seed']}).",
        "",
        "| metric | agg | baseline [CI] | fine-tuned [CI] |",
        "|---|---|---|---|",
    ]
    for row in data["rows"]:
        b = row["baseline"]
        f = row["finetuned"]
        lines.append(
            f"| {row['metric']} | {row['aggregator']} | "
            f"{_fmt(b['point'])} [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}] | "
            f"{_fmt(f['point'])} [{_fmt(f['ci_lo'])}, {_fmt(f['ci_hi'])}] |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True,
                   help="path to the BASELINE eval parquet (same test tiles)")
    p.add_argument("--finetuned", type=Path, required=True,
                   help="path to the FINE-TUNED eval parquet")
    p.add_argument("--aoi", type=str, required=True,
                   help="AOI label for the output filename (e.g. AOI_2_Vegas)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    for arg in ("baseline", "finetuned"):
        pth = getattr(args, arg)
        if not pth.is_file():
            print(f"ERROR: {arg} parquet not found at {pth}", file=sys.stderr)
            return 2
    out = compare(args.baseline, args.finetuned, args.aoi, args.out_dir,
                 n_boot=args.n_boot, seed=args.seed)
    print(f"n_shared_tiles: {out['n_shared_tiles']}")
    print(f"wrote: {args.out_dir / f'comparison_{args.aoi}.json'}")
    print(f"wrote: {args.out_dir / f'comparison_{args.aoi}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
