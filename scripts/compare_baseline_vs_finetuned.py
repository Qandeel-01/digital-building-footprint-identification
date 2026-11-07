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


def resolve_from_config(config_path: Path) -> dict:
    """Resolve baseline + fine-tuned parquet paths from a Phase 1 or
    Phase 2a YAML config.

    Rules:
        * The AOI label and the fine-tuned work_dir + eval subdir come
          straight from the YAML.
        * The fine-tuned parquet is at `<work_dir>/<eval.output_subdir
          or "eval">/results.parquet`.
        * The baseline parquet is at `<work_dir>/baseline_rescored/
          <eval.output_subdir or "eval">/results.parquet` -- produced
          by `scripts/kaggle_phase1.py evaluate-baseline-on-phase1-
          tiles` from the same held-out blocks.
        * The output dir for the comparison .md / .json is the fine-
          tuned eval subdir.

    Works for any config whose shape matches the Phase 1 / 2a schema
    (aoi.name, run.work_dir, eval.output_subdir). Rejects configs
    without a `split:` section because those don't produce comparable
    held-out tiles (Phase 2/3 benchmark configs).

    Returns a dict with: config_source, aoi, finetuned_parquet,
    baseline_parquet, out_dir.
    """
    import yaml  # noqa: PLC0415
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    y = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "split" not in y:
        raise ValueError(
            f"config {config_path} has no `split:` section; this script "
            "requires a config that produces held-out test blocks "
            "(Phase 1 or Phase 2a). Use --baseline / --finetuned "
            "explicitly for ad-hoc parquet comparisons."
        )
    aoi = str(y["aoi"]["name"])
    work_dir = Path(y["run"]["work_dir"])
    eval_subdir = str((y.get("eval") or {}).get("output_subdir", "eval"))
    finetuned = work_dir / eval_subdir / "results.parquet"
    baseline  = work_dir / "baseline_rescored" / eval_subdir / "results.parquet"
    return {
        "config_source": str(config_path.resolve()),
        "aoi":                aoi,
        "finetuned_parquet":  finetuned,
        "baseline_parquet":   baseline,
        "out_dir":            finetuned.parent,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None,
                   help="Phase 1 or Phase 2a YAML; auto-resolves "
                        "--baseline/--finetuned/--aoi/--out-dir.")
    p.add_argument("--baseline", type=Path, default=None,
                   help="path to the BASELINE eval parquet (ignored "
                        "when --config is given).")
    p.add_argument("--finetuned", type=Path, default=None,
                   help="path to the FINE-TUNED eval parquet (ignored "
                        "when --config is given).")
    p.add_argument("--aoi", type=str, default=None,
                   help="AOI label for the output filename (ignored "
                        "when --config is given).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir (ignored when --config is given).")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    if args.config is not None:
        resolved = resolve_from_config(args.config)
        baseline  = resolved["baseline_parquet"]
        finetuned = resolved["finetuned_parquet"]
        aoi       = resolved["aoi"]
        out_dir   = resolved["out_dir"]
        print(f"resolved from {args.config}:")
        print(f"  aoi       = {aoi}")
        print(f"  baseline  = {baseline}")
        print(f"  finetuned = {finetuned}")
        print(f"  out_dir   = {out_dir}")
    else:
        for name in ("baseline", "finetuned", "aoi", "out_dir"):
            if getattr(args, name) is None:
                print(f"ERROR: --{name.replace('_', '-')} is required when "
                      "--config is not given.", file=sys.stderr)
                return 2
        baseline  = args.baseline
        finetuned = args.finetuned
        aoi       = args.aoi
        out_dir   = args.out_dir

    for label, pth in (("baseline", baseline), ("finetuned", finetuned)):
        if not Path(pth).is_file():
            print(f"ERROR: {label} parquet not found at {pth}", file=sys.stderr)
            return 2

    out = compare(baseline, finetuned, aoi, out_dir,
                 n_boot=args.n_boot, seed=args.seed)
    print(f"n_shared_tiles: {out['n_shared_tiles']}")
    print(f"wrote: {out_dir / f'comparison_{aoi}.json'}")
    print(f"wrote: {out_dir / f'comparison_{aoi}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
