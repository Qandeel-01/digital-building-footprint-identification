"""Phase 2 overlay figure: merge rate vs local density (200 m window)
with four series (baseline, Phase 1, Phase 2a, Phase 2b) and 95 %
bootstrap CI bands per bin.

Reads up to four eval parquets (missing ones are skipped, so the plot
is usable at every stage of the rollout).

The baseline curve matches the existing
`docs/results-figures/merge_rate_vs_density_w200.png` so a reader can
compare directly.

Usage
-----
    python scripts/merge_rate_overlay.py \\
        --baseline  runs/vegas_prepare/eval/results.parquet \\
        --phase1    runs/vegas_prepare/eval_phase1/results.parquet \\
        --phase2a   runs/vegas_prepare/eval_phase2a/results.parquet \\
        --phase2b   runs/vegas_prepare/eval_phase2b/results.parquet \\
        --out       docs/results-figures/merge_rate_overlay_vegas.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bfbench.metrics.bootstrap import bootstrap_mean_ci  # noqa: E402

_LOG = logging.getLogger("bfbench.merge_rate_overlay")

SERIES_COLOURS = {
    "baseline": "#1f77b4",   # blue
    "Phase 1":  "#2ca02c",   # green
    "Phase 2a": "#ff7f0e",   # orange
    "Phase 2b": "#d62728",   # red
}


def _pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Same shape as scripts/aoi_summary.py's pivot: one row per tile,
    columns per metric. Keep enrichment columns intact."""
    tile_cols = ["aoi", "tile_z", "tile_x", "tile_y",
                 "ref_density_per_ha_w200"]
    ok = df[df["status"] == "ok"]
    return ok.pivot_table(
        index=tile_cols, columns="metric", values="value", aggfunc="first",
    ).reset_index()


def _bin_merge_rate(wide: pd.DataFrame, n_bins: int = 5
                   ) -> list[tuple[float, float, float, int]]:
    """For each density quantile bin, return (bin_centre, merge_rate,
    ci_half_width, n_tiles). Uses bootstrap_mean_ci."""
    if "ref_density_per_ha_w200" not in wide.columns or "merge_rate" not in wide.columns:
        return []
    wide = wide.dropna(subset=["ref_density_per_ha_w200", "merge_rate"])
    if len(wide) == 0:
        return []
    try:
        bins = pd.qcut(wide["ref_density_per_ha_w200"], q=n_bins,
                       duplicates="drop")
    except ValueError:
        return []
    out = []
    for interval, group in wide.groupby(bins, observed=True):
        values = group["merge_rate"].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        ci = bootstrap_mean_ci(values, n_boot=1000, seed=0)
        half = (ci.hi - ci.lo) / 2.0
        centre = 0.5 * (interval.left + interval.right)
        out.append((float(centre), float(ci.point), float(half), int(len(values))))
    return out


def plot(parquets: dict[str, Path], out_png: Path, title: str = "") -> Path:
    """Overlay merge rate vs density_w200 for the given series.

    `parquets` is a dict `{series_label: parquet_path}`. Missing files
    silently skip that series.
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    any_plotted = False
    for label, pq in parquets.items():
        if pq is None or not Path(pq).is_file():
            _LOG.info("skip %s: parquet not on disk (%s)", label, pq)
            continue
        df = pd.read_parquet(pq)
        wide = _pivot_wide(df)
        bins = _bin_merge_rate(wide)
        if not bins:
            _LOG.warning("no density/merge data for %s in %s", label, pq)
            continue
        xs = np.array([b[0] for b in bins])
        ys = np.array([b[1] for b in bins])
        hs = np.array([b[2] for b in bins])
        colour = SERIES_COLOURS.get(label)
        ax.errorbar(xs, ys, yerr=hs, marker="o", capsize=3, label=label,
                    color=colour, linewidth=1.6)
        any_plotted = True

    if not any_plotted:
        # Produce a labelled empty figure rather than crashing; useful
        # when the figure is regenerated after each phase.
        ax.text(0.5, 0.5,
                "No eval parquets available for the four Phase 2 series.\n"
                "Run bfbench eval after each phase.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    else:
        ax.set_xlabel("local building density (per ha, 200 m window)")
        ax.set_ylabel("merge rate (mean, 95% bootstrap CI)")
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("wrote %s", out_png)
    return out_png


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=None)
    p.add_argument("--phase1",   type=Path, default=None)
    p.add_argument("--phase2a",  type=Path, default=None)
    p.add_argument("--phase2b",  type=Path, default=None)
    p.add_argument("--out",      type=Path, required=True)
    p.add_argument("--title",    type=str, default="")
    args = p.parse_args()
    plot(
        {
            "baseline": args.baseline,
            "Phase 1":  args.phase1,
            "Phase 2a": args.phase2a,
            "Phase 2b": args.phase2b,
        },
        args.out, title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
