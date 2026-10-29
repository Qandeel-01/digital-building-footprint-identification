"""Phase 4 real figures for docs/results.md.

Consumes per-AOI eval Parquets, concatenates them, and writes:

    * f1_vs_density_w{100,200,500}.png    -- F1 vs local density, one
                                             line per AOI, one plot per
                                             window size, with 95 % CI
                                             bootstrap bands.
    * f1_vs_size_decile.png               -- F1 vs median-ref-area
                                             quantile, one line per AOI.
    * merge_rate_vs_density_w200.png      -- merge rate vs local density
                                             at the 200 m window.

The qualitative 3-panel (imagery / prediction / reference) for the
median and worst tile is produced separately by
scripts/qualitative_panels.py per AOI.

This is a pytest-runnable script because bare python <script> on the
target Windows env segfaults matplotlib. See tests/
test_real_figures_integration.py.
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

_LOG = logging.getLogger("bfbench.real_figures")


def _load_and_pivot(parquets: list[Path]) -> pd.DataFrame:
    """Read every AOI parquet, pivot to per-tile wide form, concatenate."""
    frames = []
    tile_cols = ["aoi", "tile_z", "tile_x", "tile_y", "n_pred", "n_ref",
                 "median_ref_area_m2",
                 "ref_density_per_ha_w100", "ref_density_per_ha_w200",
                 "ref_density_per_ha_w500", "ref_density_per_ha"]
    for p in parquets:
        df = pd.read_parquet(p)
        ok = df[df["status"] == "ok"]
        wide = ok.pivot_table(
            index=tile_cols, columns="metric", values="value", aggfunc="first",
        ).reset_index()
        frames.append(wide)
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 500,
                  seed: int = 0) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _bin_and_agg(wide: pd.DataFrame, x_col: str, y_col: str,
                 n_bins: int = 5) -> pd.DataFrame:
    rows = []
    for aoi, sub in wide.groupby("aoi", sort=True):
        if len(sub) == 0 or x_col not in sub.columns:
            continue
        # Prefer AOI-local quantiles.
        try:
            bins = pd.qcut(sub[x_col], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        for interval, group in sub.groupby(bins, observed=True):
            vals = group[y_col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            lo, hi = _bootstrap_ci(vals)
            rows.append({
                "aoi":       aoi,
                "bin_left":  float(interval.left),
                "bin_right": float(interval.right),
                "bin_mid":   0.5 * (float(interval.left) + float(interval.right)),
                "mean":      float(vals.mean()),
                "ci_lo":     lo,
                "ci_hi":     hi,
                "n":         int(len(vals)),
            })
    return pd.DataFrame(rows)


def _plot_line_per_aoi(agg: pd.DataFrame, x_label: str, y_label: str,
                        title: str, out_path: Path, ylim: tuple[float, float] | None = (0, 1)) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for aoi, sub in agg.groupby("aoi", sort=True):
        sub = sub.sort_values("bin_mid")
        line, = ax.plot(sub["bin_mid"], sub["mean"], marker="o", label=aoi)
        ax.fill_between(sub["bin_mid"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.18, color=line.get_color())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_all_figures(parquets: list[Path], out_dir: Path) -> list[Path]:
    wide = _load_and_pivot(parquets)
    outs: list[Path] = []
    y_col = "instance_prf.f1"

    for w in (100, 200, 500):
        agg = _bin_and_agg(wide, f"ref_density_per_ha_w{w}", y_col)
        out = out_dir / f"f1_vs_density_w{w}.png"
        _plot_line_per_aoi(
            agg,
            x_label=f"Local reference density (bldgs/ha, {w} m window)",
            y_label="F1  (per-tile mean, 95% bootstrap CI)",
            title=f"F1 vs local density -- {w} m window",
            out_path=out,
        )
        outs.append(out)

    agg = _bin_and_agg(wide, "median_ref_area_m2", y_col)
    out = out_dir / "f1_vs_size_decile.png"
    _plot_line_per_aoi(
        agg,
        x_label="Per-tile median reference building area (m2)",
        y_label="F1  (per-tile mean, 95% bootstrap CI)",
        title="F1 vs building-size decile",
        out_path=out,
    )
    outs.append(out)

    agg = _bin_and_agg(wide, "ref_density_per_ha_w200", "merge_rate")
    out = out_dir / "merge_rate_vs_density_w200.png"
    _plot_line_per_aoi(
        agg,
        x_label="Local reference density (bldgs/ha, 200 m window)",
        y_label="Merge rate  (per-tile mean, 95% bootstrap CI)",
        title="Merge rate vs local density -- 200 m window",
        out_path=out,
    )
    outs.append(out)

    return outs


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, action="append", required=True,
                   help="Path to a bfbench eval parquet (repeat per AOI).")
    p.add_argument("--out", type=Path,
                   default=Path("docs/results-figures"))
    args = p.parse_args(argv)
    outs = build_all_figures(args.parquet, args.out)
    for o in outs:
        _LOG.info("wrote %s", o)


if __name__ == "__main__":
    main()
