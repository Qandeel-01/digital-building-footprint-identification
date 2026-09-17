"""Figure generator for the Phase 2 evaluation results.

Consumes ONE tidy long-format parquet produced by `bfbench eval`,
and writes four figures:

    1. F1 vs local building density   (one line per AOI, CI bands)
    2. F1 vs building-size decile     (one line per AOI)
    3. merge_rate vs local density
    4. qualitative 3-panel per AOI    (imagery / prediction / reference)
                                      at the median and worst tile

The script has TWO modes:

    real:   read the parquet at --results, plot from real numbers.

    synthetic: generate a placeholder dataset with the same schema
               and plot it. Every output file is unmistakably marked
               fake:
                    - filename ends in `_SYNTHETIC_PLACEHOLDER.png`
                    - title includes "SYNTHETIC DATA -- NOT A RESULT"
                    - a diagonal watermark is drawn across the axes

Synthetic mode is here so the plotting code can be tested in CI
BEFORE any real metric is implemented. Do not paste these images
into a report; the point of the visible marker is to make that
mistake self-correcting.

Usage:
    python scripts/figures.py --synthetic --out runs/figures
    python scripts/figures.py --results runs/spacenet_vegas/eval/results.parquet \\
                              --out runs/spacenet_vegas/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LOG = logging.getLogger("bfbench.figures")

# ---------------------------------------------------------------------------
# Synthetic-data helper
# ---------------------------------------------------------------------------

def _synth_results(n_tiles_per_aoi: int = 150, seed: int = 42) -> pd.DataFrame:
    """Fake tidy DataFrame with the schema bfbench.eval writes.

    Reference columns:
        aoi, tile_z, tile_x, tile_y, metric, value, status, note,
        n_pred, n_ref, median_ref_area_m2, ref_density_per_ha

    Values are made up (uniform in [0.3, 0.9], correlated with density
    just enough that the plots have visible shape). Do not use these
    numbers for anything.
    """
    rng = np.random.default_rng(seed)
    aois = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
    rows = []
    for aoi in aois:
        aoi_offset = rng.uniform(-0.05, 0.05)  # AOI-specific baseline
        for i in range(n_tiles_per_aoi):
            density = rng.uniform(5.0, 120.0)
            med_area = float(rng.lognormal(mean=4.5, sigma=0.5))  # 40..300 m2
            base_f1 = 0.75 + aoi_offset - 0.002 * (density - 40.0) + rng.normal(0, 0.05)
            base_f1 = float(np.clip(base_f1, 0.0, 1.0))
            merge = float(np.clip(0.02 + 0.001 * density + rng.normal(0, 0.02), 0.0, 1.0))
            base = {
                "aoi": aoi,
                "tile_z": 17, "tile_x": 10000 + i, "tile_y": 20000,
                "n_pred": int(density * 0.9),
                "n_ref":  int(density),
                "median_ref_area_m2": med_area,
                "ref_density_per_ha": density,
                "status": "ok", "note": "",
            }
            rows.append({**base, "metric": "instance_prf.f1",   "value": base_f1})
            rows.append({**base, "metric": "merge_rate",         "value": merge})
            rows.append({**base, "metric": "pixel_iou",
                         "value": float(np.clip(base_f1 - 0.05 + rng.normal(0, 0.03), 0, 1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Watermarking / placeholder labelling
# ---------------------------------------------------------------------------

def _watermark(ax, text: str = "SYNTHETIC DATA -- NOT A RESULT") -> None:
    """Diagonal watermark across the current axes. Loud on purpose."""
    ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes,
        fontsize=28, color="red",
        rotation=30, ha="center", va="center",
        alpha=0.3, weight="bold",
    )


def _suffix_and_title(synthetic: bool, title: str) -> tuple[str, str]:
    if synthetic:
        return "_SYNTHETIC_PLACEHOLDER", f"{title}  [SYNTHETIC DATA -- NOT A RESULT]"
    return "", title


# ---------------------------------------------------------------------------
# Figure implementations
# ---------------------------------------------------------------------------

def _bin_and_aggregate(df: pd.DataFrame, x_col: str, y_col: str,
                       n_bins: int = 5) -> pd.DataFrame:
    """Quantile-bin by `x_col`, per-AOI, and aggregate `y_col`.

    Returns a DataFrame with columns (aoi, bin_mid, y_mean, y_lo, y_hi, n).
    The confidence band is a naive bootstrap over each bin: implementers
    may want to swap in the metrics-module's stratify_* once implemented.
    """
    out_rows = []
    for aoi, sub in df.groupby("aoi"):
        try:
            bins = pd.qcut(sub[x_col], q=n_bins, duplicates="drop")
        except ValueError:
            continue
        for _, group in sub.groupby(bins, observed=True):
            if len(group) == 0:
                continue
            vals = group[y_col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            rng = np.random.default_rng(0)
            n_boot = 200
            boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
            out_rows.append({
                "aoi": aoi,
                "bin_mid": float(group[x_col].median()),
                "y_mean": float(vals.mean()),
                "y_lo": float(np.percentile(boots, 2.5)),
                "y_hi": float(np.percentile(boots, 97.5)),
                "n": int(len(vals)),
            })
    return pd.DataFrame(out_rows)


def _fig_f1_vs_density(df: pd.DataFrame, out_dir: Path,
                       synthetic: bool) -> None:
    f1 = df[df["metric"] == "instance_prf.f1"].copy()
    if f1.empty:
        _LOG.warning("no instance_prf.f1 rows -- skipping F1-vs-density figure")
        return
    agg = _bin_and_aggregate(f1, "ref_density_per_ha", "value")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for aoi, sub in agg.groupby("aoi"):
        sub = sub.sort_values("bin_mid")
        ax.plot(sub["bin_mid"], sub["y_mean"], marker="o", label=aoi)
        ax.fill_between(sub["bin_mid"], sub["y_lo"], sub["y_hi"], alpha=0.15)

    ax.set_xlabel("Local reference building density (per ha)")
    ax.set_ylabel("F1")
    suffix, title = _suffix_and_title(synthetic, "F1 vs local building density")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if synthetic:
        _watermark(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"f1_vs_density{suffix}.png", dpi=150)
    plt.close(fig)


def _fig_f1_vs_size_decile(df: pd.DataFrame, out_dir: Path,
                            synthetic: bool) -> None:
    f1 = df[df["metric"] == "instance_prf.f1"].copy()
    if f1.empty:
        return
    agg = _bin_and_aggregate(f1, "median_ref_area_m2", "value", n_bins=10)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for aoi, sub in agg.groupby("aoi"):
        sub = sub.sort_values("bin_mid")
        ax.plot(sub["bin_mid"], sub["y_mean"], marker="o", label=aoi)
        ax.fill_between(sub["bin_mid"], sub["y_lo"], sub["y_hi"], alpha=0.15)

    ax.set_xlabel("Median reference building area (m2) -- per-tile")
    ax.set_ylabel("F1")
    suffix, title = _suffix_and_title(synthetic, "F1 vs building-size decile")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    if synthetic:
        _watermark(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"f1_vs_size_decile{suffix}.png", dpi=150)
    plt.close(fig)


def _fig_merge_vs_density(df: pd.DataFrame, out_dir: Path,
                          synthetic: bool) -> None:
    merge = df[df["metric"] == "merge_rate"].copy()
    if merge.empty:
        _LOG.warning("no merge_rate rows -- skipping merge-vs-density figure")
        return
    agg = _bin_and_aggregate(merge, "ref_density_per_ha", "value")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for aoi, sub in agg.groupby("aoi"):
        sub = sub.sort_values("bin_mid")
        ax.plot(sub["bin_mid"], sub["y_mean"], marker="s", label=aoi)
        ax.fill_between(sub["bin_mid"], sub["y_lo"], sub["y_hi"], alpha=0.15)

    ax.set_xlabel("Local reference building density (per ha)")
    ax.set_ylabel("Merge rate")
    suffix, title = _suffix_and_title(synthetic, "Merge rate vs local density")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if synthetic:
        _watermark(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"merge_vs_density{suffix}.png", dpi=150)
    plt.close(fig)


def _fig_qualitative_panels(df: pd.DataFrame, out_dir: Path,
                            synthetic: bool) -> None:
    """3-panel figure (imagery / prediction / reference) for the median
    and worst tile per AOI.

    In synthetic mode this uses coloured placeholder rectangles (no
    real imagery on disk). In real mode this reads the tile PNG from
    `<work_dir>/tiles/images/<z>/<x>/<y>.png` and the corresponding
    predicted/reference geojson clip -- BUT that plumbing is left for
    the implementer, because the geojson-to-raster overlay depends on
    which rasterisation snapping the pixel_iou implementer picked.
    """
    f1 = df[df["metric"] == "instance_prf.f1"].copy()
    if f1.empty:
        return

    aois = sorted(f1["aoi"].unique())
    fig, axes = plt.subplots(nrows=len(aois), ncols=6,
                             figsize=(14, 2.6 * len(aois)))
    if len(aois) == 1:
        axes = axes[None, :]

    for row, aoi in enumerate(aois):
        sub = f1[f1["aoi"] == aoi].sort_values("value")
        worst = sub.iloc[0] if len(sub) else None
        median = sub.iloc[len(sub) // 2] if len(sub) else None

        for col_offset, (label, tile) in enumerate([("worst", worst),
                                                    ("median", median)]):
            for panel_idx, panel_name in enumerate(("imagery", "prediction", "reference")):
                ax = axes[row, col_offset * 3 + panel_idx]
                ax.set_xticks([])
                ax.set_yticks([])
                if tile is None:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes)
                else:
                    ax.add_patch(plt.Rectangle(
                        (0.05, 0.05), 0.9, 0.9,
                        color={"imagery": "#556B7A",
                               "prediction": "#88AABB",
                               "reference": "#AA88BB"}[panel_name],
                        alpha=0.5, transform=ax.transAxes,
                    ))
                    ax.text(0.5, 0.5,
                            f"{panel_name}\nz={int(tile['tile_z'])}\n"
                            f"x={int(tile['tile_x'])}, y={int(tile['tile_y'])}\n"
                            f"F1={tile['value']:.2f}",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=8)
                if panel_idx == 0:
                    ax.set_ylabel(f"{aoi}\n({label})", fontsize=9)
                if row == 0:
                    ax.set_title(panel_name, fontsize=10)
                if synthetic:
                    _watermark(ax, "SYNTHETIC")

    suffix, title = _suffix_and_title(synthetic,
        "Qualitative panels: imagery / prediction / reference")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"qualitative_panels{suffix}.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=None,
                   help="Path to tidy Parquet produced by `bfbench eval`.")
    p.add_argument("--synthetic", action="store_true",
                   help="Ignore --results; generate + plot synthetic placeholder data. "
                        "Outputs are unmistakably marked FAKE.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for PNGs.")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        df = _synth_results()
        _LOG.info("SYNTHETIC mode: %d rows generated.", len(df))
    else:
        if args.results is None or not args.results.is_file():
            raise SystemExit("--results is required unless --synthetic is set.")
        df = pd.read_parquet(args.results)
        _LOG.info("read %d rows from %s", len(df), args.results)

    if args.synthetic:
        # augment df with the density/area columns that the eval schema
        # provides in real mode via the summary step
        pass
    else:
        for col in ("ref_density_per_ha", "median_ref_area_m2"):
            if col not in df.columns:
                _LOG.warning(
                    "column %r missing from parquet; the eval harness "
                    "needs to be extended to write it before this figure "
                    "makes sense in real mode.", col,
                )

    _fig_f1_vs_density(df, args.out, synthetic=args.synthetic)
    _fig_f1_vs_size_decile(df, args.out, synthetic=args.synthetic)
    _fig_merge_vs_density(df, args.out, synthetic=args.synthetic)
    _fig_qualitative_panels(df, args.out, synthetic=args.synthetic)
    _LOG.info("wrote figures to %s", args.out)


if __name__ == "__main__":
    main()
