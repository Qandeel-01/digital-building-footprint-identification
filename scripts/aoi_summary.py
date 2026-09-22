"""Aggregate per-AOI numbers from a bfbench eval parquet.

Reports AOI-level:
    n_tiles, sum n_pred, sum n_ref,
    precision, recall, F1, TP/FP/FN totals,
    merge rate, split rate (weighted by tile),
    pixel_iou (mean, median),
    count error (absolute + AOI relative).

Also emits stratified breakdowns:
    * by median_ref_area_m2 decile
    * by ref_density_per_ha at each window in {100, 200, 500} m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the tidy long-format parquet into a per-tile wide table."""
    tile_cols = ["aoi", "tile_z", "tile_x", "tile_y", "n_pred", "n_ref",
                 "median_ref_area_m2",
                 "ref_density_per_ha_w100", "ref_density_per_ha_w200",
                 "ref_density_per_ha_w500", "ref_density_per_ha"]
    ok = df[df["status"] == "ok"].copy()
    wide = ok.pivot_table(
        index=tile_cols, columns="metric", values="value", aggfunc="first",
    ).reset_index()
    return wide


def _aoi_summary(wide: pd.DataFrame) -> dict:
    """AOI-level PRF from summed TP/FP/FN (micro-average) + tile means."""
    n_tiles = len(wide)
    tp = int(wide["instance_prf.tp"].sum())
    fp = int(wide["instance_prf.fp"].sum())
    fn = int(wide["instance_prf.fn"].sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = 0.0 if not (np.isnan(precision) or np.isnan(recall)) else float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    n_pred = int(wide["n_pred"].sum())
    n_ref  = int(wide["n_ref"].sum())
    signed = n_pred - n_ref
    rel = signed / n_ref if n_ref > 0 else float("nan")

    def _stat(col: str) -> dict:
        s = wide[col].dropna().astype(float)
        return {
            "n":     int(len(s)),
            "mean":  float(s.mean()) if len(s) else float("nan"),
            "median":float(s.median()) if len(s) else float("nan"),
            "p10":   float(s.quantile(0.10)) if len(s) else float("nan"),
            "p90":   float(s.quantile(0.90)) if len(s) else float("nan"),
        }

    return {
        "n_tiles":  n_tiles,
        "n_pred":   n_pred,
        "n_ref":    n_ref,
        "count_ratio_aoi": n_pred / n_ref if n_ref > 0 else float("nan"),
        "count_signed_error":   signed,
        "count_relative_error": rel,
        "prf_micro": {"tp": tp, "fp": fp, "fn": fn,
                      "precision": precision, "recall": recall, "f1": f1},
        "merge_rate_mean":       _stat("merge_rate"),
        "split_rate_mean":       _stat("split_rate"),
        "pixel_iou":             _stat("pixel_iou"),
        "pixel_iou_all_touched": _stat("pixel_iou_all_touched"),
        "instance_prf.f1_per_tile":         _stat("instance_prf.f1"),
        "instance_prf.precision_per_tile":  _stat("instance_prf.precision"),
        "instance_prf.recall_per_tile":     _stat("instance_prf.recall"),
        "boundary_distance.chamfer":        _stat("boundary_distance.mean_chamfer"),
        "boundary_distance.hausdorff":      _stat("boundary_distance.mean_hausdorff"),
    }


def _stratified(wide: pd.DataFrame, x_col: str, n_bins: int = 5) -> pd.DataFrame:
    if x_col not in wide.columns:
        return pd.DataFrame()
    try:
        bins = pd.qcut(wide[x_col], q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    metric_cols = ["instance_prf.f1", "instance_prf.precision",
                   "instance_prf.recall", "merge_rate", "split_rate",
                   "pixel_iou", "pixel_iou_all_touched"]
    rows = []
    for interval, group in wide.groupby(bins, observed=True):
        row = {
            "aoi": group["aoi"].iloc[0] if len(group) else "",
            "bin":         str(interval),
            "bin_left":    float(interval.left),
            "bin_right":   float(interval.right),
            "n_tiles":     int(len(group)),
        }
        for m in metric_cols:
            if m in group.columns:
                v = group[m].dropna().astype(float)
                row[f"{m}_mean"]   = float(v.mean()) if len(v) else float("nan")
                row[f"{m}_median"] = float(v.median()) if len(v) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", type=Path)
    p.add_argument("--out",  type=Path, default=None,
                   help="If set, write the summary JSON here.")
    p.add_argument("--stratified-dir", type=Path, default=None,
                   help="If set, write stratified CSVs into this dir.")
    args = p.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    wide = _pivot(df)

    summary = _aoi_summary(wide)
    aoi = wide["aoi"].iloc[0] if len(wide) else "UNKNOWN"
    summary = {"aoi": aoi, **summary}

    if args.stratified_dir is not None:
        args.stratified_dir.mkdir(parents=True, exist_ok=True)
        by_size = _stratified(wide, "median_ref_area_m2", n_bins=5)
        by_size.to_csv(args.stratified_dir / f"{aoi}_by_size_decile.csv", index=False)
        for w in (100, 200, 500):
            by_den = _stratified(wide, f"ref_density_per_ha_w{w}", n_bins=5)
            by_den.to_csv(args.stratified_dir / f"{aoi}_by_density_w{w}.csv", index=False)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
