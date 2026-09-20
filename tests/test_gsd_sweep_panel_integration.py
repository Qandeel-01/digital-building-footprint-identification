"""Generate the GSD-sweep panel PNG under pytest (workaround for a
Windows matplotlib delay-load DLL 0xC06D007F crash on bare `python`
invocation of scripts/gsd_sweep_v2.py's panel step).

Skipped when the sweep results directory is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "runs" / "gsd_sweep_v2"
PARQUET_RESULTS = SWEEP / "results.json"
CFG = REPO / "configs" / "spacenet_vegas.yaml"

pytestmark = pytest.mark.skipif(
    not PARQUET_RESULTS.is_file(),
    reason="GSD sweep results.json not present; run scripts/gsd_sweep_v2.py first.",
)


def test_generate_gsd_sweep_panel_png():
    import json

    import geopandas as gpd
    import mercantile
    import numpy as np
    import pandas as pd
    import yaml
    from PIL import Image

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    work_dir = Path(cfg["run"]["work_dir"])
    payload = json.loads((work_dir / "sampled_tiles.json").read_text(encoding="utf-8"))
    tiles_z17 = [mercantile.Tile(x, y, z) for z, x, y in payload["tiles"][:20]]

    per_tile = pd.read_csv(SWEEP / "per_tile.csv")
    per_tile["count_ratio"] = per_tile["n_pred"] / per_tile["n_ref"].replace(0, np.nan)

    sub17 = per_tile[per_tile["zoom"] == 17].reset_index(drop=True).dropna(subset=["count_ratio"]).sort_values("count_ratio").reset_index(drop=True)
    median_row = sub17.iloc[len(sub17) // 2]
    median_tile = mercantile.Tile(int(median_row["tile_x"]),
                                  int(median_row["tile_y"]),
                                  int(median_row["tile_z"]))
    b = mercantile.bounds(median_tile)

    ref = gpd.read_file(work_dir / "reference.geojson")
    if ref.crs is None:
        ref = ref.set_crs("EPSG:4326")
    else:
        ref = ref.to_crs("EPSG:4326")
    ref_in_view = ref.cx[b.west:b.east, b.south:b.north]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for row, z in enumerate((16, 17, 18)):
        tiles_root = SWEEP / f"z{z}" / "tiles" / "images"
        pred_path = SWEEP / f"z{z}" / "predicted.geojson"
        pgdf = gpd.read_file(pred_path).to_crs("EPSG:4326") if pred_path.is_file() and pred_path.stat().st_size > 100 else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        pgdf_in = pgdf.cx[b.west:b.east, b.south:b.north]

        # Pick the tile at zoom z whose footprint contains the median z=17 centroid
        cx = 0.5 * (b.west + b.east)
        cy = 0.5 * (b.south + b.north)
        tile_z = mercantile.tile(cx, cy, z)
        img_path = tiles_root / str(z) / str(tile_z.x) / f"{tile_z.y}.png"
        img = np.asarray(Image.open(img_path).convert("RGB")) if img_path.is_file() else None
        tb = mercantile.bounds(tile_z)

        for col in range(3):
            ax = axes[row, col]
            if img is not None:
                ax.imshow(img, extent=(tb.west, tb.east, tb.south, tb.north))
            if col == 0:
                ax.set_title(f"z={z}  gsd~{(0.964, 0.482, 0.241)[row]:.2f}m  imagery")
            elif col == 1:
                for g in pgdf_in.geometry:
                    if g is None or g.is_empty:
                        continue
                    for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                        xs, ys = poly.exterior.xy
                        ax.fill(xs, ys, facecolor="deeppink", alpha=0.35,
                                edgecolor="deeppink", linewidth=1.0)
                ax.set_title(f"z={z}  pred  n={len(pgdf_in)}")
            else:
                for g in ref_in_view.geometry:
                    if g is None or g.is_empty:
                        continue
                    for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                        xs, ys = poly.exterior.xy
                        ax.fill(xs, ys, facecolor="cyan", alpha=0.35,
                                edgecolor="cyan", linewidth=1.0)
                ax.set_title(f"z={z}  ref   n={len(ref_in_view)}")
            ax.set_xlim(b.west, b.east); ax.set_ylim(b.south, b.north)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"GSD sweep -- median z=17 tile ({median_tile.x}, {median_tile.y})",
                 fontsize=12)
    out = SWEEP / "gsd_sweep_panel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()
    assert out.stat().st_size > 5_000
