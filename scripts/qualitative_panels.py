"""Qualitative 3-panel figure for the median and worst tiles.

Reads a bfbench eval Parquet + the corresponding work_dir (tiles/
imagery + predicted.geojson + reference.geojson) and produces one
figure with rows for MEDIAN and WORST tile by n_pred/n_ref count
ratio, and columns for imagery / prediction / reference.

No headline performance numbers appear in the figure. Only the tile
identifier and the count ratio, plus the polygon counts, are labelled.
This obeys the "no F1 / IoU numbers in README" constraint.
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

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mercantile  # noqa: E402
import numpy as np
import pandas as pd
import yaml
from PIL import Image

_LOG = logging.getLogger("bfbench.qualitative_panels")


def _tile_imagery(work_dir: Path, tile: mercantile.Tile,
                  channel_name: str = "images") -> np.ndarray | None:
    path = work_dir / "tiles" / channel_name / str(tile.z) / str(tile.x) / f"{tile.y}.png"
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def _plot_tile(ax, imagery: np.ndarray | None,
               tile: mercantile.Tile, gdf_wgs: gpd.GeoDataFrame | None,
               title: str, poly_edge: str, poly_fill: tuple[float, ...]) -> None:
    b = mercantile.bounds(tile)
    extent = (b.west, b.east, b.south, b.north)
    if imagery is not None:
        ax.imshow(imagery, extent=extent, origin="upper")
    if gdf_wgs is not None and len(gdf_wgs) > 0:
        gdf_wgs.plot(ax=ax, facecolor=poly_fill, edgecolor=poly_edge, linewidth=1.2)
    ax.set_xlim(b.west, b.east)
    ax.set_ylim(b.south, b.north)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def _polygons_in_tile(gdf_wgs: gpd.GeoDataFrame,
                      tile: mercantile.Tile) -> gpd.GeoDataFrame:
    b = mercantile.bounds(tile)
    return gdf_wgs.cx[b.west:b.east, b.south:b.north]


def build_figure(parquet_path: Path, cfg_path: Path, out_path: Path,
                 aoi_label: str | None = None) -> Path:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    work_dir = Path(cfg["run"]["work_dir"])
    if aoi_label is None:
        aoi_label = cfg["aoi"]["name"]

    df = pd.read_parquet(parquet_path)

    # Per-tile n_pred / n_ref; rank by ratio (asc: worst = lowest).
    per_tile = (df.drop_duplicates(subset=["tile_z", "tile_x", "tile_y"])
                  [["tile_z", "tile_x", "tile_y", "n_pred", "n_ref"]]
                  .reset_index(drop=True))
    per_tile["count_ratio"] = per_tile["n_pred"] / per_tile["n_ref"].replace(0, np.nan)
    per_tile = per_tile.dropna(subset=["count_ratio"]).sort_values("count_ratio")

    worst_row = per_tile.iloc[0]
    median_row = per_tile.iloc[len(per_tile) // 2]

    ref_wgs = gpd.read_file(work_dir / "reference.geojson")
    if ref_wgs.crs is None:
        ref_wgs = ref_wgs.set_crs("EPSG:4326")
    pred_wgs = gpd.read_file(work_dir / "predicted.geojson")
    if pred_wgs.crs is None:
        pred_wgs = pred_wgs.set_crs("EPSG:4326")
    else:
        pred_wgs = pred_wgs.to_crs("EPSG:4326")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8),
                             constrained_layout=True)
    for row_i, (label, row) in enumerate([("worst",  worst_row),
                                          ("median", median_row)]):
        tile = mercantile.Tile(int(row["tile_x"]), int(row["tile_y"]),
                               int(row["tile_z"]))
        imagery = _tile_imagery(work_dir, tile)
        _plot_tile(axes[row_i, 0], imagery, tile, None,
                   f"{label}: imagery  z/x/y={tile.z}/{tile.x}/{tile.y}",
                   poly_edge="none", poly_fill=(0, 0, 0, 0))
        _plot_tile(axes[row_i, 1], imagery, tile,
                   _polygons_in_tile(pred_wgs, tile),
                   f"{label}: prediction  n_pred={int(row['n_pred'])}",
                   poly_edge="deeppink", poly_fill=(1, 0.08, 0.58, 0.35))
        _plot_tile(axes[row_i, 2], imagery, tile,
                   _polygons_in_tile(ref_wgs, tile),
                   f"{label}: reference   n_ref={int(row['n_ref'])}  "
                   f"ratio={row['count_ratio']:.2f}",
                   poly_edge="cyan", poly_fill=(0, 1, 1, 0.35))
    fig.suptitle(f"{aoi_label} qualitative panels -- worst and median tile by n_pred/n_ref",
                 fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path,
                   default=Path("runs/vegas_prepare/eval/results.parquet"))
    p.add_argument("--config", type=Path,
                   default=Path("configs/spacenet_vegas.yaml"))
    p.add_argument("--out", type=Path,
                   default=Path("runs/vegas_prepare/eval/qualitative_panels.png"))
    args = p.parse_args(argv)
    out = build_figure(args.parquet, args.config, args.out)
    _LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
