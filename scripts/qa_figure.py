"""Produce a single QA figure for a run: imagery / predicted polygons / SpaceNet
reference polygons, side by side.

This is a manual sanity-check, not a metric. It exists so a human can
eyeball whether the pipeline is producing plausible outputs before any
Phase-2 metric code is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image


def _load_rgb(image_path: Path) -> tuple[list, tuple[float, float, float, float]]:
    """Return (rgb_uint8_2d_or_3d, extent_in_image_crs)."""
    with rasterio.open(image_path) as src:
        arr = src.read(indexes=[1, 2, 3])
        img = reshape_as_image(arr)
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    # stretch p2/p98 for display, mirroring the adapter's behaviour
    import numpy as np
    lo, hi = np.percentile(img, (2, 98))
    if hi <= lo:
        hi = lo + 1
    disp = ((img.astype(np.float32) - lo) * (255.0 / (hi - lo))).clip(0, 255).astype("uint8")
    return disp, extent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True, help="SpaceNet PS-RGB .tif")
    p.add_argument("--pred", type=Path, required=True, help="predicted buildings geojson")
    p.add_argument("--ref", type=Path, required=True, help="SpaceNet reference buildings geojson")
    p.add_argument("--out", type=Path, required=True, help="output PNG")
    args = p.parse_args()

    img, extent = _load_rgb(args.image)

    with rasterio.open(args.image) as src:
        image_crs = src.crs

    gdf_pred = gpd.read_file(args.pred)
    if gdf_pred.crs is None:
        gdf_pred.set_crs("EPSG:4326", inplace=True)
    gdf_ref = gpd.read_file(args.ref)
    if gdf_ref.crs is None:
        gdf_ref.set_crs("EPSG:4326", inplace=True)

    gdf_pred = gdf_pred.to_crs(image_crs)
    gdf_ref = gdf_ref.to_crs(image_crs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax in axes:
        ax.imshow(img, extent=extent, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    axes[0].set_title("imagery")
    gdf_pred.plot(ax=axes[1], facecolor=(1.0, 0.078, 0.576, 0.35), edgecolor="deeppink",
                  linewidth=2.0)
    axes[1].set_title(f"predicted ({len(gdf_pred)} polys)")
    gdf_ref.plot(ax=axes[2], facecolor=(0.0, 1.0, 1.0, 0.30), edgecolor="cyan", linewidth=2.0)
    axes[2].set_title(f"SpaceNet reference ({len(gdf_ref)} polys)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
