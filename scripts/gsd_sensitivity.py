"""Run the same SpaceNet chip through the pipeline at three XYZ zooms
(z=16, 17, 18) and record polygon counts + a side-by-side visual panel.

This validates the Phase 1 adapter GSD decision empirically. It does
NOT compute metrics -- Phase 2 owns those. The output is descriptive:
counts and a picture.

Effective per-pixel GSD at each zoom, tile_size=512, lat=36.2 Vegas:
    z=16 -> 0.965 m/px (2x coarser than xBD training)
    z=17 -> 0.482 m/px (~ xBD training GSD; Phase 1 default)
    z=18 -> 0.241 m/px (2x finer than xBD training)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image

from bfbench.data.adapter import AdapterConfig, adapt_spacenet_image

REPO = Path(__file__).resolve().parent.parent


def _stretch(img):
    import numpy as np
    lo, hi = np.percentile(img, (2, 98))
    if hi <= lo:
        hi = lo + 1
    return ((img.astype(np.float32) - lo) * (255.0 / (hi - lo))).clip(0, 255).astype("uint8")


def _run_one(zoom: int, src_image: Path, out_root: Path, checkpoint: Path,
             abd_config: Path) -> dict:
    print(f"\n=== z={zoom} ===")
    tiles_root = out_root / "tiles"
    if tiles_root.exists():
        shutil.rmtree(tiles_root)
    tiles_root.mkdir(parents=True, exist_ok=True)

    cfg = AdapterConfig(target_gsd_m=0.5, tile_size=512, keep_borders=True,
                        zoom=zoom, zoom_tolerance=10.0)
    r = adapt_spacenet_image(src_image, tiles_root, config=cfg)
    print(f"  adapted: {r.tiles_written} tiles, effective_gsd={r.effective_gsd_m:.3f} m")

    masks_dir = out_root / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "predict_local.py"),
         "--dataset", str(tiles_root),
         "--checkpoint", str(checkpoint),
         "--config", str(abd_config),
         "--cover", str(tiles_root / "cover.csv"),
         "--out", str(masks_dir),
         "--metatiles", "--keep_borders"],
        check=True,
    )
    raw = out_root / "predicted_raw.geojson"
    subprocess.run(
        ["abd", "vectorize", "--masks", str(masks_dir),
         "--type", "Building", "--config", str(abd_config), "--out", str(raw)],
        check=True,
    )
    filt = out_root / "predicted.geojson"
    subprocess.run(
        ["filter-buildings", "--data", str(raw), "--dest", str(filt)],
        check=True,
    )
    gdf = gpd.read_file(filt).to_crs("EPSG:4326") if filt.stat().st_size > 100 else \
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return {
        "zoom": zoom,
        "effective_gsd_m": r.effective_gsd_m,
        "tiles_written": r.tiles_written,
        "pred_polys_final": int(len(gdf)),
        "geojson": str(filt),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path,
                   default=REPO / "runs" / "spacenet_vegas_sample" / "spacenet" / "PS-RGB" /
                   "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img101.tif")
    p.add_argument("--ref", type=Path,
                   default=REPO / "runs" / "spacenet_vegas_sample" / "spacenet" /
                   "geojson_buildings" /
                   "SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img101.geojson")
    p.add_argument("--checkpoint", type=Path, default=REPO / "weights" / "neat-fullxview-epoch75.pth")
    p.add_argument("--abd-config", type=Path, default=REPO / "external" / "abd" / "input" / "config.toml")
    p.add_argument("--out", type=Path, default=REPO / "runs" / "gsd_sensitivity")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for z in (16, 17, 18):
        r = _run_one(z, args.src, args.out / f"z{z}", args.checkpoint, args.abd_config)
        results.append(r)

    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Panel figure
    with rasterio.open(args.src) as src:
        arr = src.read(indexes=[1, 2, 3])
        img = _stretch(reshape_as_image(arr))
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
        image_crs = src.crs
    ref = gpd.read_file(args.ref)
    if ref.crs is None:
        ref.set_crs("EPSG:4326", inplace=True)
    ref = ref.to_crs(image_crs)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    axes[0].imshow(img, extent=extent, origin="upper")
    ref.plot(ax=axes[0], facecolor=(0, 1, 1, 0.30), edgecolor="cyan", linewidth=1.6)
    axes[0].set_title(f"SpaceNet reference ({len(ref)} polys)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    for i, r in enumerate(results, start=1):
        gdf = gpd.read_file(r["geojson"]).to_crs(image_crs) if Path(r["geojson"]).exists() else \
            gpd.GeoDataFrame(geometry=[], crs=image_crs)
        axes[i].imshow(img, extent=extent, origin="upper")
        gdf.plot(ax=axes[i], facecolor=(1, 0.08, 0.58, 0.35), edgecolor="deeppink", linewidth=1.6)
        axes[i].set_title(f"z={r['zoom']}  gsd={r['effective_gsd_m']:.2f} m  "
                          f"{len(gdf)} polys")
    out_png = args.out / "gsd_panel.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nwrote {out_png}")
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
