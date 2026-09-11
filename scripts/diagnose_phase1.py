"""Diagnose the Phase 1 result of 6 predicted polygons vs 31 SpaceNet
reference polygons on the AOI_2_Vegas img101 sample tile.

Produces a set of facts and figures, not a narrative. Written into
`docs/phase1-diagnosis.md` by hand after reading this script's stdout
and the PNGs it drops under `runs/spacenet_vegas_sample/diagnosis/`.

Facts collected:
    1. Reference polygons whose centroid lies inside the tile footprint
       vs the chip footprint. The tile is the correct denominator.
    2. Nodata (zero) fraction of the input tile.
    3. Raw sigmoid probability histogram: min/max/mean, and the fraction
       of pixels in the [0.3, 0.7] uncertainty band.
    4. Re-run the model with the nodata region masked out at output time
       (as opposed to zero-filled at input time as Phase 1 did) and
       compare polygon counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import rasterio
import torch
from abd_model.core import load_config, make_palette
from abd_model.loaders.semseg import SemSeg
from abd_model.nn.albunet import Albunet
from abd_model.tiles import tile_label_to_file, tiles_from_csv
from PIL import Image
from rasterio.warp import transform_bounds
from shapely.geometry import box


def _strip_ddp_prefix(sd: dict) -> dict:
    return {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in sd.items()}


def _load_model(checkpoint: Path, device: torch.device):
    chkpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Albunet(chkpt["shape_in"], chkpt["shape_out"], chkpt["encoder"].lower())
    model.load_state_dict(_strip_ddp_prefix(chkpt["state_dict"]))
    return model.to(device).eval(), chkpt


def _count_ref_in(gdf: gpd.GeoDataFrame, poly, label: str) -> int:
    cents = gdf.geometry.centroid
    inside = cents.within(poly)
    n = int(inside.sum())
    print(f"  refs with centroid inside {label}: {n} / {len(gdf)}")
    return n


def _nodata_stats(tile_tiff: Path) -> dict:
    with rasterio.open(tile_tiff) as src:
        arr = src.read(indexes=[1, 2, 3])
    # A pixel is nodata iff all 3 bands are zero (that's how the adapter
    # zero-pads outside the reprojected raster).
    zero = np.all(arr == 0, axis=0)
    frac = float(zero.mean())
    print(f"  nodata (all-zero pixel) fraction: {frac:.4f}  ({zero.sum()}/{zero.size})")
    return {"fraction": frac, "mask_shape": list(zero.shape)}


def _sigmoid_probs(model, device, tile_tiff: Path) -> np.ndarray:
    """Run the model on ONE tile and return raw class-2 probability."""
    with rasterio.open(tile_tiff) as src:
        arr = src.read(indexes=[1, 2, 3])  # (3, H, W)
    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        logits = model(x).cpu().numpy()[0]  # (2, H, W) for [background, Building]
    return logits


def _hist_stats(probs: np.ndarray, mask_valid: np.ndarray | None) -> dict:
    """probs: (C,H,W) or (H,W). Use class-1 (Building) if C=2. Restrict to valid mask."""
    if probs.ndim == 3:
        p = probs[1]
    else:
        p = probs
    if mask_valid is not None:
        p = p[mask_valid]
    if p.size == 0:
        return {"n": 0}
    band = float(np.mean((p >= 0.3) & (p <= 0.7)))
    d = {
        "n": int(p.size),
        "min": float(p.min()),
        "max": float(p.max()),
        "mean": float(p.mean()),
        "p50": float(np.percentile(p, 50)),
        "p90": float(np.percentile(p, 90)),
        "p99": float(np.percentile(p, 99)),
        "frac_gt_0.5": float((p > 0.5).mean()),
        "frac_in_uncertainty_band_[0.3,0.7]": band,
    }
    print("  probability stats:")
    for k, v in d.items():
        print(f"    {k}: {v}")
    return d


def _plot_hist(probs: np.ndarray, out: Path, label: str) -> None:
    p = probs[1] if probs.ndim == 3 else probs
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(p.ravel(), bins=50, range=(0.0, 1.0))
    ax.set_xlabel("sigmoid p(Building)")
    ax.set_ylabel("pixel count")
    ax.set_title(f"raw probability histogram\n{label}")
    ax.axvspan(0.3, 0.7, alpha=0.15, color="grey", label="[0.3, 0.7] band")
    ax.legend()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_prob_map(probs: np.ndarray, out: Path, label: str) -> None:
    p = probs[1] if probs.ndim == 3 else probs
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(p, vmin=0, vmax=1, cmap="magma")
    ax.set_title(f"raw sigmoid p(Building)\n{label}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _threshold_and_vectorize(prob: np.ndarray, tile: mercantile.Tile, out_root: Path,
                             palette, transparency) -> int:
    """Threshold class-1 prob at 0.5, write label PNG, return positive-pixel count."""
    mask = (prob >= 0.5).astype(np.uint8)  # class index for Building = 1
    tile_label_to_file(str(out_root), tile, palette, transparency, mask)
    return int(mask.sum())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("runs/spacenet_vegas_sample"))
    p.add_argument("--checkpoint", type=Path, default=Path("weights/neat-fullxview-epoch75.pth"))
    p.add_argument("--abd-config", type=Path, default=Path("external/abd/input/config.toml"))
    args = p.parse_args()

    run = args.run
    diag = run / "diagnosis"
    diag.mkdir(parents=True, exist_ok=True)

    # ---- 1. tile vs chip footprint ----------------------------------------
    tiles = list(tiles_from_csv(str(run / "tiles" / "cover.csv")))
    assert len(tiles) == 1, f"expected exactly one tile, got {len(tiles)}"
    tile = tiles[0]
    print(f"tile: {tile}")

    xy = mercantile.xy_bounds(tile)  # EPSG:3857 metres
    wgs = mercantile.bounds(tile)    # EPSG:4326 deg
    tile_poly_wgs = box(wgs.west, wgs.south, wgs.east, wgs.north)
    print(f"tile bbox (EPSG:4326): W={wgs.west:.6f} S={wgs.south:.6f} "
          f"E={wgs.east:.6f} N={wgs.north:.6f}")
    print(f"tile bbox (EPSG:3857 m): "
          f"W={xy.left:.2f} S={xy.bottom:.2f} E={xy.right:.2f} N={xy.top:.2f}")

    ref_path = next((run / "spacenet" / "geojson_buildings").glob("*.geojson"))
    ref_gdf = gpd.read_file(ref_path).to_crs("EPSG:4326")
    print(f"\nreference polys total: {len(ref_gdf)}  (source: {ref_path.name})")

    # chip footprint from the source raster's own bounds
    img_path = next((run / "spacenet" / "PS-RGB").glob("*.tif"))
    with rasterio.open(img_path) as src:
        chip_bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    chip_poly = box(*chip_bounds)
    print(f"chip bbox (EPSG:4326): W={chip_bounds[0]:.6f} S={chip_bounds[1]:.6f} "
          f"E={chip_bounds[2]:.6f} N={chip_bounds[3]:.6f}")

    n_in_tile = _count_ref_in(ref_gdf, tile_poly_wgs, "tile")
    n_in_chip = _count_ref_in(ref_gdf, chip_poly, "chip")

    # ---- 2. nodata fraction -----------------------------------------------
    tile_tiff = run / "tiles" / "images" / str(tile.z) / str(tile.x) / f"{tile.y}.tiff"
    print(f"\nnodata stats on {tile_tiff}")
    nod = _nodata_stats(tile_tiff)

    # ---- 3. raw sigmoid probability ---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    model, chkpt = _load_model(args.checkpoint, device)
    print(f"checkpoint uuid={chkpt.get('uuid')} nn={chkpt['nn']} "
          f"shape_in={chkpt['shape_in']}")

    probs = _sigmoid_probs(model, device, tile_tiff)  # (2, 512, 512)
    print("\nraw prob stats (ALL pixels):")
    all_stats = _hist_stats(probs, mask_valid=None)

    # Valid mask = non-nodata pixels
    with rasterio.open(tile_tiff) as src:
        rgb = src.read(indexes=[1, 2, 3])
    valid_mask = ~np.all(rgb == 0, axis=0)

    print("\nraw prob stats (VALID pixels only):")
    valid_stats = _hist_stats(probs, mask_valid=valid_mask)

    _plot_hist(probs, diag / "prob_hist_all.png", "all pixels (incl. nodata)")
    _plot_hist(probs[1][valid_mask][np.newaxis, ...],  # tiny shim to reuse _plot_hist
               diag / "prob_hist_valid.png", "valid pixels only")
    _plot_prob_map(probs, diag / "prob_map.png", "raw sigmoid p(Building)")

    # ---- 4. re-vectorize with nodata masked at output time ----------------
    config = load_config(args.abd_config)
    palette, transparency = make_palette([c["color"] for c in config["classes"]])

    masked_root = diag / "masks_nodata_masked"
    if masked_root.exists():
        import shutil
        shutil.rmtree(masked_root)
    prob_b = probs[1].copy()
    prob_b[~valid_mask] = 0.0   # explicitly kill predictions on nodata pixels
    n_pos_masked = _threshold_and_vectorize(prob_b, tile, masked_root, palette, transparency)

    zerofilled_root = diag / "masks_zerofilled"
    if zerofilled_root.exists():
        import shutil
        shutil.rmtree(zerofilled_root)
    n_pos_zero = _threshold_and_vectorize(probs[1], tile, zerofilled_root, palette, transparency)

    print(f"\npositive-pixel counts after thresholding:")
    print(f"  nodata zero-filled (Phase 1 behaviour): {n_pos_zero}")
    print(f"  nodata masked at output:                 {n_pos_masked}")

    # ---- summary JSON ------------------------------------------------------
    summary = {
        "tile": {"x": tile.x, "y": tile.y, "z": tile.z},
        "ref_polys_total": int(len(ref_gdf)),
        "ref_centroids_in_tile": n_in_tile,
        "ref_centroids_in_chip": n_in_chip,
        "nodata": nod,
        "prob_stats_all_pixels": all_stats,
        "prob_stats_valid_pixels": valid_stats,
        "positive_pixels_zerofilled": n_pos_zero,
        "positive_pixels_nodata_masked": n_pos_masked,
    }
    (diag / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {diag / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
