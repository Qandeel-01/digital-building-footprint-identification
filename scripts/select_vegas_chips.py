"""Select a compact contiguous cluster of SpaceNet AOI_2_Vegas chips.

We do NOT want to download all 3851 chips (~10 GB) just to sample 150
z=17 tiles. Strategy:

    1. List all chip keys in the SpaceNet bucket (cheap: one API call
       per 1000 keys via the paginator).
    2. For each chip, read ONLY the TIFF header via rasterio + /vsis3/
       (AWS_NO_SIGN_REQUEST). This transfers only the bytes GDAL needs
       to parse the geo-metadata: empirically << 100 kB per chip.
       Concurrent with ThreadPoolExecutor to bring wall-clock down.
    3. Reproject each chip's bounds to WGS84, keep centroids.
    4. Pick the N chips nearest to the AOI's centroid. Those are our
       contiguous cluster.

Writes a JSON listing the selected chip IDs, their bounds, and the
total header-probe wall-clock time, to
`runs/vegas_prepare/chip_selection.json`.

Byte accounting: /vsis3/ header probing is essentially free compared
to a full download (bytes transferred per probe is O(64 kB), vs.
2.54 MB per full chip). The JSON also records the download budget the
selected chip set implies, so the caller can decide.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from rasterio import open as rio_open
from rasterio.warp import transform_bounds

_LOG = logging.getLogger("bfbench.select_chips")

BUCKET = "spacenet-dataset"
PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/"

VEGAS_CENTER_LAT = 36.15   # approx AOI centroid; refined by probe density
VEGAS_CENTER_LON = -115.20


def _list_chip_keys() -> list[tuple[str, int]]:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, region_name="us-east-1"))
    out: list[tuple[str, int]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].lower().endswith(".tif"):
                out.append((obj["Key"], obj["Size"]))
    return out


def _chip_id(key: str) -> str:
    stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[-1]


def _probe_one(key: str, size: int) -> tuple[str, dict | None, str | None]:
    """Probe a single chip's header. Returns (chip_id, entry_or_None, err_or_None)."""
    cid = _chip_id(key)
    try:
        with rio_open(f"/vsis3/{BUCKET}/{key}") as src:
            w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return cid, {
            "key": key, "size_bytes": size,
            "west": w, "south": s, "east": e, "north": n,
            "cx": 0.5 * (w + e), "cy": 0.5 * (s + n),
        }, None
    except Exception as ex:  # noqa: BLE001
        return cid, None, f"{type(ex).__name__}: {ex}"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=250,
                   help="Number of chips to select (closest to the AOI centroid).")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("runs/vegas_prepare/chip_selection.json"))
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    _LOG.info("listing all Vegas PS-RGB chips ...")
    keys = _list_chip_keys()
    total_bucket_bytes = sum(s for _, s in keys)
    _LOG.info("found %d chip TIFFs; total on bucket = %.2f GB",
              len(keys), total_bucket_bytes / 1e9)

    _LOG.info("probing headers with %d workers ...", args.workers)
    t0 = time.time()
    probed: dict[str, dict] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_probe_one, k, s): k for k, s in keys}
        done = 0
        for fut in as_completed(futs):
            cid, entry, err = fut.result()
            done += 1
            if entry is not None:
                probed[cid] = entry
            else:
                errors[cid] = err or "unknown"
            if done % 500 == 0:
                _LOG.info("  probed %d/%d (%.1f/s)", done, len(keys), done / (time.time() - t0))
    elapsed = time.time() - t0
    _LOG.info("probed %d chips (%d errors) in %.1fs (%.1f/s)",
              len(probed), len(errors), elapsed, len(probed) / max(elapsed, 1e-9))

    def _dist2(entry: dict) -> float:
        dx = entry["cx"] - VEGAS_CENTER_LON
        dy = entry["cy"] - VEGAS_CENTER_LAT
        return dx * dx + dy * dy

    ordered = sorted(probed.items(), key=lambda kv: _dist2(kv[1]))
    chosen = [cid for cid, _ in ordered[: args.n]]
    total_bytes = sum(probed[c]["size_bytes"] for c in chosen)

    cxs = [probed[c]["cx"] for c in chosen]
    cys = [probed[c]["cy"] for c in chosen]
    bbox = {
        "west": min(probed[c]["west"] for c in chosen),
        "south": min(probed[c]["south"] for c in chosen),
        "east": max(probed[c]["east"] for c in chosen),
        "north": max(probed[c]["north"] for c in chosen),
        "centroid_lon": sum(cxs) / len(cxs),
        "centroid_lat": sum(cys) / len(cys),
    }

    payload = {
        "aoi": "AOI_2_Vegas",
        "n_available": len(probed),
        "n_selected": len(chosen),
        "n_errors": len(errors),
        "selection_center_lat": VEGAS_CENTER_LAT,
        "selection_center_lon": VEGAS_CENTER_LON,
        "cluster_bbox_wgs84": bbox,
        "chip_ids": chosen,
        "chips": {c: probed[c] for c in chosen},
        "estimated_download_bytes": total_bytes,
        "probe_wall_clock_seconds": elapsed,
        "errors": errors,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("wrote %s (%d chips, %.1f MB to download; probe took %.1fs)",
              args.out, len(chosen), total_bytes / 1e6, elapsed)


if __name__ == "__main__":
    main()
