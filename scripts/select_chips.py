"""Select a compact contiguous chip cluster for any SpaceNet 2 AOI.

Generalises scripts/select_vegas_chips.py:

    1. List every PS-RGB chip in the AOI on the SpaceNet bucket.
    2. Probe every chip's TIFF header via rasterio + /vsis3/
       (metadata only; << 100 kB per chip; ~20-25 probes/s with
       32 workers).
    3. Compute the geographic centroid of the probed set as the
       AOI's own centroid (avoids hand-maintaining a lat/lon per
       AOI).
    4. Rank chips by distance to that centroid, pick the top N.

Writes runs/<aoi_slug>_prepare/chip_selection.json in the schema
bfbench.prepare expects (chip_ids: list[str], chips: dict).

Usage:
    python scripts/select_chips.py --aoi-id 5 --n 800
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# GDAL/PROJ env bootstrap before rasterio import (Windows).
if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))

import boto3  # noqa: E402
from botocore import UNSIGNED  # noqa: E402
from botocore.client import Config  # noqa: E402
from rasterio import open as rio_open  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

_LOG = logging.getLogger("bfbench.select_chips")

BUCKET = "spacenet-dataset"
_AOI_CITY = {2: "Vegas", 3: "Paris", 4: "Shanghai", 5: "Khartoum"}


def _prefix_for(aoi_id: int, split: str = "train") -> str:
    city = _AOI_CITY[aoi_id]
    return f"spacenet/SN2_buildings/{split}/AOI_{aoi_id}_{city}/PS-RGB/"


def _list_chip_keys(prefix: str) -> list[tuple[str, int]]:
    s3 = boto3.client("s3",
                      config=Config(signature_version=UNSIGNED, region_name="us-east-1"))
    out: list[tuple[str, int]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].lower().endswith(".tif"):
                out.append((obj["Key"], obj["Size"]))
    return out


def _chip_id(key: str) -> str:
    stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[-1]


def _probe_one(key: str, size: int) -> tuple[str, dict | None, str | None]:
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


def _slug(aoi_id: int) -> str:
    return f"{_AOI_CITY[aoi_id].lower()}"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aoi-id", type=int, required=True, choices=[2, 3, 4, 5])
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    slug = _slug(args.aoi_id)
    if args.out is None:
        args.out = Path(f"runs/{slug}_prepare/chip_selection.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    prefix = _prefix_for(args.aoi_id, args.split)
    _LOG.info("listing chips under %s ...", prefix)
    keys = _list_chip_keys(prefix)
    total = sum(s for _, s in keys)
    _LOG.info("found %d chips; total on bucket = %.2f GB", len(keys), total / 1e9)
    if not keys:
        raise SystemExit(f"No chips found under {prefix}. Aborting.")

    _LOG.info("probing headers via /vsis3/ with %d workers ...", args.workers)
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
                _LOG.info("  probed %d/%d (%.1f/s)",
                          done, len(keys), done / max(time.time() - t0, 1e-9))
    elapsed = time.time() - t0
    _LOG.info("probed %d chips (%d errors) in %.1fs",
              len(probed), len(errors), elapsed)

    # AOI centroid = mean of probed chip centres (self-calibrating).
    cxs = [e["cx"] for e in probed.values()]
    cys = [e["cy"] for e in probed.values()]
    aoi_lat = sum(cys) / len(cys)
    aoi_lon = sum(cxs) / len(cxs)

    def _dist2(entry: dict) -> float:
        dx = entry["cx"] - aoi_lon
        dy = entry["cy"] - aoi_lat
        return dx * dx + dy * dy

    ordered = sorted(probed.items(), key=lambda kv: _dist2(kv[1]))
    chosen = [cid for cid, _ in ordered[: args.n]]
    total_bytes = sum(probed[c]["size_bytes"] for c in chosen)

    cluster_cxs = [probed[c]["cx"] for c in chosen]
    cluster_cys = [probed[c]["cy"] for c in chosen]
    bbox = {
        "west": min(probed[c]["west"] for c in chosen),
        "south": min(probed[c]["south"] for c in chosen),
        "east": max(probed[c]["east"] for c in chosen),
        "north": max(probed[c]["north"] for c in chosen),
        "centroid_lon": sum(cluster_cxs) / len(cluster_cxs),
        "centroid_lat": sum(cluster_cys) / len(cluster_cys),
    }
    payload = {
        "aoi_id": args.aoi_id,
        "aoi_city": _AOI_CITY[args.aoi_id],
        "aoi_slug": slug,
        "split": args.split,
        "n_available": len(probed),
        "n_selected": len(chosen),
        "n_errors": len(errors),
        "aoi_centroid_lat": aoi_lat,
        "aoi_centroid_lon": aoi_lon,
        "cluster_bbox_wgs84": bbox,
        "chip_ids": chosen,
        "chips": {c: probed[c] for c in chosen},
        "estimated_download_bytes": total_bytes,
        "probe_wall_clock_seconds": elapsed,
        "errors": errors,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _LOG.info("wrote %s (%d chips, %.1f MB to download)",
              args.out, len(chosen), total_bytes / 1e6)


if __name__ == "__main__":
    main()
