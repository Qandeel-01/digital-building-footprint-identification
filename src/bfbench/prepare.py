"""bfbench prepare: fetch + mosaic + merge references + sample tiles
for one AOI, deterministically, resumably.

Outputs under the config's `run.work_dir`:

    <work_dir>/
        spacenet/PS-RGB/*.tif             cached chip imagery
        spacenet/geojson_buildings/*.geojson
        mosaic.vrt                        gdalbuildvrt over the chips
        reference.geojson                 merged + deduped in EPSG:4326
        sampled_tiles.json                seed + tile IDs + hash
        prepare_manifest.json             config source, versions, sha
        summary.json                      per-AOI descriptive stats

Every step is idempotent; re-running skips work that is already on disk.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mercantile
import yaml

from bfbench.data import (
    BlockSplitConfig,
    DedupRule,
    SpaceNetAOI,
    SpaceNetFetcher,
    blocks_to_geodataframe,
    build_vrt,
    merge_reference,
    sample_tiles,
    split_tiles,
    summarize_aoi,
    tile_is_fully_covered,
)
from bfbench.data.adapter import (
    pick_zoom_for_gsd,
    tile_bounds_webmercator,
    tiles_covering,
    webmercator_gsd_at,
)

_LOG = logging.getLogger("bfbench.prepare")


@dataclass(frozen=True)
class PrepareConfig:
    aoi_name: str
    aoi_id: int
    split: str
    image_kind: str
    chip_limit: int | None
    chip_ids_source: Path | None

    target_gsd_m: float
    tile_size: int
    zoom_tolerance: float
    keep_borders: bool

    dedup_method: str
    dedup_buffer_m: float
    dedup_iou_threshold: float
    equal_area_epsg: int

    n_tiles: int
    seed: int
    min_valid_fraction: float

    # Phase 1: optional block-based geographic split. If `split_section`
    # is None the behaviour of prepare is unchanged from Phase 2/3.
    split_block_size_m: float | None
    split_ratios: tuple[float, float, float] | None
    split_seed: int | None

    work_dir: Path


def _load(path: Path) -> PrepareConfig:
    with path.open("r", encoding="utf-8") as fp:
        y = yaml.safe_load(fp)
    aoi = y["aoi"]
    ad = y["adapter"]
    ref = y["reference"]["dedup"]
    smp = y["sampling"]
    run = y["run"]
    split_block = y.get("split", {}) or {}
    split_block_size_m = split_block.get("block_size_m")
    split_ratios_raw = split_block.get("ratios")
    split_ratios = (
        tuple(float(r) for r in split_ratios_raw)  # type: ignore[arg-type]
        if split_ratios_raw is not None else None
    )
    split_seed = split_block.get("seed")
    return PrepareConfig(
        aoi_name=aoi["name"],
        aoi_id=int(aoi["aoi_id"]),
        split=aoi["split"],
        image_kind=aoi["image_kind"],
        chip_limit=aoi.get("chip_limit"),
        chip_ids_source=(
            Path(aoi["chip_ids_source"]).expanduser()
            if aoi.get("chip_ids_source") else None
        ),
        target_gsd_m=float(ad["target_gsd_m"]),
        tile_size=int(ad["tile_size"]),
        zoom_tolerance=float(ad["zoom_tolerance"]),
        keep_borders=bool(ad["keep_borders"]),
        dedup_method=ref["method"],
        dedup_buffer_m=float(ref["buffer_m"]),
        dedup_iou_threshold=float(ref["iou_threshold"]),
        equal_area_epsg=int(ref["equal_area_epsg"]),
        n_tiles=int(smp["n_tiles"]),
        seed=int(smp["seed"]),
        min_valid_fraction=float(smp["min_valid_fraction"]),
        split_block_size_m=(float(split_block_size_m)
                            if split_block_size_m is not None else None),
        split_ratios=split_ratios,  # type: ignore[arg-type]
        split_seed=(int(split_seed) if split_seed is not None else None),
        work_dir=Path(run["work_dir"]),
    )


def _config_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(cfg: PrepareConfig) -> tuple[list[Path], list[Path], str]:
    """Download chips + geojson_buildings. Cache-aware."""
    fetcher = SpaceNetFetcher()
    aoi = SpaceNetAOI(aoi_id=cfg.aoi_id, split=cfg.split)
    bucket_uri = fetcher.verify_access(aoi)
    out_dir = cfg.work_dir / "spacenet"

    all_ids = fetcher.list_image_ids(aoi, image_kind=cfg.image_kind)
    if cfg.chip_ids_source is not None:
        selection = json.loads(cfg.chip_ids_source.read_text(encoding="utf-8"))
        wanted = [c for c in selection["chip_ids"] if c in set(all_ids)]
        missing = [c for c in selection["chip_ids"] if c not in set(all_ids)]
        if missing:
            _LOG.warning("chip_ids_source lists %d IDs not in bucket: %s",
                         len(missing), missing[:5])
        ids = wanted
        _LOG.info("aoi=%s available_chips=%d selected_from_source=%d file=%s",
                  cfg.aoi_name, len(all_ids), len(ids), cfg.chip_ids_source)
    elif cfg.chip_limit is not None:
        ids = all_ids[: cfg.chip_limit]
        _LOG.info("aoi=%s available_chips=%d fetching(head)=%d",
                  cfg.aoi_name, len(all_ids), len(ids))
    else:
        ids = all_ids
        _LOG.info("aoi=%s available_chips=%d fetching=all", cfg.aoi_name, len(all_ids))

    imgs, geos = [], []
    for iid in ids:
        img, geo = fetcher.download_pair(aoi, iid, out_dir,
                                         image_kind=cfg.image_kind, overwrite=False)
        imgs.append(img)
        geos.append(geo)
    return imgs, geos, bucket_uri


def _mosaic(cfg: PrepareConfig, imgs: list[Path]) -> Path:
    vrt = cfg.work_dir / "mosaic.vrt"
    if vrt.exists():
        _LOG.info("mosaic cache hit: %s", vrt)
        return vrt
    build_vrt(imgs, vrt)
    return vrt


def _reference(cfg: PrepareConfig, geos: list[Path]) -> Path:
    ref_out = cfg.work_dir / "reference.geojson"
    if ref_out.exists():
        _LOG.info("reference cache hit: %s", ref_out)
        return ref_out
    rule = DedupRule(
        method=cfg.dedup_method,       # type: ignore[arg-type]
        buffer_m=cfg.dedup_buffer_m,
        iou_threshold=cfg.dedup_iou_threshold,
        equal_area_epsg=cfg.equal_area_epsg,
    )
    res = merge_reference(geos, rule=rule, keep_crs="EPSG:4326")
    res.gdf.to_file(ref_out, driver="GeoJSON")
    (cfg.work_dir / "reference_dedup.json").write_text(
        json.dumps({
            "method": res.method,
            "n_input": res.n_input,
            "n_output": res.n_output,
            "n_duplicates_removed": res.n_duplicates_removed,
        }, indent=2),
        encoding="utf-8",
    )
    return ref_out


def _sample(cfg: PrepareConfig, vrt_path: Path) -> Path:
    out = cfg.work_dir / "sampled_tiles.json"
    if out.exists():
        _LOG.info("sampled_tiles cache hit: %s", out)
        return out

    import rasterio
    with rasterio.open(vrt_path) as src:
        from rasterio.warp import transform_bounds as _tb
        bounds_wgs = _tb(src.crs, "EPSG:4326", *src.bounds)
        native_crs = src.crs

    lat = 0.5 * (bounds_wgs[1] + bounds_wgs[3])
    zoom = pick_zoom_for_gsd(lat, cfg.target_gsd_m,
                             tolerance_m=cfg.zoom_tolerance,
                             tile_size=cfg.tile_size)
    tile_gsd_m = webmercator_gsd_at(lat, zoom, tile_size=cfg.tile_size)

    all_tiles = tiles_covering(bounds_wgs, zoom)
    _LOG.info("aoi=%s zoom=%d effective_gsd=%.3fm tiles_covering=%d",
              cfg.aoi_name, zoom, tile_gsd_m, len(all_tiles))

    # Filter to fully-covered tiles.
    from pyproj import Transformer
    to_native = Transformer.from_crs("EPSG:3857", native_crs, always_xy=True)

    covered = []
    for i, t in enumerate(all_tiles):
        w, s, e, n = tile_bounds_webmercator(t)
        # Reproject the tile bbox corners into native mosaic CRS.
        nw = to_native.transform(w, n)
        se = to_native.transform(e, s)
        bbox_native = (min(nw[0], se[0]), min(nw[1], se[1]),
                       max(nw[0], se[0]), max(nw[1], se[1]))
        if tile_is_fully_covered(vrt_path, bbox_native,
                                 min_valid_fraction=cfg.min_valid_fraction):
            covered.append(t)
        if (i + 1) % 500 == 0:
            _LOG.info("  scanned %d/%d covered=%d", i + 1, len(all_tiles), len(covered))
    _LOG.info("aoi=%s fully_covered_tiles=%d/%d", cfg.aoi_name,
              len(covered), len(all_tiles))

    result = sample_tiles(covered, cfg.n_tiles, cfg.seed)
    payload = {
        "aoi": cfg.aoi_name,
        "zoom": zoom,
        "effective_gsd_m": tile_gsd_m,
        "seed": result.seed,
        "n_requested": result.n_requested,
        "n_candidates": result.n_candidates,
        "tile_ids_hash": result.tile_ids_hash,
        "tiles": [[t.z, t.x, t.y] for t in result.tiles],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _summary(cfg: PrepareConfig, sampled_path: Path, ref_path: Path) -> Path:
    import geopandas as gpd
    out = cfg.work_dir / "summary.json"
    payload = json.loads(sampled_path.read_text(encoding="utf-8"))
    tiles = [mercantile.Tile(x, y, z) for z, x, y in payload["tiles"]]
    ref_gdf = gpd.read_file(ref_path)
    s = summarize_aoi(cfg.aoi_name, tiles, ref_gdf, cfg.equal_area_epsg)
    out.write_text(json.dumps(dataclasses.asdict(s), indent=2), encoding="utf-8")
    return out


def _manifest(cfg: PrepareConfig, cfg_path: Path, bucket_uri: str) -> None:
    m = {
        "config_source": str(cfg_path.resolve()),
        "config_sha256": _config_sha(cfg_path),
        "spacenet_bucket": bucket_uri,
        "aoi": cfg.aoi_name,
        "seed": cfg.seed,
    }
    (cfg.work_dir / "prepare_manifest.json").write_text(
        json.dumps(m, indent=2), encoding="utf-8",
    )


def _geo_split(cfg: PrepareConfig, sampled_path: Path) -> Path | None:
    """Phase 1: emit train / val / test tile IDs by geographic block.

    No-op if the config has no `split:` section. Writes
    `block_split.json` with per-split tile IDs + assignment hash + the
    full block GeoJSON next to it.
    """
    if (cfg.split_block_size_m is None
            or cfg.split_ratios is None
            or cfg.split_seed is None):
        _LOG.info("no split section in config; skipping block split")
        return None

    out = cfg.work_dir / "block_split.json"
    if out.exists():
        _LOG.info("block_split cache hit: %s", out)
        return out

    payload = json.loads(sampled_path.read_text(encoding="utf-8"))
    tiles = [mercantile.Tile(x, y, z) for z, x, y in payload["tiles"]]
    bs_cfg = BlockSplitConfig(
        block_size_m=cfg.split_block_size_m,
        ratios=cfg.split_ratios,  # type: ignore[arg-type]
        seed=cfg.split_seed,
        equal_area_epsg=cfg.equal_area_epsg,
    )
    res = split_tiles(tiles, bs_cfg)
    out.write_text(
        json.dumps({
            "aoi": cfg.aoi_name,
            "block_size_m": cfg.split_block_size_m,
            "ratios": list(cfg.split_ratios),
            "seed": cfg.split_seed,
            "equal_area_epsg": cfg.equal_area_epsg,
            "assignment_hash": res.assignment_hash,
            "n_train": len(res.train),
            "n_val":   len(res.val),
            "n_test":  len(res.test),
            "train": [[t.z, t.x, t.y] for t in res.train],
            "val":   [[t.z, t.x, t.y] for t in res.val],
            "test":  [[t.z, t.x, t.y] for t in res.test],
        }, indent=2),
        encoding="utf-8",
    )
    # Blocks GeoJSON for visualisation; small, written once.
    blocks_gdf = blocks_to_geodataframe(res.blocks, cfg.equal_area_epsg)
    blocks_gdf.to_file(cfg.work_dir / "block_split.geojson", driver="GeoJSON")
    _LOG.info("block split: train=%d val=%d test=%d hash=%s",
              len(res.train), len(res.val), len(res.test),
              res.assignment_hash[:12])
    return out


def prepare(cfg_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = _load(cfg_path)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    imgs, geos, bucket_uri = _fetch(cfg)
    vrt = _mosaic(cfg, imgs)
    ref = _reference(cfg, geos)
    sampled = _sample(cfg, vrt)
    summary = _summary(cfg, sampled, ref)
    _geo_split(cfg, sampled)
    _manifest(cfg, cfg_path, bucket_uri)
    _LOG.info("done. summary at %s", summary)
