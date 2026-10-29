"""Integration test for the Phase 1 extension of `bfbench prepare`.

Covers:
    - When a config has NO `split:` section, prepare's block-split step
      is a no-op (back-compat with Phase 2 / Phase 3 configs).
    - When a config HAS a `split:` section, `_geo_split` emits a
      `block_split.json` whose train/val/test tile sets are pairwise
      disjoint and whose hash matches `split_tiles`.
    - The train_cli loader rejects configs without a split section.

Does NOT actually run `bfbench prepare` end-to-end (that needs S3
access + GDAL mosaicking). The step we're adding is isolated behind
`_geo_split(cfg, sampled_path)` so we can test it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bfbench.data.block_split import BlockSplitConfig, split_tiles
from bfbench.data.equal_area import DEFAULT_EQUAL_AREA_EPSG
from bfbench.prepare import PrepareConfig, _geo_split


def _make_prepare_cfg(work_dir: Path, with_split: bool) -> PrepareConfig:
    """Build a PrepareConfig with just enough fields set for _geo_split."""
    return PrepareConfig(
        aoi_name="AOI_UNIT_TEST",
        aoi_id=2,
        split="train",
        image_kind="PS-RGB",
        chip_limit=None,
        chip_ids_source=None,
        target_gsd_m=0.5,
        tile_size=512,
        zoom_tolerance=0.2,
        keep_borders=True,
        dedup_method="touching_union",
        dedup_buffer_m=0.5,
        dedup_iou_threshold=0.5,
        equal_area_epsg=DEFAULT_EQUAL_AREA_EPSG,
        n_tiles=50,
        seed=0,
        min_valid_fraction=1.0,
        split_block_size_m=(1000.0 if with_split else None),
        split_ratios=((0.70, 0.15, 0.15) if with_split else None),
        split_seed=(20260911 if with_split else None),
        work_dir=work_dir,
    )


def _write_fake_sampled(work_dir: Path, n: int = 400) -> Path:
    """20 x 20 Vegas-ish tile grid as `sampled_tiles.json`."""
    tiles = [[17, 24852 + dx, 50500 + dy]
             for dx in range(int(n ** 0.5)) for dy in range(int(n ** 0.5))]
    out = work_dir / "sampled_tiles.json"
    out.write_text(json.dumps({
        "aoi": "AOI_UNIT_TEST", "zoom": 17, "effective_gsd_m": 0.5,
        "seed": 0, "n_requested": len(tiles), "n_candidates": len(tiles),
        "tile_ids_hash": "x", "tiles": tiles,
    }, indent=2), encoding="utf-8")
    return out


def test_geo_split_noop_when_section_absent(tmp_path: Path) -> None:
    cfg = _make_prepare_cfg(tmp_path, with_split=False)
    _write_fake_sampled(tmp_path)
    out = _geo_split(cfg, tmp_path / "sampled_tiles.json")
    assert out is None
    assert not (tmp_path / "block_split.json").exists()


def test_geo_split_emits_disjoint_tile_sets(tmp_path: Path) -> None:
    cfg = _make_prepare_cfg(tmp_path, with_split=True)
    sampled = _write_fake_sampled(tmp_path, n=400)
    out = _geo_split(cfg, sampled)
    assert out is not None and out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))

    train_ids = {tuple(t) for t in data["train"]}
    val_ids   = {tuple(t) for t in data["val"]}
    test_ids  = {tuple(t) for t in data["test"]}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == 400

    # Each split should have at least a handful of tiles at this grid size
    # + default ratios + default seed. If one is empty, the split failed.
    assert data["n_train"] > 0
    assert data["n_val"] > 0
    assert data["n_test"] > 0

    # The assignment hash must match a direct call to split_tiles.
    import mercantile  # noqa: PLC0415
    tiles = [mercantile.Tile(x, y, z) for z, x, y in data["train"] + data["val"] + data["test"]]
    direct = split_tiles(tiles, BlockSplitConfig(
        block_size_m=cfg.split_block_size_m,  # type: ignore[arg-type]
        ratios=cfg.split_ratios,  # type: ignore[arg-type]
        seed=cfg.split_seed,  # type: ignore[arg-type]
        equal_area_epsg=cfg.equal_area_epsg,
    ))
    assert direct.assignment_hash == data["assignment_hash"]


def test_geo_split_also_writes_block_geojson(tmp_path: Path) -> None:
    cfg = _make_prepare_cfg(tmp_path, with_split=True)
    _write_fake_sampled(tmp_path, n=400)
    _geo_split(cfg, tmp_path / "sampled_tiles.json")
    assert (tmp_path / "block_split.geojson").is_file()


def test_geo_split_is_idempotent(tmp_path: Path) -> None:
    cfg = _make_prepare_cfg(tmp_path, with_split=True)
    _write_fake_sampled(tmp_path, n=400)
    out1 = _geo_split(cfg, tmp_path / "sampled_tiles.json")
    mtime1 = out1.stat().st_mtime  # type: ignore[union-attr]
    out2 = _geo_split(cfg, tmp_path / "sampled_tiles.json")
    assert out2 == out1
    assert out2.stat().st_mtime == mtime1  # cache hit, no rewrite


# ---------------------------------------------------------------------------
# train_cli._load guards
# ---------------------------------------------------------------------------

def test_train_cli_requires_split_section(tmp_path: Path) -> None:
    from bfbench.train_cli import _load  # noqa: PLC0415

    cfg_path = tmp_path / "x.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "aoi": {"name": "AOI_UNIT_TEST", "aoi_id": 2, "split": "train",
                "image_kind": "PS-RGB"},
        "adapter": {"tile_size": 512, "channel_name": "images",
                   "target_gsd_m": 0.5},
        "run": {"work_dir": str(tmp_path)},
        "train": {"source_weight": "weights/x.pth",
                 "output_dir": str(tmp_path / "train")},
        # NO split section
    }), encoding="utf-8")
    with pytest.raises(KeyError, match="split"):
        _load(cfg_path)


def test_train_cli_requires_train_section(tmp_path: Path) -> None:
    from bfbench.train_cli import _load  # noqa: PLC0415

    cfg_path = tmp_path / "x.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "aoi": {"name": "AOI_UNIT_TEST", "aoi_id": 2, "split": "train",
                "image_kind": "PS-RGB"},
        "adapter": {"tile_size": 512, "channel_name": "images",
                   "target_gsd_m": 0.5},
        "run": {"work_dir": str(tmp_path)},
        "split": {"block_size_m": 1000.0, "ratios": [0.7, 0.15, 0.15],
                 "seed": 0},
        # NO train section
    }), encoding="utf-8")
    with pytest.raises(KeyError, match="train"):
        _load(cfg_path)
