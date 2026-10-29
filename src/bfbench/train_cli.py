"""Config -> training-run plumbing for `bfbench train`.

The CLI layer in `cli.py` just hands a YAML path here; this module
builds datasets, loads the baseline weight, and calls `train_one_run`.

Expected YAML shape (fields marked * are additions to the Phase 3
schema — all other fields match `bfbench prepare` / `eval`):

    train:                      # *
        source_weight: weights/neat-fullxview-epoch75.pth
        output_dir: runs/.../train
        batch_size: 4
        epochs: 30
        lr: 1.0e-4
        weight_decay: 1.0e-4
        amp: true
        patience: 5
        seed: 20260911
        pooled_aois: []         # names of AOIs for the pooled run; see below

    split:                      # * -- required
        block_size_m: 1000.0
        ratios: [0.70, 0.15, 0.15]
        seed: 20260911

Pooled runs: when `train.pooled_aois` is non-empty, the first config
is treated as the "parent" (its work_dir hosts the output + manifest);
the other work_dirs listed under `train.pooled_aois_work_dirs` have
their train tiles concatenated. Keeping this out-of-band from the
regular `aoi:` section means per-AOI configs stay editable without
mentioning the pooled run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import mercantile
import yaml

from bfbench.train.dataset import BuildingTileDataset
from bfbench.train.loop import (
    TrainConfig,
    build_albunet_from_checkpoint,
    seed_everything,
    train_one_run,
)
from bfbench.train.manifest import TrainingManifest, sha256_file

_LOG = logging.getLogger("bfbench.train_cli")


@dataclass(frozen=True)
class _TrainYAML:
    aoi_name: str
    work_dir: Path
    tile_size: int
    channel_name: str
    source_weight: Path
    output_dir: Path
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    amp: bool
    patience: int
    num_workers: int
    seed: int
    pooled_aois: list[str]
    pooled_aois_work_dirs: list[Path]
    split_seed: int
    split_block_size_m: float
    split_ratios: tuple[float, float, float]


def _load(cfg_path: Path) -> _TrainYAML:
    y = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if "train" not in y:
        raise KeyError(f"config {cfg_path} has no `train:` section")
    if "split" not in y:
        raise KeyError(f"config {cfg_path} has no `split:` section "
                      "(Phase 1 training requires a geographic block split)")
    tr = y["train"]
    sp = y["split"]
    return _TrainYAML(
        aoi_name=y["aoi"]["name"],
        work_dir=Path(y["run"]["work_dir"]),
        tile_size=int(y["adapter"]["tile_size"]),
        channel_name=str(y["adapter"].get("channel_name", "images")),
        source_weight=Path(tr["source_weight"]).expanduser(),
        output_dir=Path(tr["output_dir"]).expanduser(),
        batch_size=int(tr.get("batch_size", 4)),
        epochs=int(tr.get("epochs", 30)),
        lr=float(tr.get("lr", 1e-4)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        amp=bool(tr.get("amp", True)),
        patience=int(tr.get("patience", 5)),
        num_workers=int(tr.get("num_workers", 2)),
        seed=int(tr.get("seed", 20260911)),
        pooled_aois=list(tr.get("pooled_aois", []) or []),
        pooled_aois_work_dirs=[
            Path(p).expanduser() for p in (tr.get("pooled_aois_work_dirs") or [])
        ],
        split_seed=int(sp["seed"]),
        split_block_size_m=float(sp["block_size_m"]),
        split_ratios=tuple(float(r) for r in sp["ratios"]),  # type: ignore[arg-type]
    )


def _read_split(work_dir: Path) -> dict:
    p = work_dir / "block_split.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"block_split.json missing at {p}. Run `bfbench prepare "
            "<config.yaml>` first (with a `split:` section)."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _tiles_from(entries: list[list[int]]) -> list[mercantile.Tile]:
    return [mercantile.Tile(x, y, z) for z, x, y in entries]


def _ref_gdf(work_dir: Path) -> gpd.GeoDataFrame:
    p = work_dir / "reference.geojson"
    if not p.is_file():
        raise FileNotFoundError(f"reference.geojson missing at {p}")
    return gpd.read_file(p)


def train(cfg_path: Path) -> Path:
    """Entry point for `bfbench train <cfg.yaml>`. Returns the path to
    `training_manifest.json`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg_path = cfg_path.resolve()
    yml = _load(cfg_path)
    seed_everything(yml.seed)

    # Gather train / val tiles. For a pooled run the per-AOI train/val
    # lists are concatenated; the main AOI's val set is still the
    # per-AOI val (so early stopping behaviour is consistent across
    # the 4 per-AOI runs and the pooled one).
    all_train: list[mercantile.Tile] = []
    all_val:   list[mercantile.Tile] = []
    all_test:  list[mercantile.Tile] = []
    aoi_count = 0

    def _collect(work_dir: Path) -> None:
        nonlocal aoi_count
        sp = _read_split(work_dir)
        all_train.extend(_tiles_from(sp["train"]))
        all_val.extend(_tiles_from(sp["val"]))
        all_test.extend(_tiles_from(sp["test"]))
        aoi_count += 1

    _collect(yml.work_dir)
    for pooled_dir in yml.pooled_aois_work_dirs:
        _collect(pooled_dir)

    _LOG.info("tiles: train=%d val=%d test=%d (%d AOI(s))",
              len(all_train), len(all_val), len(all_test), aoi_count)

    # Build the datasets. The training dataset uses the parent AOI's
    # reference + tile tree; for pooled, the dataset is composed from
    # each AOI individually and concatenated. We model that as a
    # torch.utils.data.ConcatDataset so the loop is unchanged.
    from torch.utils.data import ConcatDataset  # noqa: PLC0415 - lazy torch import

    def _build_ds(work_dir: Path, tiles: list[mercantile.Tile]) -> BuildingTileDataset:
        return BuildingTileDataset(
            tiles=tiles,
            tiles_root=work_dir / "tiles",
            reference_gdf=_ref_gdf(work_dir),
            channel_name=yml.channel_name,
            tile_size=yml.tile_size,
            extension=".png",
        )

    train_dss = [_build_ds(yml.work_dir, _tiles_from(_read_split(yml.work_dir)["train"]))]
    val_dss   = [_build_ds(yml.work_dir, _tiles_from(_read_split(yml.work_dir)["val"]))]
    for pooled_dir in yml.pooled_aois_work_dirs:
        sp = _read_split(pooled_dir)
        train_dss.append(_build_ds(pooled_dir, _tiles_from(sp["train"])))
        val_dss.append(_build_ds(pooled_dir, _tiles_from(sp["val"])))
    train_ds = train_dss[0] if len(train_dss) == 1 else ConcatDataset(train_dss)
    val_ds   = val_dss[0]   if len(val_dss) == 1   else ConcatDataset(val_dss)

    # Load the baseline weight, build the model.
    if not yml.source_weight.is_file():
        raise FileNotFoundError(
            f"source weight not at {yml.source_weight}. Run "
            "`python scripts/fetch_weights.py` first."
        )
    model, source_meta = build_albunet_from_checkpoint(yml.source_weight,
                                                      device="cpu")
    source_sha = sha256_file(yml.source_weight)
    config_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

    manifest = TrainingManifest(
        config_source=str(cfg_path),
        config_sha256=config_sha,
        source_weight_path=str(yml.source_weight.resolve()),
        source_weight_sha256=source_sha,
        aoi_name=yml.aoi_name,
        pooled_aois=yml.pooled_aois,
        split_seed=yml.split_seed,
        split_assignment_hash=_read_split(yml.work_dir)["assignment_hash"],
        block_size_m=yml.split_block_size_m,
        split_ratios=yml.split_ratios,
        n_train_tiles=len(all_train),
        n_val_tiles=len(all_val),
        n_test_tiles=len(all_test),
        seed=yml.seed,
        batch_size=yml.batch_size,
        epochs=yml.epochs,
        lr=yml.lr,
        weight_decay=yml.weight_decay,
        amp=yml.amp,
        patience=yml.patience,
        loss="bce_dice",
        optimizer="AdamW",
        scheduler="cosine_warm_restarts",
    )

    tcfg = TrainConfig(
        seed=yml.seed,
        batch_size=yml.batch_size,
        epochs=yml.epochs,
        lr=yml.lr,
        weight_decay=yml.weight_decay,
        amp=yml.amp,
        patience=yml.patience,
        num_workers=yml.num_workers,
        device="cuda",   # falls back to cpu inside train_one_run if unavailable
    )

    result = train_one_run(
        cfg=tcfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        source_weight_path=yml.source_weight,
        output_dir=yml.output_dir,
        manifest=manifest,
        source_checkpoint_meta=source_meta,
    )
    return result.manifest_path


__all__ = ["train"]
