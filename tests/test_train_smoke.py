"""CPU smoke test for the Phase 1 training loop.

Does NOT exercise upstream Albunet — ResNet-50 on CPU would take
minutes per epoch. Instead, injects a tiny two-conv segmentation model
with the same input/output shape so the loop, dataset, AMP fallback,
manifest, and checksum paths are all covered.

The GPU + real-Albunet integration lives in a `@pytest.mark.gpu` test
below so CI can still run (CI deselects `gpu` and `network`).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

torch = pytest.importorskip("torch")
from torch import nn as _nn  # noqa: E402

from bfbench.train.dataset import BuildingTileDataset  # noqa: E402
from bfbench.train.loop import (  # noqa: E402
    TrainConfig,
    bce_dice_loss,
    seed_everything,
    train_one_run,
)
from bfbench.train.manifest import TrainingManifest, sha256_file  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: a 3-tile toy "AOI" with synthetic imagery + a reference
# polygon covering exactly one tile. The task is trivially learnable
# so a 2-conv model should drop its loss over a few epochs.
# ---------------------------------------------------------------------------

TILE_SIZE = 32  # tiny for CPU speed


def _write_tile_image(path: Path, pattern: str, tile_size: int = TILE_SIZE) -> None:
    """Write a 3-band uint8 image. pattern:
        "white"   -> all 255
        "black"   -> all 0
        "lhs"     -> left half 255, right half 0 (sanity visual pattern)
    """
    arr = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
    if pattern == "white":
        arr[:] = 255
    elif pattern == "lhs":
        arr[:, :, : tile_size // 2] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    # Give the file a dummy affine so rasterio can open it.
    transform = from_bounds(0, 0, 1, 1, tile_size, tile_size)
    with rasterio.open(path, "w", driver="GTiff",
                       height=tile_size, width=tile_size,
                       count=3, dtype="uint8", transform=transform,
                       crs="EPSG:4326") as dst:
        dst.write(arr)


@pytest.fixture
def toy_dataset(tmp_path: Path) -> dict:
    """Build:
        - 4 tiles total at a plausible SpaceNet-ish z=17 (Vegas-ish)
        - 2 of them get a reference polygon covering the whole tile
          (so the mask is 1 everywhere), 2 get no reference (mask = 0)
        - imagery is distinguishable per class so a 2-conv model can
          actually learn
    Returns a dict with tiles_root, train/val tile lists, and the
    reference_gdf.
    """
    # Four nearby tiles in Vegas.
    base_x, base_y, z = 24852, 50532, 17
    tiles = [mercantile.Tile(x=base_x + dx, y=base_y, z=z) for dx in range(4)]

    tiles_root = tmp_path / "tiles"
    for i, t in enumerate(tiles):
        p = tiles_root / "images" / str(t.z) / str(t.x) / f"{t.y}.tif"
        # Tiles 0 and 1 are "building"; tiles 2 and 3 are "no-building".
        _write_tile_image(p, "white" if i < 2 else "black")

    # Reference polygons: covers tiles 0 and 1 entirely (so mask=1),
    # nothing for 2 and 3.
    ref_polys = []
    for i in (0, 1):
        b = mercantile.bounds(tiles[i])
        ref_polys.append(box(b.west, b.south, b.east, b.north))
    ref_gdf = gpd.GeoDataFrame(geometry=ref_polys, crs="EPSG:4326")

    return {
        "tiles_root": tiles_root,
        "tiles": tiles,
        "train_tiles": tiles[:2] + tiles[2:3],  # 3 train (1 building + 1 non-building, +1 non-building)
        "val_tiles": tiles[3:],
        "reference_gdf": ref_gdf,
    }


class _TinySegModel(_nn.Module):
    """1x1 conv -> ReLU -> 1x1 conv. Enough capacity to learn
    'bright pixel -> 1, dark pixel -> 0' on the toy dataset in <= 5
    epochs; small enough to train in <1 s on CPU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = _nn.Sequential(
            _nn.Conv2d(3, 8, kernel_size=1),
            _nn.ReLU(inplace=True),
            _nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_seed_everything_is_reproducible():
    seed_everything(42)
    a = torch.randn(5).tolist()
    seed_everything(42)
    b = torch.randn(5).tolist()
    assert a == b


# ---------------------------------------------------------------------------
# Loss shape + behaviour
# ---------------------------------------------------------------------------

def test_bce_dice_loss_minimised_when_logits_match_target():
    """When logits are very positive wherever target == 1 and very
    negative where target == 0, BCE+Dice should be near zero."""
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1
    logits = (target - 0.5) * 10  # +5 for ones, -5 for zeros
    loss = bce_dice_loss(logits, target)
    assert float(loss) < 0.05


def test_bce_dice_loss_large_when_predictions_inverted():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4, :] = 1
    logits = (0.5 - target) * 10  # inverted
    loss = bce_dice_loss(logits, target)
    assert float(loss) > 0.5


# ---------------------------------------------------------------------------
# Dataset wiring (shape, dtype, mask correctness)
# ---------------------------------------------------------------------------

def test_dataset_shape_and_dtype(toy_dataset: dict) -> None:
    ds = BuildingTileDataset(
        tiles=toy_dataset["tiles"],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
    )
    assert len(ds) == 4
    img, mask = ds[0]
    assert img.shape == (3, TILE_SIZE, TILE_SIZE)
    assert mask.shape == (1, TILE_SIZE, TILE_SIZE)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32
    # Tile 0 has a reference polygon -> non-zero mask
    assert float(mask.sum()) > 0.0


def test_dataset_mask_zero_when_no_reference(toy_dataset: dict) -> None:
    ds = BuildingTileDataset(
        tiles=toy_dataset["tiles"],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
    )
    _, mask = ds[2]  # a no-building tile
    assert float(mask.sum()) == 0.0


# ---------------------------------------------------------------------------
# Smoke training run: loss drops, checkpoint + sha + manifest are written
# ---------------------------------------------------------------------------

def test_training_loop_reduces_loss_and_writes_manifest(toy_dataset: dict,
                                                       tmp_path: Path) -> None:
    seed_everything(123)
    train_ds = BuildingTileDataset(
        tiles=toy_dataset["train_tiles"],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
    )
    val_ds = BuildingTileDataset(
        tiles=toy_dataset["val_tiles"],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
    )
    model = _TinySegModel()
    cfg = TrainConfig(
        seed=123, batch_size=2, epochs=6, lr=5e-2,
        amp=False, patience=10, num_workers=0, pin_memory=False,
        device="cpu",
    )
    manifest = TrainingManifest(
        config_source="unit-test",
        config_sha256="deadbeef",
        source_weight_path="unit-test",
        source_weight_sha256="deadbeef",
        aoi_name="AOI_UNIT_TEST",
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        amp=cfg.amp,
        patience=cfg.patience,
        device=cfg.device,
    )
    out_dir = tmp_path / "run"
    result = train_one_run(
        cfg=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        source_weight_path=Path("unit-test"),
        output_dir=out_dir,
        manifest=manifest,
        source_checkpoint_meta={  # upstream-compatible payload shell
            "nn": "Albunet",
            "shape_in": (3, TILE_SIZE, TILE_SIZE),
            "shape_out": (1, TILE_SIZE, TILE_SIZE),
            "encoder": "resnet50",
            "model_version": 2,
            "uuid": "unit-test-uuid",
            "loader": "SemSeg",
        },
    )
    # Loss should have gone down.
    assert len(result.history) >= 2
    first = result.history[0].train_loss
    best = min(h.train_loss for h in result.history)
    assert best < first - 0.05, (
        f"training loss did not drop meaningfully: first={first:.4f} "
        f"best={best:.4f}"
    )

    # Checkpoint + sha + manifest exist.
    assert result.checkpoint_path.is_file()
    sha_file = result.checkpoint_path.with_suffix(result.checkpoint_path.suffix + ".sha256.json")
    assert sha_file.is_file()
    sha = json.loads(sha_file.read_text(encoding="utf-8"))["sha256"]
    assert sha == sha256_file(result.checkpoint_path)

    manifest_json = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_json["output_sha256"] == sha
    assert manifest_json["best_epoch"] >= 1
    assert manifest_json["best_val_loss"] < float("inf")

    # The saved checkpoint should carry the upstream-shaped payload so
    # run-prepared can consume it.
    chkpt = torch.load(result.checkpoint_path, map_location="cpu",
                       weights_only=False)
    for key in ("nn", "shape_in", "shape_out", "encoder", "state_dict",
                "model_version", "loader"):
        assert key in chkpt, f"missing key {key!r} in saved checkpoint"


# ---------------------------------------------------------------------------
# GPU / Albunet integration (not run in CI)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_build_albunet_from_checkpoint_loads_pretrained(tmp_path: Path) -> None:
    """Smoke-only: build the upstream Albunet from the fetched baseline
    and verify shape_in/shape_out keys are preserved. Requires upstream
    installed + the baseline checkpoint on disk."""
    ckpt = Path("weights/neat-fullxview-epoch75.pth")
    if not ckpt.is_file():
        pytest.skip(f"baseline checkpoint not at {ckpt}")
    from bfbench.train.loop import build_albunet_from_checkpoint  # noqa: PLC0415

    model, meta = build_albunet_from_checkpoint(ckpt, device="cpu")
    assert meta["nn"] == "Albunet"
    assert tuple(meta["shape_in"]) == (3, 512, 512)
    assert hasattr(model, "forward")
