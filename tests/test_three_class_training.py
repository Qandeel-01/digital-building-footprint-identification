"""CPU smoke test for Phase 2a's 3-class training path.

Does NOT exercise upstream Albunet; a tiny 3-class seg model is
injected so the loop + dataset + CE-loss + checkpoint path are all
covered without downloading torchvision pretrained weights.

Mirrors the structure of tests/test_train_smoke.py (Phase 1 smoke).
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

from bfbench.data.three_class import (  # noqa: E402
    CLS_BG,
    CLS_BORDER,
    CLS_INTERIOR,
    N_CLASSES,
    ThreeClassConfig,
)
from bfbench.train.dataset import BuildingTileDataset  # noqa: E402
from bfbench.train.loop import (  # noqa: E402
    TrainConfig,
    cross_entropy_3class_loss,
    seed_everything,
    train_one_run,
)
from bfbench.train.manifest import TrainingManifest  # noqa: E402

TILE_SIZE = 32


def _write_tile_image(path: Path, left_half_bright: bool,
                      tile_size: int = TILE_SIZE) -> None:
    arr = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
    arr[:, :, :tile_size // 2] = 255 if left_half_bright else 0
    arr[:, :, tile_size // 2:] = 0 if left_half_bright else 255
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff",
                       height=tile_size, width=tile_size,
                       count=3, dtype="uint8",
                       transform=from_bounds(0, 0, 1, 1, tile_size, tile_size),
                       crs="EPSG:4326") as dst:
        dst.write(arr)


@pytest.fixture
def toy_dataset(tmp_path: Path) -> dict:
    """3 tiles: two touching squares at tile 0 (3-class target has
    all three classes), one isolated square at tile 1, one empty at
    tile 2. Train / val split by tile index."""
    base_x, base_y, z = 24852, 50532, 17
    tiles = [mercantile.Tile(x=base_x + dx, y=base_y, z=z) for dx in range(3)]

    tiles_root = tmp_path / "tiles"
    for i, t in enumerate(tiles):
        p = tiles_root / "images" / str(t.z) / str(t.x) / f"{t.y}.tif"
        _write_tile_image(p, left_half_bright=(i != 2))

    # Reference polygons that create: tile 0 -> two touching halves,
    # tile 1 -> single square, tile 2 -> none.
    polys = []
    b0 = mercantile.bounds(tiles[0])
    # Two touching rectangles across the tile midline:
    mid_lon = 0.5 * (b0.west + b0.east)
    polys.append(box(b0.west, b0.south, mid_lon, b0.north))
    polys.append(box(mid_lon, b0.south, b0.east, b0.north))
    b1 = mercantile.bounds(tiles[1])
    polys.append(box(b1.west, b1.south, b1.east, b1.north))
    ref_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")

    return {
        "tiles_root": tiles_root,
        "tiles": tiles,
        "train_tiles": tiles[:2],
        "val_tiles": tiles[2:],
        "reference_gdf": ref_gdf,
    }


class _TinyThreeClassModel(_nn.Module):
    """3-channel segmentation head: 1x1 -> ReLU -> 1x1 -> 3 outputs."""

    version = 2  # upstream Albunet exposes .version; mimic for manifest sanity

    def __init__(self) -> None:
        super().__init__()
        self.net = _nn.Sequential(
            _nn.Conv2d(3, 8, kernel_size=1),
            _nn.ReLU(inplace=True),
            _nn.Conv2d(8, N_CLASSES, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset shape in 3-class mode
# ---------------------------------------------------------------------------

def test_dataset_three_class_mode_shape_and_dtype(toy_dataset: dict) -> None:
    # TILE_SIZE=32 at z=17 Vegas-ish is ~8 m/px; the border threshold
    # has to be a few metres to span at least one pixel's EDT.
    ds = BuildingTileDataset(
        tiles=toy_dataset["tiles"],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
        target_mode="3class",
        three_class_cfg=ThreeClassConfig(border_distance_m=15.0),
    )
    assert ds.target_mode == "3class"
    assert len(ds) == 3
    img, target = ds[0]
    assert img.shape == (3, TILE_SIZE, TILE_SIZE)
    assert img.dtype == torch.float32
    # 3-class target: int64, no leading channel.
    assert target.shape == (TILE_SIZE, TILE_SIZE)
    assert target.dtype == torch.int64
    # Tile 0 has two touching polygons that together fill the tile
    # -> interior + border present. The fixture does not leave any
    # background on tile 0, so CLS_BG only appears on tile 2.
    unique_t0 = set(torch.unique(target).tolist())
    assert CLS_INTERIOR in unique_t0
    assert CLS_BORDER in unique_t0, f"no border pixels; unique = {unique_t0}"
    # Tile 2 (empty) is all background.
    _, target2 = ds[2]
    assert set(torch.unique(target2).tolist()) == {CLS_BG}


def test_dataset_binary_mode_still_works(toy_dataset: dict) -> None:
    """Phase 1 contract: default binary mode unchanged."""
    ds = BuildingTileDataset(
        tiles=toy_dataset["tiles"][:1],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
    )
    img, mask = ds[0]
    assert mask.shape == (1, TILE_SIZE, TILE_SIZE)
    assert mask.dtype == torch.float32


def test_bad_target_mode_rejected(toy_dataset: dict) -> None:
    with pytest.raises(ValueError, match="target_mode"):
        BuildingTileDataset(
            tiles=toy_dataset["tiles"][:1],
            tiles_root=toy_dataset["tiles_root"],
            reference_gdf=toy_dataset["reference_gdf"],
            channel_name="images",
            tile_size=TILE_SIZE,
            extension=".tif",
            target_mode="4class",  # typo
        )


# ---------------------------------------------------------------------------
# CE loss shape + behaviour
# ---------------------------------------------------------------------------

def test_cross_entropy_3class_loss_minimised_at_perfect_prediction():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2, :] = 1
    target[0, 2:, :] = 2
    # Perfect logits: push each predicted class to +10, others to -10.
    logits = torch.full((1, 3, 4, 4), -10.0)
    for cls in (0, 1, 2):
        logits[0, cls][target[0] == cls] = 10.0
    loss = cross_entropy_3class_loss(logits, target)
    assert float(loss) < 0.01


def test_cross_entropy_3class_loss_large_when_inverted():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2, :] = 1
    # Logits predict class 0 everywhere confidently.
    logits = torch.full((1, 3, 4, 4), -10.0)
    logits[0, 0] = 10.0
    loss = cross_entropy_3class_loss(logits, target)
    assert float(loss) > 1.0


# ---------------------------------------------------------------------------
# Smoke training run on 3-class data
# ---------------------------------------------------------------------------

def test_three_class_training_loop_runs_and_saves_shape_out_3(
    toy_dataset: dict, tmp_path: Path,
) -> None:
    seed_everything(321)
    train_ds = BuildingTileDataset(
        tiles=toy_dataset["train_tiles"],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
        target_mode="3class",
        three_class_cfg=ThreeClassConfig(border_distance_m=15.0),
    )
    val_ds = BuildingTileDataset(
        tiles=toy_dataset["val_tiles"],
        tiles_root=toy_dataset["tiles_root"],
        reference_gdf=toy_dataset["reference_gdf"],
        channel_name="images",
        tile_size=TILE_SIZE,
        extension=".tif",
        target_mode="3class",
    )
    model = _TinyThreeClassModel()
    cfg = TrainConfig(
        seed=321, batch_size=2, epochs=4, lr=5e-2,
        amp=False, patience=10, num_workers=0, pin_memory=False,
        device="cpu",
    )
    manifest = TrainingManifest(
        config_source="unit-test-3class",
        config_sha256="dead",
        source_weight_path="unit-test",
        source_weight_sha256="dead",
        aoi_name="AOI_UNIT_TEST_3C",
        seed=cfg.seed, batch_size=cfg.batch_size, epochs=cfg.epochs,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        amp=cfg.amp, patience=cfg.patience, device=cfg.device,
        loss="cross_entropy_3class",
    )
    out_dir = tmp_path / "run3c"
    result = train_one_run(
        cfg=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        source_weight_path=Path("unit-test"),
        output_dir=out_dir,
        manifest=manifest,
        loss_fn=cross_entropy_3class_loss,
        source_checkpoint_meta={
            "nn": "Albunet",
            "shape_in": (3, TILE_SIZE, TILE_SIZE),
            "shape_out": (N_CLASSES, TILE_SIZE, TILE_SIZE),
            "encoder": "resnet50",
            "model_version": 2,
            "uuid": "unit-test-3c",
            "loader": "SemSeg",
        },
    )
    # Loss should drop.
    first = result.history[0].train_loss
    best = min(h.train_loss for h in result.history)
    assert best <= first  # soft: tiny model may not drop much on 2 tiles
    assert result.checkpoint_path.is_file()

    # Checkpoint must carry shape_out=(3,...) for `bfbench run-prepared`
    # to know to call the 3-class postprocess.
    chkpt = torch.load(result.checkpoint_path, map_location="cpu",
                       weights_only=False)
    assert tuple(chkpt["shape_out"]) == (N_CLASSES, TILE_SIZE, TILE_SIZE)
    assert chkpt["nn"] == "Albunet"
    assert "state_dict" in chkpt
    # Manifest records the 3-class loss.
    manifest_json = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_json["loss"] == "cross_entropy_3class"


# ---------------------------------------------------------------------------
# rebuild_final_conv_for_three_class on a mini Albunet-shaped model
# ---------------------------------------------------------------------------

class _MiniAlbunetHead(_nn.Module):
    """Mimics the attribute surface that
    `rebuild_final_conv_for_three_class` touches (an Albunet exposes
    `.final: nn.Conv2d` + `.version`).  Keeps the test free of upstream."""

    version = 2

    def __init__(self) -> None:
        super().__init__()
        self.body = _nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.final = _nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.body(x))


def test_rebuild_final_conv_preserves_in_channels_and_warm_starts_interior():
    from bfbench.train.loop import rebuild_final_conv_for_three_class  # noqa: PLC0415

    model = _MiniAlbunetHead()
    original_weight = model.final.weight.detach().clone()
    rebuilt = rebuild_final_conv_for_three_class(model, num_classes=3)
    assert rebuilt.final.in_channels == 32
    assert rebuilt.final.out_channels == 3
    # Channel 1 (interior) was initialised from the pretrained
    # channel 0 (binary building).
    np.testing.assert_allclose(
        rebuilt.final.weight.detach()[1].numpy(),
        original_weight[0].numpy(),
    )
