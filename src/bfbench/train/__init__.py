"""Phase 1 fine-tuning package.

Entry point is `bfbench.train.loop.train_one_run`. See
`docs/phase1-decisions.md` for the overall rationale and hyperparams.
"""

from bfbench.train.dataset import (
    BuildingTileDataset,
    rasterise_reference_to_tile,
)
from bfbench.train.loop import (
    EpochLog,
    TrainConfig,
    TrainResult,
    bce_dice_loss,
    build_albunet_from_checkpoint,
    seed_everything,
    train_one_run,
)
from bfbench.train.manifest import (
    TrainingManifest,
    sha256_file,
    write_training_manifest,
)

__all__ = [
    "BuildingTileDataset",
    "EpochLog",
    "TrainConfig",
    "TrainResult",
    "TrainingManifest",
    "bce_dice_loss",
    "build_albunet_from_checkpoint",
    "rasterise_reference_to_tile",
    "seed_everything",
    "sha256_file",
    "train_one_run",
    "write_training_manifest",
]
