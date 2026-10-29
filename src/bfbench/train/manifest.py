"""Training-run manifest + checkpoint checksumming.

Every fine-tuned checkpoint produced in Phase 1 ships with:

    <output_dir>/best.pth              -- the state_dict + metadata
    <output_dir>/best.sha256.json      -- {"sha256": "...", "path": "best.pth"}
    <output_dir>/training_manifest.json

The manifest records config source, split hash, source-weight sha256,
seed, hyperparams, torch/CUDA versions and per-epoch loss curve. This
is what the brief's "config + weight checksum + seed recorded in the
run manifest" rule points at.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file. Same chunk size as scripts/fetch_weights.py."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class EpochRecord:
    """One row of the per-epoch training log stored in the manifest."""

    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    seconds: float


@dataclass
class TrainingManifest:
    """Everything you need to reproduce or re-identify a training run.

    `output_sha256` is populated AFTER `best.pth` is written; keeping
    it on the dataclass means the manifest and the checksum are
    serialised atomically.
    """

    # Provenance
    config_source: str                   # path on disk to the YAML
    config_sha256: str                   # sha256 of the YAML bytes
    source_weight_path: str              # path to the baseline .pth we fine-tuned from
    source_weight_sha256: str            # sha256 of that baseline .pth
    aoi_name: str                        # "AOI_2_Vegas" or "pooled"
    pooled_aois: list[str] = field(default_factory=list)

    # Split
    split_seed: int = 0
    split_assignment_hash: str = ""      # sha256 from SplitResult
    block_size_m: float = 1000.0
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    n_train_tiles: int = 0
    n_val_tiles: int = 0
    n_test_tiles: int = 0

    # Hyperparameters
    seed: int = 0
    batch_size: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    patience: int = 5
    loss: str = "bce_dice"
    optimizer: str = "AdamW"
    scheduler: str = "cosine_warm_restarts"

    # Runtime
    device: str = "cpu"
    torch_version: str = ""
    cuda_version: str = ""
    python_version: str = sys.version
    platform: str = platform.platform()

    # Results
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    epochs_log: list[EpochRecord] = field(default_factory=list)

    # Outputs (filled in after writing the .pth)
    output_dir: str = ""
    output_checkpoint: str = ""          # relative name under output_dir, e.g. "best.pth"
    output_sha256: str = ""              # sha256 of the above

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Make ratios JSON-friendly (tuple -> list).
        d["split_ratios"] = list(self.split_ratios)
        return d


def write_training_manifest(output_dir: Path, manifest: TrainingManifest) -> Path:
    """Write both <output_dir>/training_manifest.json and the sibling
    <output_dir>/<output_checkpoint>.sha256.json. Returns the manifest
    path.

    Preconditions: `manifest.output_checkpoint` names an existing file
    in `output_dir` and `manifest.output_sha256` matches that file's
    sha256. Both are asserted so a stale manifest cannot ship.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / manifest.output_checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"manifest references checkpoint {ckpt_path} but it does not exist"
        )
    observed = sha256_file(ckpt_path)
    if manifest.output_sha256 and observed != manifest.output_sha256:
        raise ValueError(
            f"manifest sha256 {manifest.output_sha256} does not match actual "
            f"file sha256 {observed} at {ckpt_path}"
        )
    manifest.output_sha256 = observed

    # Sibling .sha256.json so downstream run_prepared can refuse to use
    # a swapped file (mirrors scripts/fetch_weights.py).
    (ckpt_path.with_suffix(ckpt_path.suffix + ".sha256.json")).write_text(
        json.dumps({"sha256": observed, "path": ckpt_path.name}, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "EpochRecord",
    "TrainingManifest",
    "sha256_bytes",
    "sha256_file",
    "write_training_manifest",
]
