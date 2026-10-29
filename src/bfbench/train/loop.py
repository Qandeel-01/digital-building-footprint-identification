"""Fine-tuning loop for the Phase 1 domain-adaptation experiment.

Design points — see docs/phase1-decisions.md for rationale:

* Reuses upstream `abd_model.nn.albunet.Albunet` unchanged. The
  pretrained checkpoint is loaded via its `state_dict` key, matching
  how upstream `predict.py` consumes the same file. The saved fine-
  tuned checkpoint preserves every key the predict pipeline expects
  (`nn`, `encoder`, `shape_in`, `shape_out`, `state_dict`,
  `model_version`, `uuid`, `loader`), so `bfbench run-prepared` can
  consume the fine-tuned weight without any code change.
* Mixed precision via `torch.amp.autocast` + `torch.amp.GradScaler`.
  Opt-out via `TrainConfig.amp=False`.
* AdamW + cosine warm restarts. Loss = BCE + Dice (standard binary
  seg baseline).
* Early stopping on validation loss with a configurable patience.
* `torch`, `numpy`, `random` all seeded from one `TrainConfig.seed`.
* The loop is "callable" from a notebook: `train_one_run(cfg,
  train_ds, val_ds, source_weight_path, output_dir)` returns a
  `TrainResult` dataclass with the final manifest + checkpoint path.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bfbench.train.manifest import (
    EpochRecord,
    TrainingManifest,
    sha256_file,
    write_training_manifest,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

_LOG = logging.getLogger("bfbench.train")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one training run.

    Defaults are the Phase 1 defaults from `docs/phase1-decisions.md`.
    All fields overridable from the YAML.
    """

    seed: int = 20260911
    batch_size: int = 4
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    patience: int = 5
    num_workers: int = 2
    pin_memory: bool = True
    scheduler_t0: int = 5     # cosine warm restart period
    scheduler_t_mult: int = 1
    device: str = "cuda"      # overridden to "cpu" when torch.cuda.is_available() == False


@dataclass
class EpochLog:
    """Mutable per-epoch log; converted to EpochRecord for the manifest."""

    epoch: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    lr: float = 0.0
    seconds: float = 0.0


@dataclass
class TrainResult:
    """What `train_one_run` returns."""

    output_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    manifest: TrainingManifest
    history: list[EpochLog] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python / numpy / torch for deterministic-ish training.

    Fully deterministic cuDNN requires `torch.use_deterministic_algorithms
    (True)` and matching env vars; we don't force that because it also
    disables common AMP code paths. The seed we set is still a strong
    reproducibility guarantee for data iteration order and weight init.
    """
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bce_dice_loss(logits: Any, target: Any,
                  bce_weight: float = 0.5,
                  dice_weight: float = 0.5,
                  smooth: float = 1.0) -> Any:
    """BCE + soft Dice.

    `logits` is (B, 1, H, W) raw model output (NO sigmoid applied yet);
    we use `binary_cross_entropy_with_logits` which is numerically more
    stable than sigmoid + BCE. The Dice term uses `torch.sigmoid` once.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2 * inter + smooth) / (denom + smooth)).mean()
    return bce_weight * bce + dice_weight * dice


# ---------------------------------------------------------------------------
# Upstream Albunet from a pre-trained checkpoint
# ---------------------------------------------------------------------------

def build_albunet_from_checkpoint(checkpoint_path: Path,
                                 device: str = "cpu") -> tuple[Any, dict]:
    """Load the upstream AlbuNet with the pretrained state_dict.

    Returns (model, checkpoint_metadata). `checkpoint_metadata` contains
    at least `nn`, `encoder`, `shape_in`, `shape_out`, `model_version`,
    `uuid`, `loader` — the keys the predict pipeline reads.

    Imports upstream lazily so the module can be imported in env.yml-
    less test environments.
    """
    from abd_model.nn.albunet import Albunet  # noqa: PLC0415 - lazy import

    # weights_only=False: the upstream checkpoint carries a uuid.UUID and
    # a couple of other non-tensor fields that the torch 2.6+ "safe"
    # unpickler refuses. The baseline checkpoint is pinned by sha256
    # (scripts/fetch_weights.py); we trust it.
    chkpt = torch.load(str(checkpoint_path), map_location="cpu",
                      weights_only=False)
    for required in ("nn", "shape_in", "shape_out", "encoder", "state_dict"):
        if required not in chkpt:
            raise KeyError(
                f"baseline checkpoint {checkpoint_path} missing key "
                f"{required!r}; expected an upstream abd_model format"
            )
    model = Albunet(chkpt["shape_in"], chkpt["shape_out"],
                   chkpt["encoder"].lower())
    # Upstream saves the state_dict as nested under DistributedDataParallel
    # when trained multi-GPU. Strip the "module." prefix if present.
    sd = chkpt["state_dict"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device)
    return model, chkpt


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------

def _one_epoch(model: Any, loader: Any, optimiser: Any,
               scaler: Any, device: str, amp: bool,
               loss_fn: Callable[..., Any],
               train: bool = True) -> float:
    """Mean loss over the loader. If `train=False` runs in eval mode."""
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if train:
                optimiser.zero_grad(set_to_none=True)
            if amp and device.startswith("cuda"):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
            else:
                logits = model(images)
                loss = loss_fn(logits, masks)
                if train:
                    loss.backward()
                    optimiser.step()
            total += float(loss.item())
            n_batches += 1
    return total / max(1, n_batches)


def train_one_run(cfg: TrainConfig,
                  train_dataset: Any,
                  val_dataset: Any,
                  *,
                  model: Any,
                  source_weight_path: Path,
                  output_dir: Path,
                  manifest: TrainingManifest,
                  loss_fn: Callable[..., Any] | None = None,
                  save_checkpoint_fn: Callable[[Any, Path, dict], None] | None = None,
                  source_checkpoint_meta: dict | None = None,
                  ) -> TrainResult:
    """Run one fine-tuning job to completion.

    `model` is the network to fine-tune, already loaded with the
    baseline state_dict and moved to `cfg.device`. The caller is
    responsible for that so unit tests can inject a tiny toy model.

    `save_checkpoint_fn(model, path, meta)` is called with
    `source_checkpoint_meta` extended to persist the fine-tuned
    state_dict in an upstream-compatible format. If None, a default
    saver is used that mirrors upstream's `torch.save({..., 'state_dict':
    model.state_dict(), ...}, path)` layout.
    """
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("train_one_run requires torch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = loss_fn or bce_dice_loss
    source_checkpoint_meta = source_checkpoint_meta or {}

    # Resolve device at run time: honour user intent but fall back to
    # cpu if cuda is not actually available.
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        _LOG.warning("cfg.device=%r but torch.cuda.is_available()=False; "
                     "falling back to cpu", device)
        device = "cpu"
    model = model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=max(1, cfg.scheduler_t0), T_mult=cfg.scheduler_t_mult,
    )
    scaler = torch.amp.GradScaler(device="cuda") if cfg.amp and device.startswith("cuda") \
        else _NoopScaler()

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory and device.startswith("cuda"),
                              drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=cfg.pin_memory and device.startswith("cuda"),
                            drop_last=False)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[EpochLog] = []
    best_ckpt_path = output_dir / "best.pth"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = _one_epoch(model, train_loader, optimiser, scaler,
                                device, cfg.amp, loss_fn, train=True)
        val_loss = _one_epoch(model, val_loader, optimiser, scaler,
                              device, cfg.amp, loss_fn, train=False)
        scheduler.step(epoch)
        lr = float(optimiser.param_groups[0]["lr"])
        dt = time.time() - t0
        _LOG.info("epoch %d/%d  train=%.4f  val=%.4f  lr=%.2e  (%.1fs)",
                  epoch, cfg.epochs, train_loss, val_loss, lr, dt)
        history.append(EpochLog(epoch=epoch, train_loss=train_loss,
                                val_loss=val_loss, lr=lr, seconds=dt))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            _default_save(model, best_ckpt_path, source_checkpoint_meta,
                          saver=save_checkpoint_fn)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                _LOG.info("early stop at epoch %d (no val improvement in %d "
                          "epochs)", epoch, cfg.patience)
                break

    # Finalise manifest + sha256
    manifest.best_epoch = best_epoch
    manifest.best_val_loss = float(best_val)
    manifest.epochs_log = [EpochRecord(epoch=h.epoch,
                                       train_loss=h.train_loss,
                                       val_loss=h.val_loss,
                                       lr=h.lr,
                                       seconds=h.seconds) for h in history]
    manifest.output_dir = str(output_dir.resolve())
    manifest.output_checkpoint = best_ckpt_path.name
    manifest.output_sha256 = sha256_file(best_ckpt_path) if best_ckpt_path.is_file() else ""
    manifest_path = write_training_manifest(output_dir, manifest)

    return TrainResult(output_dir=output_dir,
                      checkpoint_path=best_ckpt_path,
                      manifest_path=manifest_path,
                      manifest=manifest,
                      history=history)


class _NoopScaler:
    """A GradScaler-shaped object that does nothing. Used on CPU where
    AMP is a no-op."""

    def scale(self, loss: Any) -> Any:
        return loss

    def step(self, optimiser: Any) -> None:
        optimiser.step()

    def update(self) -> None:
        pass


def _default_save(model: Any, path: Path, source_meta: dict,
                  saver: Callable[[Any, Path, dict], None] | None) -> None:
    """Save a fine-tuned checkpoint in upstream-compatible format."""
    if saver is not None:
        saver(model, path, source_meta)
        return
    sd = {k.removeprefix("module."): v
          for k, v in model.state_dict().items()}
    # Build a payload the upstream predict loader will consume.
    payload = dict(source_meta)
    payload["state_dict"] = sd
    # Upstream reads model_version from this dict and asserts equality
    # vs the model object's `.version`; preserve the source value if
    # present, else fall back to the one on the model instance (Albunet
    # exposes `.version`).
    if "model_version" not in payload and hasattr(model, "version"):
        payload["model_version"] = model.version
    torch.save(payload, str(path))


__all__ = [
    "EpochLog",
    "TrainConfig",
    "TrainResult",
    "bce_dice_loss",
    "build_albunet_from_checkpoint",
    "seed_everything",
    "train_one_run",
]
