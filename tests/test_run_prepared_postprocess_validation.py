"""Tests for `run_prepared._validate_checkpoint_vs_postprocess`.

A 3-class model's per-tile output PNGs carry values in {0, 1, 2}; if
they are fed to `abd vectorize` as-is, the result looks superficially
plausible (plenty of polygons) but is nonsense because class 2
(touching-pair border) is polygonised into thin rings on every
building. This module must prevent that failure mode by raising
loudly before predict runs.

Covers the two silent-garbage cases:

  1. 3-class checkpoint + missing `postprocess.method` -> loud error.
  2. 3-class checkpoint + `postprocess.method: watershed` -> loud
     error (watershed is 2b; argmax_interior is 2a; they are not
     interchangeable for a 3-class model).

And the two legitimate cases:

  3. Binary checkpoint + no postprocess section -> OK (Phase 1).
  4. 3-class checkpoint + `postprocess.method: argmax_interior` -> OK.

Also covers the config-parsing guard that rejects invalid postprocess
method names at `_load` time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from bfbench.run_prepared import (  # noqa: E402
    _load,
    _validate_checkpoint_vs_postprocess,
)


def _write_fake_checkpoint(path: Path, n_classes: int) -> None:
    """Save a minimal upstream-compatible .pth whose only purpose is to
    carry `shape_out`. We do NOT serialise any model state; the
    validator only peeks at the metadata."""
    payload = {
        "nn": "Albunet",
        "encoder": "resnet50",
        "shape_in":  (3, 512, 512),
        "shape_out": (n_classes, 512, 512),
        "state_dict": {},
        "model_version": 2,
        "uuid": "unit-test",
        "loader": "SemSeg",
    }
    torch.save(payload, str(path))


def _write_config(path: Path, work_dir: Path, checkpoint: Path,
                 postprocess: dict | None) -> Path:
    cfg = {
        "aoi":     {"name": "AOI_UNIT_TEST", "provider": "spacenet2",
                   "aoi_id": 2, "split": "train", "image_kind": "PS-RGB"},
        "adapter": {"target_gsd_m": 0.5, "tile_size": 512,
                   "zoom_tolerance": 0.2, "keep_borders": True,
                   "channel_name": "images"},
        "reference": {"dedup": {"method": "touching_union", "buffer_m": 0.5,
                               "iou_threshold": 0.5, "equal_area_epsg": 6933}},
        "sampling": {"n_tiles": 10, "seed": 0, "min_valid_fraction": 1.0},
        "abd":     {"config": "external/abd/input/config.toml",
                   "checkpoint": str(checkpoint),
                   "vectorize_type": "Building", "metatiles": True},
        "run":     {"work_dir": str(work_dir)},
    }
    if postprocess is not None:
        cfg["postprocess"] = postprocess
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The two silent-garbage cases MUST raise.
# ---------------------------------------------------------------------------

def test_three_class_checkpoint_without_postprocess_raises_loudly(
    tmp_path: Path,
) -> None:
    """The headline anti-regression: a 3-class checkpoint being run
    without `postprocess.method` was the silent-garbage mode called
    out in Phase 2a's Kaggle runbook. It must crash before predict."""
    ckpt = tmp_path / "phase2a.pth"
    _write_fake_checkpoint(ckpt, n_classes=3)
    cfg_path = _write_config(tmp_path / "cfg.yaml", tmp_path, ckpt,
                             postprocess=None)
    cfg = _load(cfg_path)
    assert cfg.postprocess_method is None  # sanity
    with pytest.raises(RuntimeError) as exc:
        _validate_checkpoint_vs_postprocess(cfg)
    msg = str(exc.value).lower()
    assert "3-class" in msg or "shape_out" in msg
    assert "argmax_interior" in msg, (
        "error should name the exact postprocess.method the user must set"
    )


def test_three_class_checkpoint_with_watershed_raises_loudly(
    tmp_path: Path,
) -> None:
    """watershed is Phase 2b; it does NOT substitute for argmax_interior
    on a 3-class model. The validator must catch this."""
    ckpt = tmp_path / "phase2a.pth"
    _write_fake_checkpoint(ckpt, n_classes=3)
    cfg_path = _write_config(
        tmp_path / "cfg.yaml", tmp_path, ckpt,
        postprocess={"method": "watershed", "min_marker_distance_m": 5.0},
    )
    cfg = _load(cfg_path)
    assert cfg.postprocess_method == "watershed"
    with pytest.raises(RuntimeError, match="3-class"):
        _validate_checkpoint_vs_postprocess(cfg)


def test_binary_checkpoint_with_argmax_interior_raises_loudly(
    tmp_path: Path,
) -> None:
    """Inverse mistake: the user forgot to switch the Phase 2a config
    back to binary after fine-tuning a plain Phase 1 model. The
    argmax_interior transform would discard almost all pixels."""
    ckpt = tmp_path / "phase1.pth"
    _write_fake_checkpoint(ckpt, n_classes=1)
    cfg_path = _write_config(
        tmp_path / "cfg.yaml", tmp_path, ckpt,
        postprocess={"method": "argmax_interior"},
    )
    cfg = _load(cfg_path)
    with pytest.raises(RuntimeError, match="binary"):
        _validate_checkpoint_vs_postprocess(cfg)


# ---------------------------------------------------------------------------
# The two legitimate cases MUST pass silently.
# ---------------------------------------------------------------------------

def test_binary_checkpoint_without_postprocess_is_phase_1_ok(tmp_path: Path) -> None:
    ckpt = tmp_path / "phase1.pth"
    _write_fake_checkpoint(ckpt, n_classes=1)
    cfg_path = _write_config(tmp_path / "cfg.yaml", tmp_path, ckpt,
                             postprocess=None)
    cfg = _load(cfg_path)
    # Should not raise.
    _validate_checkpoint_vs_postprocess(cfg)


def test_three_class_checkpoint_with_argmax_interior_is_phase_2a_ok(
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "phase2a.pth"
    _write_fake_checkpoint(ckpt, n_classes=3)
    cfg_path = _write_config(
        tmp_path / "cfg.yaml", tmp_path, ckpt,
        postprocess={"method": "argmax_interior"},
    )
    cfg = _load(cfg_path)
    _validate_checkpoint_vs_postprocess(cfg)


def test_binary_checkpoint_with_watershed_is_phase_2b_ok(tmp_path: Path) -> None:
    """Phase 2b: watershed on top of a binary (Phase 1) checkpoint."""
    ckpt = tmp_path / "phase1.pth"
    _write_fake_checkpoint(ckpt, n_classes=1)
    cfg_path = _write_config(
        tmp_path / "cfg.yaml", tmp_path, ckpt,
        postprocess={"method": "watershed", "min_marker_distance_m": 5.0},
    )
    cfg = _load(cfg_path)
    _validate_checkpoint_vs_postprocess(cfg)


# ---------------------------------------------------------------------------
# Config parse guards.
# ---------------------------------------------------------------------------

def test_invalid_postprocess_method_rejected_at_load(tmp_path: Path) -> None:
    ckpt = tmp_path / "phase1.pth"
    _write_fake_checkpoint(ckpt, n_classes=1)
    cfg_path = _write_config(
        tmp_path / "cfg.yaml", tmp_path, ckpt,
        postprocess={"method": "majority_vote"},  # typo
    )
    with pytest.raises(ValueError, match="postprocess.method must be one of"):
        _load(cfg_path)


def test_missing_checkpoint_file_warns_but_does_not_block(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """If we can't peek shape_out (e.g. the file does not exist yet
    because the user is running --stop-after adapt), validation must
    NOT block the pipeline. It only blocks when it can prove a
    mismatch."""
    import logging  # noqa: PLC0415
    cfg_path = _write_config(
        tmp_path / "cfg.yaml", tmp_path, tmp_path / "nonexistent.pth",
        postprocess={"method": "argmax_interior"},
    )
    cfg = _load(cfg_path)
    with caplog.at_level(logging.WARNING, logger="bfbench.run_prepared"):
        _validate_checkpoint_vs_postprocess(cfg)  # must not raise
    joined = " ".join(r.message for r in caplog.records)
    assert "could not peek shape_out" in joined
