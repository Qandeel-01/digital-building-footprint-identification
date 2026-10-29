"""Tests for the training manifest + checksum helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bfbench.train.manifest import (
    TrainingManifest,
    sha256_bytes,
    sha256_file,
    write_training_manifest,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    data = b"the quick brown fox" * 1000
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(path) == expected
    assert sha256_bytes(data) == expected


def test_write_manifest_asserts_sha_consistency(tmp_path: Path) -> None:
    """If `manifest.output_sha256` is set but does NOT match the file's
    actual sha256, writing must fail. This is what guards against a
    stale manifest shipping with a swapped checkpoint."""
    pth = tmp_path / "best.pth"
    pth.write_bytes(b"content-A")
    manifest = TrainingManifest(
        config_source="x",
        config_sha256="x",
        source_weight_path="x",
        source_weight_sha256="x",
        aoi_name="x",
        output_dir=str(tmp_path),
        output_checkpoint="best.pth",
        output_sha256="deadbeef",  # intentionally wrong
    )
    with pytest.raises(ValueError, match="sha256"):
        write_training_manifest(tmp_path, manifest)


def test_write_manifest_populates_missing_sha(tmp_path: Path) -> None:
    """If `manifest.output_sha256` is left blank, the writer computes it
    (same semantics as scripts/fetch_weights.py on first download)."""
    pth = tmp_path / "best.pth"
    pth.write_bytes(b"content-A")
    expected = hashlib.sha256(b"content-A").hexdigest()
    manifest = TrainingManifest(
        config_source="x",
        config_sha256="x",
        source_weight_path="x",
        source_weight_sha256="x",
        aoi_name="x",
        output_dir=str(tmp_path),
        output_checkpoint="best.pth",
        output_sha256="",
    )
    out_path = write_training_manifest(tmp_path, manifest)
    assert out_path.name == "training_manifest.json"
    assert manifest.output_sha256 == expected
    # Sibling sha256 file written for run-prepared to verify at load.
    sidecar = tmp_path / "best.pth.sha256.json"
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["sha256"] == expected
    # Manifest JSON contains the same value.
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["output_sha256"] == expected
    # Ratios field was serialised as a list (not a tuple).
    assert isinstance(data["split_ratios"], list)


def test_write_manifest_fails_if_checkpoint_missing(tmp_path: Path) -> None:
    manifest = TrainingManifest(
        config_source="x",
        config_sha256="x",
        source_weight_path="x",
        source_weight_sha256="x",
        aoi_name="x",
        output_dir=str(tmp_path),
        output_checkpoint="best.pth",
    )
    with pytest.raises(FileNotFoundError):
        write_training_manifest(tmp_path, manifest)
