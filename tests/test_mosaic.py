"""Tests for the mosaic coverage check."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from bfbench.data.mosaic import build_vrt, tile_is_fully_covered


def _gdalbuildvrt_available() -> bool:
    """True if gdalbuildvrt is on PATH or in the running env's Library/bin."""
    if shutil.which("gdalbuildvrt"):
        return True
    cand = Path(sys.prefix) / "Library" / "bin" / "gdalbuildvrt.exe"
    if cand.is_file():
        return True
    try:
        return subprocess.run(["gdalbuildvrt", "--version"],
                              capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _gdalbuildvrt_available(),
    reason="gdalbuildvrt not on PATH",
)


def _write_chip(path: Path, left: float, bottom: float, right: float, top: float,
                width: int, height: int, fill: int = 100) -> None:
    """Write a single-band uint16 chip filled with `fill` in EPSG:4326."""
    tr = from_bounds(left, bottom, right, top, width, height)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width, count=1,
        dtype=np.uint16, crs="EPSG:4326", transform=tr,
    ) as dst:
        dst.write(np.full((height, width), fill, dtype=np.uint16), 1)


def test_fully_covered_tile_returns_true(tmp_path: Path):
    """One chip that fully covers the requested bbox -> covered."""
    chip = tmp_path / "chip.tif"
    _write_chip(chip, -1.0, -1.0, 1.0, 1.0, 200, 200)
    vrt = tmp_path / "m.vrt"
    build_vrt([chip], vrt)
    # bbox strictly inside the chip
    assert tile_is_fully_covered(vrt, (-0.5, -0.5, 0.5, 0.5), min_valid_fraction=1.0)


def test_partially_covered_tile_returns_false(tmp_path: Path):
    """Chip covers only the left half of the requested bbox -> not covered."""
    chip = tmp_path / "chip.tif"
    _write_chip(chip, -1.0, -1.0, 0.0, 1.0, 100, 200)  # right = 0
    vrt = tmp_path / "m.vrt"
    build_vrt([chip], vrt)
    assert not tile_is_fully_covered(vrt, (-0.5, -0.5, 0.5, 0.5), min_valid_fraction=1.0)


def test_bbox_outside_mosaic_returns_false(tmp_path: Path):
    chip = tmp_path / "chip.tif"
    _write_chip(chip, -1.0, -1.0, 1.0, 1.0, 200, 200)
    vrt = tmp_path / "m.vrt"
    build_vrt([chip], vrt)
    # bbox in a far-away location, entirely outside the mosaic
    assert not tile_is_fully_covered(vrt, (10.0, 10.0, 11.0, 11.0), min_valid_fraction=1.0)


def test_min_valid_fraction_relaxes_the_threshold(tmp_path: Path):
    """A tile that is 50% covered fails at 1.0 but passes at 0.4."""
    chip = tmp_path / "chip.tif"
    _write_chip(chip, -1.0, -1.0, 0.0, 1.0, 100, 200)
    vrt = tmp_path / "m.vrt"
    build_vrt([chip], vrt)
    bbox = (-0.5, -0.5, 0.5, 0.5)
    assert not tile_is_fully_covered(vrt, bbox, min_valid_fraction=1.0)
    assert tile_is_fully_covered(vrt, bbox, min_valid_fraction=0.4)


def test_zero_filled_chip_is_not_valid(tmp_path: Path):
    """A chip filled entirely with zeros must NOT be counted as covered."""
    chip = tmp_path / "chip.tif"
    _write_chip(chip, -1.0, -1.0, 1.0, 1.0, 200, 200, fill=0)
    vrt = tmp_path / "m.vrt"
    build_vrt([chip], vrt)
    assert not tile_is_fully_covered(vrt, (-0.5, -0.5, 0.5, 0.5), min_valid_fraction=1.0)
