"""Tests for deterministic tile sampling."""

from __future__ import annotations

import mercantile
import pytest

from bfbench.data.sampling import sample_tiles


def _grid(n: int) -> list[mercantile.Tile]:
    """A deterministic list of n tiles at z=17 for testing."""
    return [mercantile.Tile(x=1000 + i, y=5000, z=17) for i in range(n)]


def test_same_seed_yields_identical_tiles():
    tiles = _grid(500)
    a = sample_tiles(tiles, n=50, seed=42)
    b = sample_tiles(tiles, n=50, seed=42)
    assert a.tiles == b.tiles
    assert a.tile_ids_hash == b.tile_ids_hash


def test_different_seed_yields_different_tiles():
    tiles = _grid(500)
    a = sample_tiles(tiles, n=50, seed=42)
    b = sample_tiles(tiles, n=50, seed=43)
    assert a.tiles != b.tiles
    assert a.tile_ids_hash != b.tile_ids_hash


def test_input_order_does_not_matter():
    """Selection depends on the SORTED input, not on the order the
    caller passed. Reversing the input must not change the output."""
    tiles = _grid(200)
    a = sample_tiles(tiles, n=25, seed=7)
    b = sample_tiles(list(reversed(tiles)), n=25, seed=7)
    assert a.tiles == b.tiles
    assert a.tile_ids_hash == b.tile_ids_hash


def test_n_larger_than_candidates_returns_all():
    tiles = _grid(30)
    res = sample_tiles(tiles, n=100, seed=1)
    assert len(res.tiles) == 30
    assert res.n_requested == 100
    assert res.n_candidates == 30


def test_n_zero_raises():
    with pytest.raises(ValueError):
        sample_tiles(_grid(10), n=0, seed=1)


def test_output_is_sorted_by_zxy():
    res = sample_tiles(_grid(500), n=50, seed=42)
    xs = [(t.z, t.x, t.y) for t in res.tiles]
    assert xs == sorted(xs)
