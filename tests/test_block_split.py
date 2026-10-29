"""Geographic block-split tests.

Phase 1 non-negotiable: training and evaluation tiles must be spatially
disjoint. These tests prove it by actual polygon intersection in the
equal-area CRS, not by set-difference of tile IDs.
"""

from __future__ import annotations

import geopandas as gpd
import mercantile
import pytest

from bfbench.data.block_split import (
    BlockSplitConfig,
    blocks_to_geodataframe,
    split_tiles,
)
from bfbench.data.equal_area import DEFAULT_EQUAL_AREA_EPSG


def _tiles_grid(x0: int, y0: int, w: int, h: int, z: int = 17) -> list[mercantile.Tile]:
    """`w` x `h` tile grid starting at (x0, y0, z)."""
    return [mercantile.Tile(x=x0 + dx, y=y0 + dy, z=z)
            for dx in range(w) for dy in range(h)]


# ---------------------------------------------------------------------------
# Zero spatial overlap -- the non-negotiable
# ---------------------------------------------------------------------------

def test_train_test_blocks_have_zero_polygon_overlap_area():
    """The intersection area of the UNION of all train blocks with the
    UNION of all test blocks must be exactly zero in the equal-area CRS.

    This is stronger than 'no shared tile IDs': it rules out any patch
    of ground sitting in both splits, including blocks that happen to
    contain zero tiles.
    """
    # Big enough grid that 70/15/15 lands at least a few blocks in each
    # split for any reasonable seed (LLN >> 15%).
    tiles = _tiles_grid(x0=19872, y0=25580, w=30, h=30, z=17)  # ~Vegas area
    res = split_tiles(tiles, BlockSplitConfig(block_size_m=1000.0, seed=20260911))

    train_blocks = [b.polygon_ea for b in res.blocks if b.split == "train"]
    test_blocks  = [b.polygon_ea for b in res.blocks if b.split == "test"]
    val_blocks   = [b.polygon_ea for b in res.blocks if b.split == "val"]
    # Sanity: all three splits populated at this grid size + seed.
    assert train_blocks and test_blocks and val_blocks

    gs_train = gpd.GeoSeries(train_blocks, crs=f"EPSG:{DEFAULT_EQUAL_AREA_EPSG}")
    gs_test  = gpd.GeoSeries(test_blocks,  crs=f"EPSG:{DEFAULT_EQUAL_AREA_EPSG}")

    union_train = gs_train.union_all()
    union_test  = gs_test.union_all()
    inter = union_train.intersection(union_test)

    # 1 m^2 of slack to tolerate float round-trip noise; realistic blocks
    # are 1 km^2 = 1e6 m^2 so this is 1 ppm.
    assert inter.area < 1.0, (
        f"train/test block polygons overlap by {inter.area:.6f} m^2 "
        "(must be exactly 0)"
    )


def test_tile_assignments_are_pairwise_disjoint():
    """A tile may land in at most one of train / val / test."""
    tiles = _tiles_grid(x0=19872, y0=25580, w=10, h=10, z=17)
    res = split_tiles(tiles, BlockSplitConfig(seed=1234))
    train_ids = {(t.z, t.x, t.y) for t in res.train}
    val_ids   = {(t.z, t.x, t.y) for t in res.val}
    test_ids  = {(t.z, t.x, t.y) for t in res.test}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    # And every input tile ends up assigned somewhere (equal-area bbox
    # construction guarantees this; the test pins it).
    assert len(train_ids | val_ids | test_ids) == len(tiles)
    assert not res.unassigned


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_split_is_deterministic_across_calls():
    tiles = _tiles_grid(19872, 25580, 8, 8)
    a = split_tiles(tiles, BlockSplitConfig(seed=42))
    b = split_tiles(tiles, BlockSplitConfig(seed=42))
    assert a.assignment_hash == b.assignment_hash
    assert [(t.z, t.x, t.y) for t in a.train] == [(t.z, t.x, t.y) for t in b.train]
    assert [(t.z, t.x, t.y) for t in a.test]  == [(t.z, t.x, t.y) for t in b.test]


def test_different_seeds_produce_different_assignments():
    tiles = _tiles_grid(19872, 25580, 8, 8)
    a = split_tiles(tiles, BlockSplitConfig(seed=1))
    b = split_tiles(tiles, BlockSplitConfig(seed=2))
    assert a.assignment_hash != b.assignment_hash, (
        "two different seeds produced byte-identical assignments; "
        "the hash is probably not using the seed"
    )


def test_order_of_input_tiles_does_not_matter():
    tiles = _tiles_grid(19872, 25580, 6, 6)
    reversed_tiles = list(reversed(tiles))
    a = split_tiles(tiles, BlockSplitConfig(seed=99))
    b = split_tiles(reversed_tiles, BlockSplitConfig(seed=99))
    assert a.assignment_hash == b.assignment_hash
    assert {(t.z, t.x, t.y) for t in a.train} == {(t.z, t.x, t.y) for t in b.train}


# ---------------------------------------------------------------------------
# Ratio behaviour
# ---------------------------------------------------------------------------

def test_ratios_hold_within_tolerance_over_many_blocks():
    """Direct test of the per-block hash. Across many (i, j) pairs the
    realised ratios must be close to the requested ones; this is where
    'the hash is roughly uniform' lives, independent of tile geometry.
    """
    from bfbench.data.block_split import _assign_block  # noqa: PLC0415

    cfg = BlockSplitConfig(ratios=(0.70, 0.15, 0.15), seed=0)
    # 10,000 synthetic blocks is enough for 1pp tolerance by Hoeffding.
    pairs = [(i, j) for i in range(100) for j in range(100)]
    counts = {"train": 0, "val": 0, "test": 0}
    for i, j in pairs:
        counts[_assign_block(i, j, cfg)] += 1
    n = len(pairs)
    # 2 percentage points of slack is comfortable at n = 10,000.
    assert abs(counts["train"] / n - 0.70) < 0.02
    assert abs(counts["val"]   / n - 0.15) < 0.02
    assert abs(counts["test"]  / n - 0.15) < 0.02


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_invalid_ratio_sum_is_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        BlockSplitConfig(ratios=(0.5, 0.3, 0.3))


def test_invalid_block_size_is_rejected():
    with pytest.raises(ValueError, match="> 0"):
        BlockSplitConfig(block_size_m=0.0)


def test_empty_tile_list_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        split_tiles([])


# ---------------------------------------------------------------------------
# GeoDataFrame serialisation
# ---------------------------------------------------------------------------

def test_blocks_to_geodataframe_preserves_split_column_and_crs():
    tiles = _tiles_grid(19872, 25580, 6, 6)
    res = split_tiles(tiles, BlockSplitConfig(seed=7))
    gdf = blocks_to_geodataframe(res.blocks, DEFAULT_EQUAL_AREA_EPSG)
    assert set(gdf["split"].unique()).issubset({"train", "val", "test"})
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == DEFAULT_EQUAL_AREA_EPSG
    # The geometry areas should all be ~ block_size_m ** 2
    assert (gdf.geometry.area - res.config.block_size_m ** 2).abs().max() < 1.0


# ---------------------------------------------------------------------------
# Equal-area CRS assertion is actually invoked
# ---------------------------------------------------------------------------

def test_split_uses_equal_area_crs_internally():
    """If someone sets a non-equal-area EPSG in the config, splitting
    must raise before any assignment is written."""
    tiles = _tiles_grid(19872, 25580, 3, 3)
    # EPSG:3857 (Web Mercator) is NOT equal-area.
    cfg = BlockSplitConfig(equal_area_epsg=3857)
    with pytest.raises(Exception) as e:
        split_tiles(tiles, cfg)
    msg = str(e.value).lower()
    assert "equal-area" in msg or "equal area" in msg
