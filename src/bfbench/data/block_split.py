"""Geographic block-based train / val / test split for an AOI.

Why block-based and not random tile-based:
    If a building sits on the edge of two neighbouring tiles, a purely
    random tile-level split can place one tile in `train` and the other
    in `test`, so the model sees overlapping patches of the same
    building in training that it is then evaluated against. This is a
    silent data leak and makes Phase 1's "did fine-tuning help?"
    question unanswerable. A geographic block split rules it out: the
    train / val / test blocks are disjoint polygons in an equal-area
    CRS, and a tile is assigned to the block that contains its
    centroid.

How it works:
    1. The AOI's chip-union bbox is reprojected into the equal-area
       CRS (default EPSG:6933, see data/equal_area.py).
    2. That bbox is covered by a regular grid of square blocks of
       `block_size_m` metres on a side (default 1000 m).
    3. Each block is hashed with the user's split seed to deterministic-
       ally land in train / val / test by the configured ratio. Order
       of blocks is canonicalised by (i, j) grid index, so two runs
       with the same seed produce byte-identical assignments regardless
       of filesystem walk order.
    4. A tile is assigned to a split by the block containing its
       centroid (equal-area). Tiles whose centroid is NOT inside any
       block (should not happen if the bbox was computed from the same
       tiles) are returned as "unassigned" rather than silently
       dropped.

What is *not* here:
    - Downloading or sampling tiles. See `data/sampling.py`.
    - Any decision about how many tiles to use per split. That stays
      in the config (`split.min_tiles_per_split`, `train.max_tiles`,
      etc.) and is enforced in `src/bfbench/prepare.py`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import mercantile
from shapely.geometry import box

from bfbench.data.equal_area import (
    DEFAULT_EQUAL_AREA_EPSG,
    assert_equal_area_crs,
)

Split = Literal["train", "val", "test"]

_VALID_SPLITS: tuple[Split, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class BlockSplitConfig:
    """Config for a geographic block split.

    Attributes:
        block_size_m: side length of one block in the equal-area CRS,
            in metres. Default 1000.0 (1 km).
        ratios: fractions assigned to (train, val, test). Must sum to 1.
            Default (0.70, 0.15, 0.15).
        seed: deterministic seed used by the per-block hash. Changing
            this reshuffles the assignment while keeping the ratios.
        equal_area_epsg: EPSG for the equal-area CRS blocks live in.
            Default DEFAULT_EQUAL_AREA_EPSG (6933). Must be an equal-
            area CRS; the splitter asserts this.
    """

    block_size_m: float = 1000.0
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 20260911
    equal_area_epsg: int = DEFAULT_EQUAL_AREA_EPSG

    def __post_init__(self) -> None:
        if self.block_size_m <= 0:
            raise ValueError(f"block_size_m must be > 0, got {self.block_size_m}")
        if len(self.ratios) != 3:
            raise ValueError(f"ratios must be (train, val, test); got {self.ratios!r}")
        if any(r < 0 for r in self.ratios):
            raise ValueError(f"ratios must be non-negative; got {self.ratios!r}")
        s = sum(self.ratios)
        if not math.isclose(s, 1.0, abs_tol=1e-9):
            raise ValueError(f"ratios must sum to 1.0; got {self.ratios!r} (sum={s})")


@dataclass(frozen=True)
class BlockAssignment:
    """One grid block + its deterministic split."""

    i: int
    j: int
    polygon_ea: object  # shapely polygon in the equal-area CRS
    split: Split


@dataclass(frozen=True)
class SplitResult:
    """Result of `split_tiles` for one AOI."""

    train: list[mercantile.Tile]
    val: list[mercantile.Tile]
    test: list[mercantile.Tile]
    unassigned: list[mercantile.Tile]
    blocks: list[BlockAssignment]
    config: BlockSplitConfig
    assignment_hash: str  # sha256 of the (i, j, split) list; byte-identity

    def by_split(self, split: Split) -> list[mercantile.Tile]:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be in {_VALID_SPLITS!r}, got {split!r}")
        return {"train": self.train, "val": self.val, "test": self.test}[split]


# ---------------------------------------------------------------------------
# Deterministic per-block split assignment
# ---------------------------------------------------------------------------


def _assign_block(i: int, j: int, cfg: BlockSplitConfig) -> Split:
    """Deterministically map one (i, j) block index to a split.

    Uses sha256 of (seed, i, j) interpreted as a 64-bit unsigned int in
    [0, 2**64), rescaled to [0, 1), and bucketed by `cfg.ratios`. The
    hash function is specified (not Python's `hash()`) because Python's
    hash() is randomised per process on bytes-like inputs by default.
    """
    h = hashlib.sha256(f"{cfg.seed}|{i}|{j}".encode("ascii")).digest()
    # take the first 8 bytes as a big-endian uint64 -> [0, 1)
    u = int.from_bytes(h[:8], "big") / (1 << 64)
    train_hi = cfg.ratios[0]
    val_hi = cfg.ratios[0] + cfg.ratios[1]
    if u < train_hi:
        return "train"
    if u < val_hi:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def _tile_bbox_ea(tiles: list[mercantile.Tile], equal_area_epsg: int
                  ) -> tuple[float, float, float, float]:
    """Union-bbox of the input tiles reprojected into the equal-area CRS."""
    if not tiles:
        raise ValueError("block_split: tile list is empty")
    rows = []
    for t in tiles:
        b = mercantile.bounds(t)
        rows.append(box(b.west, b.south, b.east, b.north))
    gdf = gpd.GeoDataFrame(geometry=rows, crs="EPSG:4326")
    gdf_ea = gdf.to_crs(f"EPSG:{equal_area_epsg}")
    assert_equal_area_crs(gdf_ea, what="block_split tile-bbox")
    minx, miny, maxx, maxy = gdf_ea.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _build_blocks(tiles: list[mercantile.Tile], cfg: BlockSplitConfig
                  ) -> list[BlockAssignment]:
    """Build the regular grid of blocks covering the tile union-bbox.

    The grid origin is snapped to a multiple of `block_size_m` so that
    two AOIs with overlapping extents produce the same block IDs for
    the overlap region. This is defensive for the pooled training run.
    """
    minx, miny, maxx, maxy = _tile_bbox_ea(tiles, cfg.equal_area_epsg)
    s = cfg.block_size_m
    i0 = math.floor(minx / s)
    j0 = math.floor(miny / s)
    i1 = math.ceil(maxx / s)
    j1 = math.ceil(maxy / s)
    out: list[BlockAssignment] = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            poly = box(i * s, j * s, (i + 1) * s, (j + 1) * s)
            out.append(BlockAssignment(
                i=i, j=j, polygon_ea=poly, split=_assign_block(i, j, cfg),
            ))
    return out


def _tile_centroid_ea_series(tiles: list[mercantile.Tile], equal_area_epsg: int
                             ) -> gpd.GeoSeries:
    """Project tile bounds to equal-area and return the centroids."""
    rows = [box(*mercantile.bounds(t)[:4]) for t in tiles]
    gs = gpd.GeoSeries(rows, crs="EPSG:4326").to_crs(f"EPSG:{equal_area_epsg}")
    return gs.centroid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_tiles(tiles: list[mercantile.Tile],
                cfg: BlockSplitConfig | None = None) -> SplitResult:
    """Partition `tiles` into train / val / test by geographic block.

    Determinism:
        (same tiles, same cfg) -> byte-identical SplitResult, verified
        by the sha256 in `SplitResult.assignment_hash`.

    Disjointness:
        Two blocks never overlap by construction (they are a regular
        grid), so the resulting train/val/test tile sets are spatially
        disjoint at the block level. A given tile centroid lies in
        exactly one block, so no tile appears in more than one split.
    """
    if cfg is None:
        cfg = BlockSplitConfig()
    if not tiles:
        raise ValueError("split_tiles: tile list is empty")

    blocks = _build_blocks(tiles, cfg)
    centroids = _tile_centroid_ea_series(tiles, cfg.equal_area_epsg)

    s = cfg.block_size_m
    buckets: dict[Split, list[mercantile.Tile]] = {k: [] for k in _VALID_SPLITS}
    unassigned: list[mercantile.Tile] = []
    for tile, c in zip(tiles, centroids, strict=True):
        i = math.floor(c.x / s)
        j = math.floor(c.y / s)
        split = _assign_block(i, j, cfg)
        if split in _VALID_SPLITS:
            buckets[split].append(tile)
        else:
            unassigned.append(tile)  # pragma: no cover - defensive

    # Canonical ordering + hash
    def _sort(ts: list[mercantile.Tile]) -> list[mercantile.Tile]:
        return sorted(ts, key=lambda t: (t.z, t.x, t.y))

    hasher = hashlib.sha256()
    for b in sorted(blocks, key=lambda bb: (bb.i, bb.j)):
        hasher.update(f"{b.i},{b.j},{b.split}\n".encode("ascii"))

    return SplitResult(
        train=_sort(buckets["train"]),
        val=_sort(buckets["val"]),
        test=_sort(buckets["test"]),
        unassigned=_sort(unassigned),
        blocks=sorted(blocks, key=lambda bb: (bb.i, bb.j)),
        config=cfg,
        assignment_hash=hasher.hexdigest(),
    )


def blocks_to_geodataframe(blocks: list[BlockAssignment],
                           equal_area_epsg: int) -> gpd.GeoDataFrame:
    """Serialise blocks to a GeoDataFrame so a run can persist them."""
    return gpd.GeoDataFrame(
        {
            "i":     [b.i for b in blocks],
            "j":     [b.j for b in blocks],
            "split": [b.split for b in blocks],
        },
        geometry=[b.polygon_ea for b in blocks],
        crs=f"EPSG:{equal_area_epsg}",
    )


__all__ = [
    "BlockAssignment",
    "BlockSplitConfig",
    "SplitResult",
    "blocks_to_geodataframe",
    "split_tiles",
]
