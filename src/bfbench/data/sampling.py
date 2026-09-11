"""Deterministic tile sampling.

Given a list of candidate tiles for an AOI, pick `n` of them using a
seeded PRNG so that (same tile list, same seed) -> byte-identical
selection every time. The seed and the selected tile IDs are written
back into the config so the whole benchmark is exactly reproducible
from that file alone.

Sorting the input by (z, x, y) BEFORE sampling is required. `set` and
`dict` iteration order in Python is insertion-order-stable, but the
insertion order can depend on filesystem walk order, which is not
portable. Sorting the input makes the sampling depend on exactly one
input: the tile IDs.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import mercantile


@dataclass(frozen=True)
class SamplingResult:
    tiles: list[mercantile.Tile]
    seed: int
    n_requested: int
    n_candidates: int
    tile_ids_hash: str  # sha256 of the sorted output; a byte-identity check


def sample_tiles(candidates: list[mercantile.Tile], n: int,
                 seed: int) -> SamplingResult:
    """Pick min(n, len(candidates)) tiles from `candidates` with `seed`.

    Order preservation and stability guarantees:
        - Input is sorted by (z, x, y) before sampling. The sampler is
          seeded per call, not from process state.
        - Output tiles are returned in the sorted (z, x, y) order of the
          sampled subset; the SHA-256 in the result is computed on that
          canonical ordering.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    sorted_cands = sorted(candidates, key=lambda t: (t.z, t.x, t.y))
    rng = random.Random(seed)
    k = min(n, len(sorted_cands))
    picked = rng.sample(sorted_cands, k)
    picked_sorted = sorted(picked, key=lambda t: (t.z, t.x, t.y))
    h = hashlib.sha256()
    for t in picked_sorted:
        h.update(f"{t.z},{t.x},{t.y}\n".encode("ascii"))
    return SamplingResult(
        tiles=picked_sorted,
        seed=seed,
        n_requested=n,
        n_candidates=len(sorted_cands),
        tile_ids_hash=h.hexdigest(),
    )
