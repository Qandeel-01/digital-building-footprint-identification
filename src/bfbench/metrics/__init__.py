"""bfbench evaluation metrics: interfaces, docstrings, and failing tests only.

Every function here is a STUB that raises NotImplementedError. Filling
these in is out of scope for the Phase 1/2 harness -- the metric
implementations are authored separately. This module is here so the
evaluation harness in `bfbench.cli.eval` and the figures in
`scripts/figures.py` can be written and tested against a stable set of
signatures, and so tests can assert the decisions the implementer has
to make BEFORE writing any code that hides those decisions in a
one-liner.

Every metric that computes an area, a distance, or a density MUST
receive geometry in an equal-area CRS. This is enforced at each stub's
top via `bfbench.data.equal_area.assert_equal_area_crs`. Do NOT smuggle
a different CRS into the implementation; document why if you do.
"""

from __future__ import annotations

from bfbench.metrics.boundary import boundary_distance
from bfbench.metrics.count import count_error, count_error_aoi
from bfbench.metrics.instance import (
    MatchResult,
    instance_prf,
    match_instances,
    merge_rate,
    split_rate,
)
from bfbench.metrics.pixel import pixel_iou
from bfbench.metrics.stratify import stratify_by_area, stratify_by_density

__all__ = [
    "MatchResult",
    "boundary_distance",
    "count_error",
    "count_error_aoi",
    "instance_prf",
    "match_instances",
    "merge_rate",
    "pixel_iou",
    "split_rate",
    "stratify_by_area",
    "stratify_by_density",
]
