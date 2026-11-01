"""Phase 2 post-processing: convert raw model output into a vector
polygon set, optionally splitting touching buildings along the way.

Two methods, selected by `postprocess.method` in a config:

    argmax_interior    Phase 2a. The model is 3-class; take the
                       interior class only (border pixels become
                       separators), then vectorise as today.
    watershed          Phase 2b. The model is binary; split touching
                       predictions with a distance-transform
                       watershed, then polygonise per label.

Both ultimately produce a per-tile binary mask directory with the
same slippy-map layout `abd vectorize` consumes, so the rest of the
pipeline is unchanged.
"""

from bfbench.postprocess.three_class import argmax_interior
from bfbench.postprocess.watershed import (
    DEFAULT_MIN_MARKER_DISTANCE_M,
    DEFAULT_PROB_THRESHOLD,
    WatershedConfig,
    watershed_split_mask,
)

__all__ = [
    "DEFAULT_MIN_MARKER_DISTANCE_M",
    "DEFAULT_PROB_THRESHOLD",
    "WatershedConfig",
    "argmax_interior",
    "watershed_split_mask",
]
