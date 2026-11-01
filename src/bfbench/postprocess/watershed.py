"""Watershed instance separation for Phase 2b.

Input: a per-tile binary mask (`uint8` or `bool`) from an already-
trained model. The model is NOT retrained; this is pure
post-processing. The brief phrases it as "watershed instance
separation on the probability map", and when a probability map is
available we use it directly as the elevation surface. When only a
binary mask is on disk (what upstream `tile_label_to_file` writes
today), we fall back to the standard pattern of using the Euclidean
distance transform of the foreground as the elevation.

Algorithm:

    1. Foreground = (prob >= threshold) if prob given else (binary > 0).
    2. dist = EDT(foreground).
    3. markers = local maxima of dist, separated by at least
       `min_marker_distance_m` metres (converted to pixels using
       the tile's pixel resolution).
    4. labels = watershed(-dist, markers, mask=foreground).
    5. Return either (a) a per-label binary mask (one label at a
       time; useful for polygonising via rasterio.features.shapes)
       or (b) a single uint8 mask whose foreground is relabelled
       such that touching components are now separated by a 1-px
       background gap. (b) drops into the existing `abd vectorize`
       flow with no interface change.

Why not use the probability map directly for the elevation surface:
upstream `predict.py` writes an argmax PNG (0 or label index), not
probabilities. Reading-the-logits requires modifying
`scripts/predict_local.py` to dump probabilities alongside labels;
that is in scope for a Phase 2b follow-up but not for the first cut.
When probabilities ARE available (`watershed_split_mask(prob_map=...)`)
the function uses them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

DEFAULT_MIN_MARKER_DISTANCE_M = 5.0
DEFAULT_PROB_THRESHOLD = 0.5


@dataclass(frozen=True)
class WatershedConfig:
    """Config for the watershed instance-separation post-process.

    Attributes:
        min_marker_distance_m: minimum metric separation between two
            seed markers (= two detected building centres). Default
            5 m: smaller than a single-family house footprint, larger
            than the typical gap between adjacent row houses on
            SpaceNet 2. Expressed in metres; `watershed_split_mask`
            converts to pixels using `pixel_resolution_m`.
        prob_threshold: binarisation threshold applied when a
            probability map is passed in. Ignored for binary input.
            Default 0.5 (standard sigmoid cut).
    """

    min_marker_distance_m: float = DEFAULT_MIN_MARKER_DISTANCE_M
    prob_threshold: float = DEFAULT_PROB_THRESHOLD

    def __post_init__(self) -> None:
        if self.min_marker_distance_m <= 0:
            raise ValueError(
                f"min_marker_distance_m must be > 0, "
                f"got {self.min_marker_distance_m}"
            )
        if not 0 < self.prob_threshold < 1:
            raise ValueError(
                f"prob_threshold must be in (0, 1), got {self.prob_threshold}"
            )


def watershed_split_mask(mask: np.ndarray,
                         pixel_resolution_m: float,
                         cfg: WatershedConfig | None = None,
                         prob_map: np.ndarray | None = None,
                         ) -> np.ndarray:
    """Split a binary mask into instance labels via watershed.

    Returns an int32 label raster with 0 = background and 1..N = per-
    instance labels. Touching foreground pixels that previously formed
    a single connected component will carry different labels; running
    `rasterio.features.shapes` on `(labels > 0).astype(uint8)` would
    merge them back, so downstream polygonisation must iterate over
    labels (see `label_raster_to_polygons`).

    Parameters
    ----------
    mask
        2-D uint8 or bool array. Non-zero = foreground.
    pixel_resolution_m
        Metres per pixel for this tile. Used to convert the marker-
        distance config from metres to pixels.
    cfg
        WatershedConfig; defaults used when None.
    prob_map
        Optional probability map, same shape as `mask`. When given,
        the EDT elevation is replaced by the probability itself. The
        binary foreground is still `mask > 0` (not derived from
        prob_map) so the caller can decide how to threshold.
    """
    cfg = cfg or WatershedConfig()
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if pixel_resolution_m <= 0:
        raise ValueError(
            f"pixel_resolution_m must be > 0, got {pixel_resolution_m}"
        )
    foreground = mask > 0
    if not foreground.any():
        return np.zeros_like(mask, dtype=np.int32)

    dist = distance_transform_edt(foreground)
    elevation = prob_map if prob_map is not None else dist

    min_dist_px = max(1, int(round(cfg.min_marker_distance_m / pixel_resolution_m)))

    # peak_local_max returns an (N, 2) array of (y, x) peak coords.
    # exclude_border=False so markers near a tile edge still seed.
    coords = peak_local_max(
        dist,
        min_distance=min_dist_px,
        labels=foreground.astype(np.int32),
        exclude_border=False,
    )
    if len(coords) == 0:
        # No clear maximum (very small blob); fall back to one marker
        # at the overall argmax so the whole foreground gets label 1.
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        coords = np.array([[iy, ix]])

    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    labels = watershed(-elevation, markers=markers, mask=foreground)
    return labels.astype(np.int32)


def label_raster_to_binary_with_gaps(labels: np.ndarray) -> np.ndarray:
    """Convert an instance-label raster into a binary mask whose
    touching instances are separated by a 1-px background gap.

    Useful for feeding the output into upstream `abd vectorize`
    without changing its interface: it polygonises connected 1-pixel
    components, so one-pixel gaps between the two formerly-touching
    labels produce two polygons instead of one.

    This is a lossy conversion -- small instances (< 3 px wide) can
    disappear entirely. For a loss-free workflow, polygonise each
    label independently (see `scripts/watershed_vectorize.py`).
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    out = np.zeros_like(labels, dtype=np.uint8)
    out[labels > 0] = 1

    # Find pixels on the boundary between two different labels.
    # Compare each pixel to its right / down neighbour; if different
    # (and both non-zero), the current pixel is on a shared boundary.
    diff_right = np.zeros_like(labels, dtype=bool)
    diff_right[:, :-1] = (labels[:, :-1] != labels[:, 1:]) & \
                         (labels[:, :-1] > 0) & (labels[:, 1:] > 0)
    diff_down = np.zeros_like(labels, dtype=bool)
    diff_down[:-1, :] = (labels[:-1, :] != labels[1:, :]) & \
                        (labels[:-1, :] > 0) & (labels[1:, :] > 0)
    out[diff_right | diff_down] = 0
    return out


__all__ = [
    "DEFAULT_MIN_MARKER_DISTANCE_M",
    "DEFAULT_PROB_THRESHOLD",
    "WatershedConfig",
    "label_raster_to_binary_with_gaps",
    "watershed_split_mask",
]
