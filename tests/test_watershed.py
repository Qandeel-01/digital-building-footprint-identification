"""Correctness tests for the Phase 2b watershed instance separator.

Covers the three invariants Phase 2b depends on:

  1. Isolated building -> ONE label. If this breaks, we over-segment
     every single building into multiple pieces.
  2. Two touching discs (narrow isthmus) -> TWO labels. This is the
     whole point of Phase 2b.
  3. Config validation + empty-input handling.

Also tests the companion `label_raster_to_binary_with_gaps` since
that is what the upstream `abd vectorize` consumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from bfbench.postprocess.three_class import argmax_interior
from bfbench.postprocess.watershed import (
    WatershedConfig,
    label_raster_to_binary_with_gaps,
    watershed_split_mask,
)


def _disc(cx: int, cy: int, r: int, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a filled disc of radius `r` centred at (cx, cy)."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


# ---------------------------------------------------------------------------
# Invariant 1: isolated foreground blob -> exactly one label
# ---------------------------------------------------------------------------

def test_isolated_blob_yields_one_label():
    mask = _disc(64, 64, 20, (128, 128)).astype(np.uint8)
    labels = watershed_split_mask(mask, pixel_resolution_m=0.5)
    nz = labels[labels > 0]
    assert nz.size > 0
    unique = np.unique(nz)
    assert len(unique) == 1, (
        f"isolated disc was split into {len(unique)} labels "
        f"(expected 1): {unique.tolist()}"
    )


# ---------------------------------------------------------------------------
# Invariant 2: two touching discs -> two labels (THE Phase 2b test)
# ---------------------------------------------------------------------------

def test_two_touching_discs_split_into_two_labels():
    """Two 20-px-radius discs centred 36 px apart -> touching with a
    narrow isthmus. Watershed on EDT must split them into two labels."""
    r = 20
    gap = 4  # overlap of 2*r - 36 = 40 - 36 = 4 px isthmus
    left = _disc(50, 64, r, (128, 128))
    right = _disc(50 + 2 * r - gap, 64, r, (128, 128))
    mask = (left | right).astype(np.uint8)

    labels = watershed_split_mask(
        mask, pixel_resolution_m=0.5,
        cfg=WatershedConfig(min_marker_distance_m=3.0),  # 6 px minimum
    )
    nz = labels[labels > 0]
    unique = np.unique(nz)
    assert len(unique) == 2, (
        f"two touching discs produced {len(unique)} labels "
        f"(expected 2): {unique.tolist()}"
    )

    # Each label should contain the appropriate disc centre.
    centre_left = labels[64, 50]
    centre_right = labels[64, 50 + 2 * r - gap]
    assert centre_left != 0 and centre_right != 0
    assert centre_left != centre_right, (
        "both disc centres landed in the same watershed basin"
    )


# ---------------------------------------------------------------------------
# Binary-with-gaps conversion
# ---------------------------------------------------------------------------

def test_label_raster_to_binary_with_gaps_separates_touching_labels():
    """Build a label raster where labels 1 and 2 are side-by-side with
    no bg gap; the converter must insert a 1-px bg strip between them."""
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:12, 4:10] = 1
    labels[4:12, 10:14] = 2
    binary = label_raster_to_binary_with_gaps(labels)
    # Where labels differ along the x=10 edge, binary must be 0.
    assert binary[6, 9] == 0 or binary[6, 10] == 0, (
        "no gap inserted between touching labels"
    )
    # And the far interior of each label is still 1.
    assert binary[6, 5] == 1
    assert binary[6, 13] == 1


def test_label_raster_to_binary_with_gaps_no_op_on_isolated_label():
    """Single label, no boundary with ANOTHER label -> no gap carved."""
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 2:6] = 1
    binary = label_raster_to_binary_with_gaps(labels)
    np.testing.assert_array_equal(binary, (labels == 1).astype(np.uint8))


# ---------------------------------------------------------------------------
# Empty / edge-case input
# ---------------------------------------------------------------------------

def test_all_background_returns_zero_labels():
    mask = np.zeros((32, 32), dtype=np.uint8)
    labels = watershed_split_mask(mask, pixel_resolution_m=0.5)
    assert labels.shape == mask.shape
    assert labels.max() == 0


def test_single_foreground_pixel_still_gets_a_label():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[7, 7] = 1
    labels = watershed_split_mask(mask, pixel_resolution_m=0.5)
    assert labels[7, 7] == 1


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_invalid_marker_distance_rejected():
    with pytest.raises(ValueError, match="> 0"):
        WatershedConfig(min_marker_distance_m=0.0)
    with pytest.raises(ValueError, match="> 0"):
        WatershedConfig(min_marker_distance_m=-1.0)


def test_invalid_prob_threshold_rejected():
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        WatershedConfig(prob_threshold=0.0)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        WatershedConfig(prob_threshold=1.0)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        WatershedConfig(prob_threshold=-0.5)


def test_bad_mask_shape_rejected():
    with pytest.raises(ValueError, match="2-D"):
        watershed_split_mask(np.zeros((2, 2, 2), dtype=np.uint8),
                             pixel_resolution_m=0.5)


def test_bad_resolution_rejected():
    with pytest.raises(ValueError, match="> 0"):
        watershed_split_mask(np.zeros((2, 2), dtype=np.uint8),
                             pixel_resolution_m=0.0)


# ---------------------------------------------------------------------------
# argmax_interior interoperability (Phase 2a path)
# ---------------------------------------------------------------------------

def test_argmax_interior_from_label_raster():
    mask = np.array([[0, 1, 2],
                     [1, 2, 0]], dtype=np.uint8)
    out = argmax_interior(mask)
    np.testing.assert_array_equal(out, np.array([[0, 1, 0],
                                                 [1, 0, 0]], dtype=np.uint8))


def test_argmax_interior_from_prob_map():
    # 3-channel probs; class 1 wins at (0,0), class 2 at (0,1), class 0 at (1,1).
    probs = np.zeros((3, 2, 2), dtype=np.float32)
    probs[1, 0, 0] = 0.9
    probs[0, 0, 0] = 0.1
    probs[2, 0, 1] = 0.8
    probs[1, 0, 1] = 0.2
    probs[1, 1, 0] = 0.6
    probs[0, 1, 0] = 0.4
    probs[0, 1, 1] = 0.7
    probs[1, 1, 1] = 0.3
    out = argmax_interior(probs)
    np.testing.assert_array_equal(out, np.array([[1, 0],
                                                 [1, 0]], dtype=np.uint8))


def test_argmax_interior_bad_shape_rejected():
    with pytest.raises(ValueError, match="2-D label|prob map"):
        argmax_interior(np.zeros((2, 3, 3, 3)))
