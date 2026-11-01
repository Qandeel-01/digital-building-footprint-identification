"""Phase 2a postprocess: take a 3-class model output and emit the
binary interior mask that `abd vectorize` consumes.

The 3-class model's output per tile is either:

    * an argmax PNG with values {0: bg, 1: interior, 2: border}
      (upstream `tile_label_to_file` writes this when the predict
      step receives a 3-class model), OR
    * a (3, H, W) float32 probability map (optional; produced only
      when `scripts/predict_local.py` is run with `--save-probs`).

`argmax_interior` normalises both to a binary HxW `uint8` mask of
`class == CLS_INTERIOR`. Border pixels become 0, which is the point:
they become the gap that separates two touching buildings in the
vectorised output.
"""

from __future__ import annotations

import numpy as np

from bfbench.data.three_class import CLS_INTERIOR, N_CLASSES


def argmax_interior(mask_or_probs: np.ndarray) -> np.ndarray:
    """Normalise a 3-class model output to a binary interior mask.

    Parameters
    ----------
    mask_or_probs
        One of:
          * 2-D uint8 label raster, values in {0, 1, 2}. Returned as
            `(mask_or_probs == 1).astype(uint8)`.
          * 3-D float probability map, shape (N_CLASSES, H, W).
            Returned as `(argmax(axis=0) == 1).astype(uint8)`.
    """
    arr = np.asarray(mask_or_probs)
    if arr.ndim == 2:
        return (arr == CLS_INTERIOR).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == N_CLASSES:
        return (arr.argmax(axis=0) == CLS_INTERIOR).astype(np.uint8)
    raise ValueError(
        f"expected 2-D label raster or (3, H, W) prob map; "
        f"got shape {arr.shape}"
    )


__all__ = ["argmax_interior"]
