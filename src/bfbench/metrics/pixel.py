"""Pixel-level IoU against a rasterised reference.

Implemented in Phase 3 (matching-independent).

Signature change from Phase 2: the metric now takes the RAW probability
mask directly (numpy array or path to a raster), not vectorised
predictions. The model's native output is a per-pixel probability;
routing it through `abd vectorize` and then back to a raster inside
pixel_iou compounds discretisation error twice. See
docs/metric-decisions.md section 3.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds as _rio_transform_from_bounds

from bfbench.data.equal_area import assert_equal_area_crs


def _load_pred_mask(pred_prob: np.ndarray | Path,
                    expected_shape: tuple[int, int] | None) -> np.ndarray:
    """Return a boolean HxW array. `pred_prob` may be an array, a path,
    or a rasterio dataset-like. Callers pass a probability [0, 1] or a
    binary mask; both are accepted.
    """
    if isinstance(pred_prob, (str, Path)):
        with rasterio.open(str(pred_prob)) as src:
            arr = src.read(1)
    else:
        arr = np.asarray(pred_prob)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"pixel_iou: pred_prob must be a 2-D HxW array (got shape {arr.shape})."
        )
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"pixel_iou: pred_prob shape {arr.shape} does not match expected "
            f"grid {expected_shape} derived from bounds+resolution."
        )
    return arr


def pixel_iou(pred_prob: np.ndarray | Path,
              ref_gdf: gpd.GeoDataFrame,
              bounds: tuple[float, float, float, float],
              resolution: float,
              *,
              threshold: float = 0.5,
              all_touched: bool = False) -> float:
    """Pixel IoU between a raster prediction and a rasterised reference.

    Args:
        pred_prob: HxW numpy array or path to a raster containing either
            a sigmoid probability in [0, 1] (thresholded at `threshold`)
            or a binary/label mask (non-zero = building).
        ref_gdf: reference polygons in the SAME equal-area CRS the
            `bounds` are expressed in (metres).
        bounds: (left, bottom, right, top) in the equal-area CRS. The
            metric asserts that the grid derived from bounds + resolution
            matches pred_prob's shape when pred_prob is a numpy array.
        resolution: metres per pixel. Must match the pred grid.
        threshold: probability threshold applied to `pred_prob` iff its
            values look continuous (max > 1 or dtype float in [0, 1]).
            Ignored if `pred_prob` is already binary.
        all_touched: reference-rasterisation snapping. Default False
            (CENTRE-ONLY, no size-dependent bias). See
            docs/metric-decisions.md section 2 for why.

    Returns:
        IoU as a float in [0, 1]. NaN if both rasters are entirely
        background (empty-vs-empty is undefined; propagating a NaN
        makes that visible to aggregation code).

    Preconditions:
        Reference CRS must be equal-area (assertion enforced).
        Resolution > 0.
    """
    assert_equal_area_crs(ref_gdf, what="pixel_iou(ref_gdf)")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    left, bottom, right, top = bounds
    if not (right > left and top > bottom):
        raise ValueError(
            f"pixel_iou: bounds must satisfy right>left and top>bottom, got {bounds}"
        )

    width  = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"pixel_iou: derived grid ({height} x {width}) is degenerate for "
            f"bounds={bounds} resolution={resolution}"
        )

    expected_shape = (height, width) if isinstance(pred_prob, np.ndarray) else None
    arr = _load_pred_mask(pred_prob, expected_shape)

    if arr.dtype.kind == "f":
        pred_bin = arr >= threshold
    else:
        pred_bin = arr.astype(bool)

    # Handle grid-shape mismatch when a Path was given by resampling on
    # read is out of scope; we assert instead so any caller mismatch
    # surfaces immediately.
    if arr.shape != (height, width):
        raise ValueError(
            f"pixel_iou: pred_prob shape {arr.shape} != derived grid ({height}, {width})"
        )

    transform = _rio_transform_from_bounds(left, bottom, right, top, width, height)
    if len(ref_gdf) == 0:
        ref_bin = np.zeros((height, width), dtype=bool)
    else:
        shapes = [(g, 1) for g in ref_gdf.geometry if g is not None and not g.is_empty]
        if not shapes:
            ref_bin = np.zeros((height, width), dtype=bool)
        else:
            ref_bin = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0, all_touched=all_touched, dtype="uint8",
            ).astype(bool)

    inter = int(np.logical_and(pred_bin, ref_bin).sum())
    union = int(np.logical_or(pred_bin, ref_bin).sum())
    if union == 0:
        return math.nan
    return inter / union
