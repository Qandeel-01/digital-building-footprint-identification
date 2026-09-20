"""Boundary distance metrics: symmetric Chamfer and symmetric Hausdorff.

Implemented in Phase 4 against the actual MatchResult structure landed
in Phase 4 (see bfbench.metrics.instance).

Sampling: every polygon boundary is densified at `sample_step_m`
intervals BEFORE the point-set metric is computed. Vertex-only
sampling is deliberately not supported -- see
docs/metric-decisions.md section 6.

Unmatched handling: the mean is taken over MATCHED pairs only.
Unmatched predictions / references are excluded rather than treated
as +infinity: an "average boundary distance" that includes unmatched
polygons becomes dominated by the tile size and stops measuring
boundary quality. This is consistent with match-level analysis
(precision / recall already capture the unmatched behaviour).

Hausdorff variant: SYMMETRIC = max(directed(A,B), directed(B,A)).
This is the definitional Hausdorff distance; the directed variant is
useful for asymmetric error analysis but not for a headline metric.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry.base import BaseGeometry

from bfbench.metrics.instance import MatchResult


def _densify_polygon(geom: BaseGeometry, step_m: float) -> np.ndarray:
    """Return an (N, 2) array of points sampled along the polygon's
    boundary at approximately `step_m` intervals.

    Handles Polygon and MultiPolygon. Interior rings (holes) contribute
    points too -- they are legitimate boundaries.
    """
    if geom is None or geom.is_empty:
        return np.empty((0, 2), dtype=float)
    if geom.geom_type == "MultiPolygon":
        pieces = list(geom.geoms)
    else:
        pieces = [geom]
    out: list[np.ndarray] = []
    for poly in pieces:
        rings = [poly.exterior]
        if poly.interiors:
            rings += list(poly.interiors)
        for ring in rings:
            if ring is None or ring.is_empty:
                continue
            length = ring.length
            if length <= 0:
                continue
            n = max(4, int(math.ceil(length / step_m)))
            distances = np.linspace(0.0, length, n, endpoint=False)
            pts = [ring.interpolate(d).coords[0] for d in distances]
            out.append(np.asarray(pts, dtype=float))
    if not out:
        return np.empty((0, 2), dtype=float)
    return np.concatenate(out, axis=0)


def _nearest_neighbour_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of `a`, the Euclidean distance to the nearest row
    of `b`. Uses scipy.spatial.cKDTree when available, else falls back
    to a chunked brute-force computation.
    """
    if a.size == 0 or b.size == 0:
        return np.array([], dtype=float)
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415
        tree = cKDTree(b)
        d, _ = tree.query(a, k=1)
        return np.asarray(d, dtype=float)
    except ImportError:  # pragma: no cover -- fallback
        # Chunk to keep memory bounded: dist matrix would be |a| x |b|.
        out = np.empty(len(a), dtype=float)
        chunk = 256
        for i in range(0, len(a), chunk):
            aa = a[i:i + chunk][:, None, :]  # (chunk, 1, 2)
            diff = aa - b[None, :, :]        # (chunk, |b|, 2)
            d = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)
            out[i:i + chunk] = d
        return out


def boundary_distance(match_result: MatchResult,
                      sample_step_m: float = 0.5,
                      pred_gdf=None,
                      ref_gdf=None) -> dict[str, float]:
    """Mean symmetric Chamfer + mean symmetric Hausdorff across matched pairs.

    Args:
        match_result: from `match_instances`.
        sample_step_m: densification step in metres along polygon
            boundaries. Default 0.5 m.
        pred_gdf, ref_gdf: the equal-area GeoDataFrames the match
            was computed on. Required so this function can look up
            the actual geometries. If either is None the function
            returns NaN + a diagnostic note.

    Returns:
        {"mean_chamfer": ..., "mean_hausdorff": ..., "n_pairs": ...}
        Distances in metres. Both mean values are NaN when there are
        no matched pairs.
    """
    if sample_step_m <= 0:
        raise ValueError(f"sample_step_m must be positive, got {sample_step_m}")
    if pred_gdf is None or ref_gdf is None:
        return {"mean_chamfer": float("nan"),
                "mean_hausdorff": float("nan"),
                "n_pairs": 0.0}
    if not match_result.matched:
        return {"mean_chamfer": float("nan"),
                "mean_hausdorff": float("nan"),
                "n_pairs": 0.0}

    chamfers: list[float] = []
    hausdorffs: list[float] = []
    for pi, ri, _iou in match_result.matched:
        pg = pred_gdf.geometry.iloc[pi]
        rg = ref_gdf.geometry.iloc[ri]
        p_pts = _densify_polygon(pg, sample_step_m)
        r_pts = _densify_polygon(rg, sample_step_m)
        if p_pts.size == 0 or r_pts.size == 0:
            continue
        d_pr = _nearest_neighbour_distances(p_pts, r_pts)
        d_rp = _nearest_neighbour_distances(r_pts, p_pts)
        # Symmetric Chamfer = mean of the two directed means.
        chamfer = 0.5 * (float(d_pr.mean()) + float(d_rp.mean()))
        # Symmetric Hausdorff = max of the two directed maxima.
        hausdorff = max(float(d_pr.max()), float(d_rp.max()))
        chamfers.append(chamfer)
        hausdorffs.append(hausdorff)

    if not chamfers:
        return {"mean_chamfer": float("nan"),
                "mean_hausdorff": float("nan"),
                "n_pairs": 0.0}
    return {
        "mean_chamfer":   float(np.mean(chamfers)),
        "mean_hausdorff": float(np.mean(hausdorffs)),
        "n_pairs":        float(len(chamfers)),
    }
