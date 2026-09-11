"""Merge and deduplicate SpaceNet reference polygons across chips.

Each SpaceNet chip carries its own `geojson_buildings/*.geojson`
containing building polygons *clipped to that chip's footprint*.
A building that straddles a chip seam therefore appears in BOTH
adjacent chips, each as a partial polygon whose new edge lies exactly
on the seam. If we just concatenate every chip's geojson we double-count
those seam buildings, and per-tile metrics get corrupted near seams.

This module concatenates all chip polygons for one AOI, reprojects to
EPSG:4326, and applies a configurable dedup rule to reunite seam
partials into single features.

Dedup rules (`DedupRule.method`):

    "touching_union" (default)
        Buffer every polygon by `buffer_m` metres in an equal-area CRS,
        take unary_union, split back into connected components,
        de-buffer to keep the join area small. Two seam partials whose
        shared edge is exactly on the chip boundary have TOUCHING
        boundaries; buffer by e.g. 0.5 m makes them merge; unary_union
        collapses them into one polygon. Buildings that just happen to
        be within `buffer_m` of each other get merged too, so the
        buffer must be small. Empirically 0.5 m works for SpaceNet at
        ~30 cm GSD.

    "iou_threshold"
        For every pair with IoU >= threshold, merge via unary_union.
        Fails on seam partials because their intersection is a
        one-dimensional line (IoU = 0). Included as a strict-match
        alternative for callers who want NO merging of nearby-but-
        separate buildings, and who accept the double-counting.

    "identity"
        Only exact-WKB duplicates removed. This is the null rule and
        will NOT deduplicate seam partials. Useful as a control.

The chosen rule + parameters are recorded in the run manifest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape as _shape  # noqa: F401 -- future geojson streaming
from shapely.ops import unary_union

from bfbench.data.equal_area import DEFAULT_EQUAL_AREA_EPSG, assert_equal_area_crs

_LOG = logging.getLogger(__name__)

DedupMethod = Literal["touching_union", "iou_threshold", "identity"]


@dataclass(frozen=True)
class DedupRule:
    method: DedupMethod = "touching_union"
    buffer_m: float = 0.5           # only used for touching_union
    iou_threshold: float = 0.5      # only used for iou_threshold
    equal_area_epsg: int = DEFAULT_EQUAL_AREA_EPSG


@dataclass(frozen=True)
class DedupResult:
    gdf: gpd.GeoDataFrame
    n_input: int
    n_output: int
    n_duplicates_removed: int
    method: str


def merge_reference(chip_geojsons: list[Path],
                    rule: DedupRule | None = None,
                    keep_crs: str = "EPSG:4326") -> DedupResult:
    """Concat every chip's building geojson, dedup on seams, return one
    GeoDataFrame in `keep_crs`."""
    rule = rule or DedupRule()
    frames: list[gpd.GeoDataFrame] = []
    for p in chip_geojsons:
        g = gpd.read_file(p)
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        frames.append(g[["geometry"]] if "geometry" in g.columns else g)
    if not frames:
        empty = gpd.GeoDataFrame(geometry=[], crs=keep_crs)
        return DedupResult(empty, 0, 0, 0, rule.method)

    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs).to_crs(keep_crs)
    merged = merged[~merged.geometry.is_empty & merged.geometry.notna()].reset_index(drop=True)
    n_in = len(merged)

    if rule.method == "identity":
        out = _dedup_identity(merged)
    elif rule.method == "touching_union":
        out = _dedup_touching_union(merged, rule.buffer_m, rule.equal_area_epsg)
    elif rule.method == "iou_threshold":
        out = _dedup_iou_threshold(merged, rule.iou_threshold, rule.equal_area_epsg)
    else:  # pragma: no cover -- Literal enforces this
        raise ValueError(f"unknown dedup method: {rule.method!r}")

    out = out.to_crs(keep_crs)
    n_out = len(out)
    _LOG.info(
        "dedup(%s): %d -> %d (%d removed)",
        rule.method, n_in, n_out, n_in - n_out,
    )
    return DedupResult(out, n_in, n_out, n_in - n_out, rule.method)


def _dedup_identity(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    seen: set[bytes] = set()
    keep_idx: list[int] = []
    for i, geom in enumerate(gdf.geometry):
        w = geom.wkb
        if w in seen:
            continue
        seen.add(w)
        keep_idx.append(i)
    return gdf.iloc[keep_idx].reset_index(drop=True)


def _dedup_touching_union(gdf: gpd.GeoDataFrame, buffer_m: float,
                          equal_area_epsg: int) -> gpd.GeoDataFrame:
    # Buffer in equal-area units so the buffer distance is in metres and
    # meaningful across latitudes.
    ea_crs = f"EPSG:{equal_area_epsg}"
    assert_equal_area_crs(ea_crs, what="reference dedup buffer")
    ea = gdf.to_crs(ea_crs)
    buffed = ea.geometry.buffer(buffer_m)
    unioned = unary_union(list(buffed))
    if unioned.is_empty:
        return gpd.GeoDataFrame(geometry=[], crs=ea_crs)

    if unioned.geom_type == "Polygon":
        pieces = [unioned]
    else:
        pieces = list(unioned.geoms)

    # De-buffer to restore roughly the original outline.
    debuffered = [p.buffer(-buffer_m) for p in pieces]
    debuffered = [g for g in debuffered if (not g.is_empty) and g.is_valid]
    return gpd.GeoDataFrame(geometry=debuffered, crs=ea_crs)


def _dedup_iou_threshold(gdf: gpd.GeoDataFrame, threshold: float,
                         equal_area_epsg: int) -> gpd.GeoDataFrame:
    ea_crs = f"EPSG:{equal_area_epsg}"
    assert_equal_area_crs(ea_crs, what="reference dedup IoU")
    ea = gdf.to_crs(ea_crs)
    geoms = list(ea.geometry)
    n = len(geoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Use spatial index to avoid O(n^2)
    sindex = ea.sindex
    for i in range(n):
        cand = list(sindex.intersection(geoms[i].bounds))
        for j in cand:
            if j <= i:
                continue
            gi, gj = geoms[i], geoms[j]
            inter = gi.intersection(gj).area
            if inter == 0:
                continue
            uni = gi.union(gj).area
            if uni == 0:
                continue
            if inter / uni >= threshold:
                union(i, j)

    # Group by root, union each group.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    outs = [unary_union([geoms[k] for k in idxs]) for idxs in groups.values()]
    return gpd.GeoDataFrame(geometry=outs, crs=ea_crs)
