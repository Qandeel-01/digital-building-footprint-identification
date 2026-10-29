"""Tests for reference-polygon dedup across chip seams.

Every fixture is synthetic. Each test asserts a specific dedup
invariant with a hand-computed expected value.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from bfbench.data.reference import DedupRule, merge_reference


def _chip_geojson(tmp_path: Path, name: str, polys: list[Polygon]) -> Path:
    gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
    p = tmp_path / name
    gdf.to_file(p, driver="GeoJSON")
    return p


def test_touching_union_merges_seam_partials(tmp_path: Path):
    """Two chip files, each with one polygon whose edge lies exactly on
    the shared chip seam at x = 0.001 (roughly ~90 m at the equator).
    Expected: touching_union merges them into ONE polygon.
    """
    left_partial = Polygon([(0.000, 0.000), (0.001, 0.000),
                            (0.001, 0.0005), (0.000, 0.0005)])
    right_partial = Polygon([(0.001, 0.000), (0.002, 0.000),
                             (0.002, 0.0005), (0.001, 0.0005)])
    p1 = _chip_geojson(tmp_path, "chipA.geojson", [left_partial])
    p2 = _chip_geojson(tmp_path, "chipB.geojson", [right_partial])

    res = merge_reference(
        [p1, p2],
        rule=DedupRule(method="touching_union", buffer_m=0.5),
        keep_crs="EPSG:4326",
    )
    assert res.n_input == 2
    assert res.n_output == 1, (
        f"expected 1 merged polygon, got {res.n_output}: {list(res.gdf.geometry)}"
    )
    assert res.n_duplicates_removed == 1


def test_touching_union_leaves_distant_polys_alone(tmp_path: Path):
    """Two polygons 100 m apart in EPSG:6933 (well beyond a 0.5 m
    buffer) must not be merged.
    """
    # 0.001 degrees longitude at the equator ~= 111 m
    a = Polygon([(0.000, 0.000), (0.0005, 0.000), (0.0005, 0.0005), (0.000, 0.0005)])
    b = Polygon([(0.010, 0.000), (0.0105, 0.000), (0.0105, 0.0005), (0.010, 0.0005)])
    p = _chip_geojson(tmp_path, "chip.geojson", [a, b])

    res = merge_reference(
        [p],
        rule=DedupRule(method="touching_union", buffer_m=0.5),
        keep_crs="EPSG:4326",
    )
    assert res.n_output == 2, (
        f"distant polys must not merge; got {res.n_output}"
    )
    assert res.n_duplicates_removed == 0


def test_identity_only_removes_exact_duplicates(tmp_path: Path):
    """A polygon that appears in two chip files with IDENTICAL geometry
    must be dedup-ed by the identity rule; a slightly-different polygon
    must be kept.
    """
    exact = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    slight = Polygon([(0, 0), (1.0001, 0), (1, 1), (0, 1)])
    p1 = _chip_geojson(tmp_path, "chipA.geojson", [exact])
    p2 = _chip_geojson(tmp_path, "chipB.geojson", [exact, slight])

    res = merge_reference(
        [p1, p2],
        rule=DedupRule(method="identity"),
        keep_crs="EPSG:4326",
    )
    # Input has 3 features; two are exact duplicates -> 2 unique.
    assert res.n_input == 3
    assert res.n_output == 2


def test_iou_threshold_does_not_merge_seam_partials(tmp_path: Path):
    """iou_threshold merges polygons whose IoU >= threshold. Seam
    partials with a one-dimensional intersection have IoU = 0, so this
    rule must leave them alone (documented behaviour, not a bug).
    """
    left_partial = Polygon([(0.000, 0.000), (0.001, 0.000),
                            (0.001, 0.0005), (0.000, 0.0005)])
    right_partial = Polygon([(0.001, 0.000), (0.002, 0.000),
                             (0.002, 0.0005), (0.001, 0.0005)])
    p1 = _chip_geojson(tmp_path, "chipA.geojson", [left_partial])
    p2 = _chip_geojson(tmp_path, "chipB.geojson", [right_partial])

    res = merge_reference(
        [p1, p2],
        rule=DedupRule(method="iou_threshold", iou_threshold=0.5),
        keep_crs="EPSG:4326",
    )
    # Note: shared boundary has IoU=0 -> partials NOT merged.
    assert res.n_output == 2


def test_empty_inputs_return_empty(tmp_path: Path):
    res = merge_reference([], rule=DedupRule(), keep_crs="EPSG:4326")
    assert res.n_input == 0
    assert res.n_output == 0
    assert res.gdf.empty


def test_dedup_uses_equal_area_crs_for_buffer(tmp_path: Path):
    """The touching_union buffer is documented as being in metres. Two
    polygons whose lon-lat distance is small in *degrees* but LARGE in
    metres (near the equator) must not be merged with a 0.5 m buffer.

    ~1 degree of longitude at the equator = ~111 km. A gap of 0.001 deg
    is ~111 m. With buffer_m=0.5 the polygons stay separate.
    """
    a = Polygon([(0.0, 0.0), (0.0001, 0.0), (0.0001, 0.0001), (0.0, 0.0001)])
    b = Polygon([(0.001, 0.0), (0.0011, 0.0), (0.0011, 0.0001), (0.001, 0.0001)])
    p = _chip_geojson(tmp_path, "chip.geojson", [a, b])

    res = merge_reference(
        [p], rule=DedupRule(method="touching_union", buffer_m=0.5),
        keep_crs="EPSG:4326",
    )
    assert res.n_output == 2
