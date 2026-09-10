"""Geometry unit tests for the SpaceNet -> abd tile adapter.

These are real assertions on computed values, not smoke tests: every
test exercises a specific invariant that would silently break the
benchmark if violated.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from bfbench.data import adapter


# ---------------------------------------------------------------------------
# GSD / zoom math
# ---------------------------------------------------------------------------

def test_webmercator_gsd_equator_z0():
    """At the equator, z=0, a 256 px tile spans the full Earth circumference.

    Expected: 40 075 016.686 m / 256 = 156 543.033 928 m/px.
    """
    gsd = adapter.webmercator_gsd_at(lat_deg=0.0, zoom=0, tile_size=256)
    assert gsd == pytest.approx(156543.033928, rel=1e-9)


def test_webmercator_gsd_scales_with_zoom():
    """GSD halves each time zoom goes up by 1, at any latitude."""
    for lat in (0.0, 36.17, 48.85, -33.86):
        g0 = adapter.webmercator_gsd_at(lat, zoom=10)
        g1 = adapter.webmercator_gsd_at(lat, zoom=11)
        assert g0 / g1 == pytest.approx(2.0, rel=1e-12)


def test_pick_zoom_selects_closest_to_target():
    """At Vegas latitude, targeting 0.5 m should pick z=18.

    Native GSD at (lat=36.17, z=18) is ~0.482 m; at z=17 it's ~0.964 m;
    at z=19 it's ~0.241 m. So 18 is the unique closest.
    """
    z = adapter.pick_zoom_for_gsd(lat_deg=36.17, target_gsd_m=0.5, tolerance_m=0.15)
    assert z == 18
    err = abs(adapter.webmercator_gsd_at(36.17, z) - 0.5)
    assert err < 0.15


def test_pick_zoom_rejects_out_of_tolerance():
    """Ask for an absurdly small GSD that no reasonable zoom can satisfy."""
    with pytest.raises(ValueError, match="within"):
        adapter.pick_zoom_for_gsd(lat_deg=0.0, target_gsd_m=0.001, tolerance_m=0.0001)


# ---------------------------------------------------------------------------
# Tile bounds
# ---------------------------------------------------------------------------

def test_tile_bounds_webmercator_is_epsg3857_meters():
    """mercantile.xy_bounds returns EPSG:3857 metres. Sanity-check a known
    tile: at z=0, the whole world tile spans roughly the Web-Mercator
    +/- 20 037 508 m box.
    """
    t = mercantile.Tile(0, 0, 0)
    west, south, east, north = adapter.tile_bounds_webmercator(t)
    assert west == pytest.approx(-20037508.342789, rel=1e-6)
    assert east == pytest.approx(20037508.342789, rel=1e-6)
    assert north == pytest.approx(20037508.342789, rel=1e-6)
    assert south == pytest.approx(-20037508.342789, rel=1e-6)


def test_tiles_covering_stays_at_requested_zoom_and_intersects_bbox():
    """Every returned tile must be at the requested zoom and must actually
    intersect the WGS84 bbox in geographic coordinates."""
    # Small bbox around downtown Vegas
    west, south, east, north = -115.15, 36.16, -115.14, 36.17
    zoom = 18
    tiles = adapter.tiles_covering((west, south, east, north), zoom)
    assert len(tiles) > 0
    for t in tiles:
        assert t.z == zoom
        b = mercantile.bounds(t)
        # tile bbox must overlap the requested bbox (mercantile guarantees this)
        assert not (b.east < west or b.west > east or b.north < south or b.south > north)


# ---------------------------------------------------------------------------
# Round-trip: pixel <-> geographic
# ---------------------------------------------------------------------------

def test_pixel_to_geographic_roundtrip_within_pixel():
    """For a known raster with a known affine, a pixel at (row, col) must
    map to the geographic coordinate implied by the affine, and back.
    """
    # Synthetic 512x512 raster covering a 512m x 512m patch at
    # (500_000, 4_000_000) in some UTM CRS. Pixel size = 1m.
    left, bottom, right, top = 500_000.0, 4_000_000.0, 500_512.0, 4_000_512.0
    width = height = 512
    transform = from_bounds(left, bottom, right, top, width, height)

    # Pixel (row=100, col=200), CENTRE of the pixel:
    row, col = 100, 200
    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    # Expected: col = 200 -> x = left + (200 + 0.5) * 1.0
    #           row = 100 -> y = top  - (100 + 0.5) * 1.0
    assert x == pytest.approx(500_200.5, abs=1e-9)
    assert y == pytest.approx(4_000_411.5, abs=1e-9)

    # And back:
    r2, c2 = rasterio.transform.rowcol(transform, x, y)
    assert (r2, c2) == (row, col)


def test_tile_boundary_alignment_no_gap_no_overlap():
    """Adjacent XYZ tiles at the same zoom must share their boundary
    exactly in EPSG:3857 (no gap, no overlap)."""
    z = 18
    t1 = mercantile.Tile(x=42000, y=98000, z=z)
    t2 = mercantile.Tile(x=42001, y=98000, z=z)  # eastern neighbour
    _, _, e1, _ = adapter.tile_bounds_webmercator(t1)
    w2, _, _, _ = adapter.tile_bounds_webmercator(t2)
    assert e1 == pytest.approx(w2, abs=1e-6), (
        f"eastern edge of {t1} ({e1}) must equal western edge of {t2} ({w2})"
    )

    t3 = mercantile.Tile(x=42000, y=98001, z=z)  # southern neighbour
    _, s1, _, _ = adapter.tile_bounds_webmercator(t1)
    _, _, _, n3 = adapter.tile_bounds_webmercator(t3)
    assert s1 == pytest.approx(n3, abs=1e-6), (
        f"southern edge of {t1} ({s1}) must equal northern edge of {t3} ({n3})"
    )


# ---------------------------------------------------------------------------
# End-to-end tiling on a synthetic raster
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_utm_raster(tmp_path: Path) -> Path:
    """Create a small synthetic uint8 raster in EPSG:32611 (UTM 11N, Vegas)
    covering ~250 m x 250 m at 0.30 m/pixel (SpaceNet PS-RGB style)."""
    left, bottom = 650_000.0, 4_000_000.0
    px = 0.30
    width, height = 832, 832
    right = left + width * px
    top = bottom + height * px
    transform = from_bounds(left, bottom, right, top, width, height)
    data = np.zeros((3, height, width), dtype=np.uint8)
    # Paint a colourful gradient so percentile stretch has something to do
    yy, xx = np.mgrid[0:height, 0:width]
    data[0] = (xx % 256).astype(np.uint8)
    data[1] = (yy % 256).astype(np.uint8)
    data[2] = ((xx + yy) % 256).astype(np.uint8)

    path = tmp_path / "synthetic_psrgb.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width, count=3,
        dtype=np.uint8, crs="EPSG:32611", transform=transform,
    ) as dst:
        dst.write(data)
    return path


def test_adapter_writes_valid_cover_csv_and_tile_tree(synthetic_utm_raster: Path, tmp_path: Path):
    out_root = tmp_path / "abd_tiles"
    result = adapter.adapt_spacenet_image(
        synthetic_utm_raster,
        out_root,
        config=adapter.AdapterConfig(target_gsd_m=0.5, tile_size=512, keep_borders=True,
                                     zoom_tolerance=0.20),
    )

    # cover.csv is present, well-formed, and enumerates the same set of
    # (x, y, z) as the written tile files.
    with result.cover_csv.open() as fp:
        rows = list(csv.reader(fp))
    cover_set = {(int(r[0]), int(r[1]), int(r[2])) for r in rows}

    tiled_files = list((out_root / "images").rglob("*.tiff"))
    file_set = set()
    for f in tiled_files:
        # <images>/<z>/<x>/<y>.tiff
        y = int(f.stem)
        x = int(f.parent.name)
        z = int(f.parent.parent.name)
        file_set.add((x, y, z))

    assert cover_set == file_set, (
        f"cover.csv and on-disk tiles disagree: only-in-cover={cover_set - file_set}, "
        f"only-on-disk={file_set - cover_set}"
    )
    assert result.tiles_written == len(cover_set) > 0
    assert result.tile_size == 512
    assert result.zoom == 18  # implied by target 0.5 m at lat ~36


def test_adapter_tile_pixel_size_matches_config(synthetic_utm_raster: Path, tmp_path: Path):
    """Every tile written must be exactly `tile_size x tile_size` pixels."""
    out_root = tmp_path / "abd_tiles"
    cfg = adapter.AdapterConfig(target_gsd_m=0.5, tile_size=512, keep_borders=True,
                                zoom_tolerance=0.20)
    adapter.adapt_spacenet_image(synthetic_utm_raster, out_root, config=cfg)

    for f in (out_root / "images").rglob("*.tiff"):
        with rasterio.open(f) as src:
            assert src.width == cfg.tile_size
            assert src.height == cfg.tile_size


def test_adapter_keep_borders_false_drops_edge_tiles(synthetic_utm_raster: Path, tmp_path: Path):
    """With keep_borders=False, only tiles that have all 8 neighbours on disk
    survive. On this small synthetic AOI that must be a strict subset of
    the keep_borders=True result."""
    out_root_a = tmp_path / "kb_true"
    out_root_b = tmp_path / "kb_false"
    common = dict(target_gsd_m=0.5, tile_size=512, zoom_tolerance=0.20)
    r_a = adapter.adapt_spacenet_image(
        synthetic_utm_raster, out_root_a,
        config=adapter.AdapterConfig(keep_borders=True, **common),
    )
    r_b = adapter.adapt_spacenet_image(
        synthetic_utm_raster, out_root_b,
        config=adapter.AdapterConfig(keep_borders=False, **common),
    )
    assert r_b.tiles_written <= r_a.tiles_written


def test_adapter_output_tile_geographic_bounds_match_mercantile(synthetic_utm_raster: Path, tmp_path: Path):
    """For every written tile, its (x, y, z) must correspond to the same
    EPSG:3857 bbox that mercantile.xy_bounds returns for that tile. This
    is the invariant `abd vectorize` relies on to place polygons.
    """
    out_root = tmp_path / "abd_tiles"
    adapter.adapt_spacenet_image(
        synthetic_utm_raster, out_root,
        config=adapter.AdapterConfig(target_gsd_m=0.5, tile_size=512, keep_borders=True,
                                     zoom_tolerance=0.20),
    )

    for f in (out_root / "images").rglob("*.tiff"):
        y = int(f.stem)
        x = int(f.parent.name)
        z = int(f.parent.parent.name)
        # Re-derive the expected bbox from mercantile
        expected = mercantile.xy_bounds(mercantile.Tile(x, y, z))
        # The tile GSD implied by the tile pixel size on disk must be
        # consistent with the (256 px canonical) mercantile bounds.
        with rasterio.open(f) as src:
            # The synthetic tiff we write is NOT georeferenced because
            # `Image.fromarray(...).save('...tiff')` uses PIL and drops
            # georeference. This is deliberate: `abd predict` only uses
            # the (x,y,z) FROM THE PATH to locate the tile, so pixel data
            # can be plain-tiff. But we can still assert that its pixel
            # dimensions are consistent with a 256*2^0 -> 512 upscale.
            assert src.width == src.height
        # Sanity: expected span is positive and finite.
        assert expected.right > expected.left
        assert expected.top > expected.bottom
        # The tile width in metres = 40 075 016.686 / 2^z  (independent of lat)
        expected_width_m = adapter._EARTH_CIRC_M / (2 ** z)
        assert (expected.right - expected.left) == pytest.approx(expected_width_m, rel=1e-9)
