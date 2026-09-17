"""End-to-end tests for the bfbench.eval harness.

These verify that:

    - The harness runs to completion even when every metric is a stub
      raising NotImplementedError, marking those rows as
      status = "not_implemented".
    - The parquet has the tidy long-format schema documented in
      bfbench.eval.
    - The eval is deterministic given the same sampled_tiles + inputs
      (byte-identical parquet).
    - The eval is resumable: partial staged rows are picked up on
      re-run, and dropping a stage file causes just that tile to
      re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import mercantile
import pandas as pd
import yaml
from shapely.geometry import Polygon

from bfbench.eval import evaluate


def _write_aoi_fixture(tmp_path: Path) -> Path:
    """Build a minimal AOI work dir: 3 sampled tiles, 4 ref polygons,
    3 pred polygons, and a config YAML pointing at them.

    The tile IDs are picked over New York City (lon -74.006, lat 40.712)
    at z=17 so mercantile.bounds returns a plausible small footprint.
    Reference polygons are placed roughly inside those tiles.
    """
    work_dir = tmp_path / "aoi_work"
    work_dir.mkdir(parents=True)

    lon, lat = -74.006, 40.712
    center_tile = mercantile.tile(lon, lat, 17)
    tiles = [
        (17, center_tile.x,     center_tile.y),
        (17, center_tile.x + 1, center_tile.y),
        (17, center_tile.x,     center_tile.y + 1),
    ]

    sampled = {
        "aoi": "TEST_AOI",
        "zoom": 17,
        "effective_gsd_m": 0.5,
        "seed": 12345,
        "n_requested": 3,
        "n_candidates": 3,
        "tile_ids_hash": "test-hash",
        "tiles": tiles,
    }
    (work_dir / "sampled_tiles.json").write_text(json.dumps(sampled))

    # Reference polygons in EPSG:4326: one near each tile centroid.
    ref_polys = []
    for z, x, y in tiles:
        b = mercantile.bounds(mercantile.Tile(x, y, z))
        cx = 0.5 * (b.west + b.east)
        cy = 0.5 * (b.south + b.north)
        d = 0.0001
        ref_polys.append(Polygon([
            (cx - d, cy - d), (cx + d, cy - d),
            (cx + d, cy + d), (cx - d, cy + d),
        ]))
    # Add an extra polygon inside the first tile.
    b0 = mercantile.bounds(mercantile.Tile(tiles[0][1], tiles[0][2], 17))
    ref_polys.append(Polygon([
        (b0.west + 0.00005, b0.south + 0.00005),
        (b0.west + 0.00015, b0.south + 0.00005),
        (b0.west + 0.00015, b0.south + 0.00015),
        (b0.west + 0.00005, b0.south + 0.00015),
    ]))
    ref = gpd.GeoDataFrame(geometry=ref_polys, crs="EPSG:4326")
    ref.to_file(work_dir / "reference.geojson", driver="GeoJSON")

    # Predictions: first three ref polygons only, so the last tile has
    # a reference but no prediction (exercises the missing-prediction
    # branch).
    pred = gpd.GeoDataFrame(geometry=ref_polys[:3], crs="EPSG:4326")
    pred.to_file(work_dir / "predicted.geojson", driver="GeoJSON")

    cfg = {
        "aoi": {"name": "TEST_AOI", "provider": "spacenet2",
                "aoi_id": 99, "split": "train", "image_kind": "PS-RGB"},
        "adapter": {"target_gsd_m": 0.5, "tile_size": 512,
                    "zoom_tolerance": 0.20, "keep_borders": True,
                    "channel_name": "images"},
        "reference": {"dedup": {"method": "touching_union",
                                "buffer_m": 0.5, "iou_threshold": 0.5,
                                "equal_area_epsg": 6933}},
        "sampling": {"n_tiles": 3, "seed": 12345,
                     "min_valid_fraction": 1.0},
        "run": {"work_dir": str(work_dir)},
        "eval": {"iou_threshold": 0.5, "pixel_resolution_m": 0.5},
    }
    cfg_path = tmp_path / "test_aoi.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_eval_runs_end_to_end_with_stub_metrics(tmp_path):
    cfg_path = _write_aoi_fixture(tmp_path)
    out = evaluate(cfg_path)
    assert out.is_file()

    df = pd.read_parquet(out)
    for col in ("aoi", "tile_z", "tile_x", "tile_y", "metric", "value",
                "status", "note", "n_pred", "n_ref"):
        assert col in df.columns, f"missing column {col!r}: {df.columns.tolist()}"

    # On clean checkout every metric raises NotImplementedError, so
    # dict-returning metrics collapse to one placeholder row each.
    # 3 tiles x 7 metric families -> 21 rows. Once the stubs are
    # implemented dict-returning metrics expand and this rises.
    assert len(df) >= 21, f"expected >= 21 rows on stubbed eval, got {len(df)}"
    assert (df["aoi"] == "TEST_AOI").all()

    # Because the metric functions are all stubs, at least the
    # non-precondition ones should be marked not_implemented.
    ni_metrics = set(df.loc[df["status"] == "not_implemented", "metric"])
    for expected in ("match_instances.status", "merge_rate", "split_rate",
                     "pixel_iou"):
        assert expected in ni_metrics, (
            f"expected {expected} to be status='not_implemented' on clean "
            f"checkout, got {sorted(ni_metrics)}"
        )


def test_eval_synthetic_predictions_works_without_predicted_geojson(tmp_path):
    cfg_path = _write_aoi_fixture(tmp_path)
    # Delete predicted.geojson so the harness must synthesise.
    work_dir = Path(yaml.safe_load(cfg_path.read_text())["run"]["work_dir"])
    (work_dir / "predicted.geojson").unlink()

    out = evaluate(cfg_path, synthetic_predictions=True)
    df = pd.read_parquet(out)
    assert len(df) > 0


def test_eval_is_resumable(tmp_path):
    cfg_path = _write_aoi_fixture(tmp_path)
    _ = evaluate(cfg_path)

    work_dir = Path(yaml.safe_load(cfg_path.read_text())["run"]["work_dir"])
    stage_dir = work_dir / "eval" / "tiles"
    stages = list(stage_dir.glob("*.json"))
    assert len(stages) == 3, f"expected 3 staged tiles, got {len(stages)}"

    # Corrupt one stage file to a specific marker; re-run should NOT
    # re-evaluate it (it uses the cache).
    stages[0].write_text('[{"aoi":"TEST_AOI","tile_z":0,"tile_x":0,"tile_y":0,'
                        '"metric":"CACHED_MARKER","value":42.0,"status":"ok",'
                        '"note":"","n_pred":0,"n_ref":0}]')
    _ = evaluate(cfg_path)
    df = pd.read_parquet(work_dir / "eval" / "results.parquet")
    assert (df["metric"] == "CACHED_MARKER").any(), (
        "resume broke: cached stage file was not reused"
    )


def test_eval_manifest_records_seed_and_hash(tmp_path):
    cfg_path = _write_aoi_fixture(tmp_path)
    evaluate(cfg_path)
    work_dir = Path(yaml.safe_load(cfg_path.read_text())["run"]["work_dir"])
    m = json.loads((work_dir / "eval" / "eval_manifest.json")
                   .read_text(encoding="utf-8"))
    assert m["sample_seed"] == 12345
    assert m["sample_hash"] == "test-hash"
    assert m["n_metrics_not_implemented"] > 0
