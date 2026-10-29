"""End-to-end test for scripts/compare_baseline_vs_finetuned.py.

Builds a toy pair of eval parquets with known per-tile metric values,
runs the comparison, and asserts the point estimates and shared-tile
intersection are what we expect. Does NOT run bfbench eval.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compare_baseline_vs_finetuned.py"


@pytest.fixture(scope="module")
def compare_module():
    spec = importlib.util.spec_from_file_location("compare_bf", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # dataclasses look up `cls.__module__` in sys.modules; register
    # before exec so @dataclass can resolve forward refs.
    sys.modules["compare_bf"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("compare_bf", None)
        raise
    return mod


def _mk_parquet(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _mk_full_parquet(path: Path, prec: float, rec: float, f1: float,
                     merge: float, split: float,
                     pixel_iou: float, chamfer: float,
                     tiles: list[tuple[int, int, int]]) -> None:
    """Build a parquet with one row per (tile, metric) at the given values."""
    rows = []
    for z, x, y in tiles:
        base = {"aoi": "AOI_UNIT",
                "tile_z": z, "tile_x": x, "tile_y": y,
                "n_pred": 1, "n_ref": 1}
        for metric, val in (
            ("instance_prf.precision", prec),
            ("instance_prf.recall",    rec),
            ("instance_prf.f1",        f1),
            ("merge_rate",             merge),
            ("split_rate",             split),
            ("pixel_iou",              pixel_iou),
            ("boundary_distance.mean_m", chamfer),
        ):
            rows.append({**base,
                        "metric": metric,
                        "value":  val,
                        "status": "ok",
                        "note":   ""})
    _mk_parquet(path, rows)


def test_tiles_in_both_intersection(compare_module, tmp_path: Path) -> None:
    """Only tiles present in BOTH parquets are compared."""
    _mk_full_parquet(tmp_path / "b.parquet",
                     prec=0.5, rec=0.5, f1=0.5, merge=0.1, split=0.0,
                     pixel_iou=0.5, chamfer=2.0,
                     tiles=[(17, 1, 1), (17, 2, 2), (17, 3, 3)])
    _mk_full_parquet(tmp_path / "f.parquet",
                     prec=0.8, rec=0.8, f1=0.8, merge=0.05, split=0.0,
                     pixel_iou=0.7, chamfer=1.0,
                     tiles=[(17, 2, 2), (17, 3, 3), (17, 4, 4)])
    out = compare_module.compare(
        tmp_path / "b.parquet", tmp_path / "f.parquet",
        aoi_name="AOI_UNIT", out_dir=tmp_path,
        n_boot=50, seed=0,
    )
    assert out["n_shared_tiles"] == 2   # (17,2,2) and (17,3,3)
    assert (tmp_path / "comparison_AOI_UNIT.json").is_file()
    assert (tmp_path / "comparison_AOI_UNIT.md").is_file()


def test_point_estimates_match_the_fixed_values(compare_module, tmp_path: Path) -> None:
    """Fine-tuned parquet has F1=0.8 on every tile; the point estimate
    (mean across tiles) must be 0.8 modulo tiny bootstrap noise."""
    tiles = [(17, i, i) for i in range(20)]
    _mk_full_parquet(tmp_path / "b.parquet",
                     prec=0.5, rec=0.5, f1=0.5, merge=0.1, split=0.05,
                     pixel_iou=0.4, chamfer=3.0, tiles=tiles)
    _mk_full_parquet(tmp_path / "f.parquet",
                     prec=0.8, rec=0.8, f1=0.8, merge=0.05, split=0.0,
                     pixel_iou=0.7, chamfer=1.5, tiles=tiles)
    out = compare_module.compare(
        tmp_path / "b.parquet", tmp_path / "f.parquet",
        aoi_name="AOI_UNIT", out_dir=tmp_path,
        n_boot=200, seed=0,
    )
    rows = {r["metric"]: r for r in out["rows"]}
    assert rows["F1"]["baseline"]["point"]  == pytest.approx(0.5, abs=1e-6)
    assert rows["F1"]["finetuned"]["point"] == pytest.approx(0.8, abs=1e-6)
    assert rows["pixel_IoU"]["baseline"]["point"]  == pytest.approx(0.4, abs=1e-6)
    assert rows["pixel_IoU"]["finetuned"]["point"] == pytest.approx(0.7, abs=1e-6)
    assert rows["Chamfer_m"]["finetuned"]["point"] == pytest.approx(1.5, abs=1e-6)


def test_zero_shared_tiles_raises(compare_module, tmp_path: Path) -> None:
    _mk_full_parquet(tmp_path / "b.parquet",
                     prec=0.5, rec=0.5, f1=0.5, merge=0.1, split=0.0,
                     pixel_iou=0.5, chamfer=2.0, tiles=[(17, 1, 1)])
    _mk_full_parquet(tmp_path / "f.parquet",
                     prec=0.8, rec=0.8, f1=0.8, merge=0.05, split=0.0,
                     pixel_iou=0.7, chamfer=1.0, tiles=[(17, 99, 99)])
    with pytest.raises(ValueError, match="zero tiles"):
        compare_module.compare(
            tmp_path / "b.parquet", tmp_path / "f.parquet",
            aoi_name="AOI_UNIT", out_dir=tmp_path,
            n_boot=10, seed=0,
        )


def test_json_schema(compare_module, tmp_path: Path) -> None:
    tiles = [(17, i, i) for i in range(5)]
    _mk_full_parquet(tmp_path / "b.parquet",
                     prec=0.5, rec=0.5, f1=0.5, merge=0.1, split=0.0,
                     pixel_iou=0.5, chamfer=2.0, tiles=tiles)
    _mk_full_parquet(tmp_path / "f.parquet",
                     prec=0.8, rec=0.8, f1=0.8, merge=0.05, split=0.0,
                     pixel_iou=0.7, chamfer=1.0, tiles=tiles)
    compare_module.compare(
        tmp_path / "b.parquet", tmp_path / "f.parquet",
        aoi_name="AOI_UNIT", out_dir=tmp_path,
        n_boot=10, seed=0,
    )
    data = json.loads((tmp_path / "comparison_AOI_UNIT.json").read_text(encoding="utf-8"))
    assert data["aoi"] == "AOI_UNIT"
    assert data["bootstrap"]["n_boot"] == 10
    assert data["bootstrap"]["seed"] == 0
    for row in data["rows"]:
        for side in ("baseline", "finetuned"):
            for key in ("point", "ci_lo", "ci_hi", "n_tiles"):
                assert key in row[side], (row["metric"], side, key)
