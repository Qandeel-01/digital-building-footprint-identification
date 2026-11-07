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


def _mk_config(path: Path, work_dir: Path, aoi: str,
              output_subdir: str | None,
              with_split: bool) -> Path:
    """Minimal Phase 1 / Phase 2a config stub for resolve_from_config."""
    import yaml  # noqa: PLC0415
    cfg = {
        "aoi": {"name": aoi, "aoi_id": 2, "split": "train",
                "image_kind": "PS-RGB"},
        "adapter": {"target_gsd_m": 0.5, "tile_size": 512,
                   "channel_name": "images"},
        "run": {"work_dir": str(work_dir)},
    }
    if with_split:
        cfg["split"] = {"block_size_m": 1000.0, "ratios": [0.7, 0.15, 0.15],
                       "seed": 0}
    if output_subdir is not None:
        cfg["eval"] = {"output_subdir": output_subdir}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# --config resolution (Phase 1 default + Phase 2a eval_phase2a override)
# ---------------------------------------------------------------------------

def test_resolve_phase1_config_uses_default_eval_subdir(compare_module, tmp_path: Path) -> None:
    """Phase 1 config (no eval.output_subdir) -> parquet paths resolve
    to <work_dir>/eval/ and <work_dir>/baseline_rescored/eval/."""
    work_dir = tmp_path / "vegas_prepare"
    cfg = _mk_config(tmp_path / "phase1.yaml", work_dir, "AOI_2_Vegas",
                    output_subdir=None, with_split=True)
    resolved = compare_module.resolve_from_config(cfg)
    assert resolved["aoi"] == "AOI_2_Vegas"
    assert resolved["finetuned_parquet"] == work_dir / "eval" / "results.parquet"
    assert resolved["baseline_parquet"] == \
        work_dir / "baseline_rescored" / "eval" / "results.parquet"
    assert resolved["out_dir"] == work_dir / "eval"


def test_resolve_phase2a_config_uses_eval_phase2a_subdir(compare_module, tmp_path: Path) -> None:
    """Phase 2a config (eval.output_subdir: eval_phase2a) -> parquet
    paths resolve under <work_dir>/eval_phase2a/."""
    work_dir = tmp_path / "vegas_prepare"
    cfg = _mk_config(tmp_path / "phase2a.yaml", work_dir, "AOI_2_Vegas",
                    output_subdir="eval_phase2a", with_split=True)
    resolved = compare_module.resolve_from_config(cfg)
    assert resolved["finetuned_parquet"] == \
        work_dir / "eval_phase2a" / "results.parquet"
    assert resolved["baseline_parquet"] == \
        work_dir / "baseline_rescored" / "eval_phase2a" / "results.parquet"
    assert resolved["out_dir"] == work_dir / "eval_phase2a"


def test_resolve_rejects_config_without_split(compare_module, tmp_path: Path) -> None:
    """A Phase 2/3 benchmark config has no `split:` section, so there
    are no held-out tiles to compare. Must raise with a clear message."""
    work_dir = tmp_path / "vegas_prepare"
    cfg = _mk_config(tmp_path / "benchmark.yaml", work_dir, "AOI_2_Vegas",
                    output_subdir=None, with_split=False)
    with pytest.raises(ValueError, match="split"):
        compare_module.resolve_from_config(cfg)


def test_resolve_missing_config_file(compare_module, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_module.resolve_from_config(tmp_path / "nonexistent.yaml")


def test_resolve_then_compare_end_to_end(compare_module, tmp_path: Path) -> None:
    """Round-trip: resolve_from_config -> compare -> markdown/JSON in
    the right place. Uses Phase 2a (non-default subdir) to exercise
    the full path-derivation."""
    work_dir = tmp_path / "vegas_prepare"
    phase2a_cfg = _mk_config(tmp_path / "phase2a.yaml", work_dir,
                             "AOI_2_Vegas",
                             output_subdir="eval_phase2a", with_split=True)
    resolved = compare_module.resolve_from_config(phase2a_cfg)
    # Create fake parquets at the paths the resolver predicted.
    tiles = [(17, i, i) for i in range(5)]
    resolved["finetuned_parquet"].parent.mkdir(parents=True, exist_ok=True)
    resolved["baseline_parquet"].parent.mkdir(parents=True, exist_ok=True)
    _mk_full_parquet(resolved["baseline_parquet"],
                     prec=0.5, rec=0.5, f1=0.5, merge=0.1, split=0.0,
                     pixel_iou=0.5, chamfer=2.0, tiles=tiles)
    _mk_full_parquet(resolved["finetuned_parquet"],
                     prec=0.8, rec=0.8, f1=0.8, merge=0.05, split=0.0,
                     pixel_iou=0.7, chamfer=1.0, tiles=tiles)
    compare_module.compare(
        resolved["baseline_parquet"], resolved["finetuned_parquet"],
        aoi_name=resolved["aoi"], out_dir=resolved["out_dir"],
        n_boot=10, seed=0,
    )
    # Output must land inside the Phase 2a eval dir, not Phase 1's.
    assert (resolved["out_dir"] / "comparison_AOI_2_Vegas.md").is_file()
    assert (resolved["out_dir"] / "comparison_AOI_2_Vegas.json").is_file()


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
