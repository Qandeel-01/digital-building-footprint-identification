"""Phase 1 training runbook, intended to be pasted into one or more
Kaggle (or Colab) notebook cells. Not called from CI.

Prerequisites on the Kaggle/Colab machine (one time):

    !git clone --recurse-submodules <this-repo-url> bfb && cd bfb && \
      conda env create -f environment.yml && \
      conda activate bfbench && \
      python scripts/apply_upstream_patches.py && \
      pip install --no-deps -e ./external/abd/abd_model ./external/abd/abd_utils && \
      pip install --no-deps -e . && \
      python scripts/fetch_weights.py

Then, in order, for each AOI (Vegas / Paris / Shanghai / Khartoum):

    bfbench prepare   configs/spacenet_<aoi>_phase1.yaml
    bfbench run-prepared configs/spacenet_<aoi>_phase1.yaml --stop-after adapt
    bfbench train     configs/spacenet_<aoi>_phase1.yaml
    bfbench run-prepared configs/spacenet_<aoi>_phase1.yaml
    bfbench eval      configs/spacenet_<aoi>_phase1.yaml

Then the pooled run:

    bfbench train configs/spacenet_pooled_phase1.yaml

Then, for each AOI, re-run the baseline on the Phase 1 test tiles so
the comparison is apples-to-apples (same held-out tiles). This script
emits a temporary config pointing at the baseline weight and the Phase
1 work_dir, then runs eval:

    python scripts/kaggle_phase1.py evaluate-baseline-on-phase1-tiles \\
        configs/spacenet_<aoi>_phase1.yaml

Finally, build the comparison tables:

    python scripts/kaggle_phase1.py build-comparison configs/spacenet_<aoi>_phase1.yaml

Design note: this file is just orchestration; all real work happens in
`bfbench.train` / `bfbench.eval`. If something fails, run the step
underneath directly from the shell.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_LOG = logging.getLogger("bfbench.kaggle_phase1")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_WEIGHT = REPO_ROOT / "weights" / "neat-fullxview-epoch75.pth"


def _sh(cmd: list[str]) -> None:
    _LOG.info("+ %s", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"command failed (exit={r.returncode}): {' '.join(cmd)}")


def train_one(cfg_path: Path) -> None:
    """Run one AOI (or pooled) training end-to-end."""
    cfg_path = cfg_path.resolve()
    assert cfg_path.is_file(), f"missing {cfg_path}"
    _sh(["bfbench", "prepare", str(cfg_path)])
    _sh(["bfbench", "run-prepared", str(cfg_path), "--stop-after", "adapt"])
    _sh(["bfbench", "train", str(cfg_path)])


def evaluate_finetuned(cfg_path: Path) -> None:
    """Vectorise + eval against the fine-tuned checkpoint."""
    _sh(["bfbench", "run-prepared", str(cfg_path.resolve())])
    _sh(["bfbench", "eval", str(cfg_path.resolve())])


def _baseline_eval_cfg_path(cfg_path: Path) -> Path:
    """Emit a sibling config that reuses the Phase 1 / Phase 2a
    work_dir but points `abd.checkpoint` at the ORIGINAL baseline
    weight. This is how we score the baseline on the HELD-OUT test
    tiles without touching the fine-tuned artifacts.

    Also strips any `postprocess:` section: the baseline is binary and
    `argmax_interior` would raise `_validate_checkpoint_vs_postprocess`.
    """
    y = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    y["abd"]["checkpoint"] = str(BASELINE_WEIGHT.relative_to(REPO_ROOT))
    y["run"]["work_dir"] = str(Path(y["run"]["work_dir"]) / "baseline_rescored")
    # Baseline is always binary; drop any Phase 2a/2b postprocess intent.
    y.pop("postprocess", None)
    out = cfg_path.with_name(cfg_path.stem + "_baseline.yaml")
    out.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return out


def evaluate_baseline_on_phase1_tiles(cfg_path: Path) -> Path:
    """Score the UNMODIFIED baseline checkpoint on the Phase 1 / 2a
    held-out test tiles. Writes the eval parquet into
    `<work_dir>/baseline_rescored/<eval.output_subdir or "eval">/
    results.parquet` -- matching the path
    `compare_baseline_vs_finetuned.resolve_from_config` expects.

    Depends on `bfbench prepare` having been run for this AOI already
    (so sampled_tiles.json and block_split.json exist).
    """
    cfg_src = cfg_path.resolve()
    cfg_tgt = _baseline_eval_cfg_path(cfg_src)
    y_src = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))
    y_tgt = yaml.safe_load(cfg_tgt.read_text(encoding="utf-8"))
    src_dir = Path(y_src["run"]["work_dir"])
    dst_dir = Path(y_tgt["run"]["work_dir"])
    eval_subdir = str((y_src.get("eval") or {}).get("output_subdir", "eval"))
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sampled_tiles.json", "reference.geojson", "mosaic.vrt",
                "block_split.json"):
        if (src_dir / name).is_file():
            shutil.copy2(src_dir / name, dst_dir / name)
    _sh(["bfbench", "run-prepared", str(cfg_tgt)])
    _sh(["bfbench", "eval", str(cfg_tgt)])
    return dst_dir / eval_subdir / "results.parquet"


def build_comparison(cfg_path: Path) -> None:
    """Produce `comparison_<AOI>.{json,md}` next to the fine-tuned
    eval parquet. Works for Phase 1 and Phase 2a configs; paths are
    resolved by `compare_baseline_vs_finetuned.resolve_from_config`.
    """
    _sh([sys.executable, "scripts/compare_baseline_vs_finetuned.py",
         "--config", str(cfg_path.resolve())])


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub_t = sub.add_parser("train-one", help="prepare + adapt + train one AOI")
    sub_t.add_argument("config", type=Path)

    sub_e = sub.add_parser("evaluate-finetuned",
                           help="run-prepared + eval against the fine-tuned checkpoint")
    sub_e.add_argument("config", type=Path)

    sub_b = sub.add_parser("evaluate-baseline-on-phase1-tiles",
                           help="run-prepared + eval against the baseline, on the Phase 1 test tiles")
    sub_b.add_argument("config", type=Path)

    sub_c = sub.add_parser("build-comparison",
                           help="write comparison_<AOI>.{json,md}")
    sub_c.add_argument("config", type=Path)

    args = p.parse_args()
    if args.cmd == "train-one":
        train_one(args.config)
    elif args.cmd == "evaluate-finetuned":
        evaluate_finetuned(args.config)
    elif args.cmd == "evaluate-baseline-on-phase1-tiles":
        evaluate_baseline_on_phase1_tiles(args.config)
    elif args.cmd == "build-comparison":
        build_comparison(args.config)
    else:  # pragma: no cover - argparse already blocks
        raise ValueError(args.cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
