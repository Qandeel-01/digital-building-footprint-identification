# Kaggle runbook — Khartoum Phase 1 + Phase 2a

Minimal, deduplicated runbook for one AOI (Khartoum) on a Kaggle
notebook (T4 or P100, 16 GB). Total GPU time: **~70–90 min**.

## Expected split sizes (deterministic given `split.seed=20260911`)

| grid | 1 km × 1 km in EPSG:6933 |
|---|---|
| total blocks covering Khartoum extent | 108 |
| train blocks (70 %) | 76 |
| val blocks (15 %) | 9 |
| **test blocks (15 %)** | **23** |

At `sampling.n_tiles = 400`:

| bucket | expected tiles |
|---|---|
| train | ~274 |
| val | ~35 |
| **test** | **~91** |

**CI usability:** 91 test tiles gives a ~±0.04 F1 bootstrap CI
half-width — plenty of resolution for a +0.05 Phase 1 vs baseline
delta to separate clearly. If you see materially fewer test tiles
after `bfbench prepare` runs (because the sampled set was pushed
into fewer blocks than expected), downgrade your conclusions
accordingly. Verify numbers in `runs/khartoum_prepare/block_split.json`
after cell 3.

> Note: val is only ~9 % (not 15 %) because the deterministic
> sha256(seed, i, j) assignment lands few blocks in val for this
> specific AOI + seed. 70/15/15 is the long-run average, not a
> per-AOI guarantee. 35 val tiles is workable for early stopping.

---

## Cells

### Cell 1 — environment (one-time per session)

```bash
%cd /kaggle/working
!git clone --recurse-submodules \
    https://github.com/Qandeel-01/digital-building-footprint-identification.git bfb
%cd bfb
!python scripts/apply_upstream_patches.py
!pip install --quiet --no-deps \
    -e ./external/abd/abd_model ./external/abd/abd_utils -e . \
    'mercantile==1.1.5' 'scikit-image==0.24.0' 'pyarrow==17.0.0'
```

### Cell 2 — checkpoint (from your Kaggle dataset)

Upload `neat-fullxview-epoch75.pth` as a Kaggle dataset once; then:

```bash
!mkdir -p weights
!cp /kaggle/input/<your-dataset-slug>/neat-fullxview-epoch75.pth weights/
!python -c "
from pathlib import Path
from scripts.fetch_weights import _sha256, _write_pinned_checksum
p = Path('weights/neat-fullxview-epoch75.pth')
digest = _sha256(p)
print('observed sha256:', digest)
assert digest == '47d2d3211c808a6408985b2b86c9fd36194d8db7100eddea6faf9c972014e69e', 'checkpoint sha mismatch'
_write_pinned_checksum(digest)
"
```

### Cell 3 — Khartoum prepare (fetch + mosaic + split)

```bash
!bfbench prepare configs/spacenet_khartoum_phase1.yaml
!python -c "
import json
sp = json.load(open('runs/khartoum_prepare/block_split.json'))
print(f\"actual train/val/test tiles: {sp['n_train']}/{sp['n_val']}/{sp['n_test']}\")
print(f\"assignment_hash: {sp['assignment_hash'][:16]}\")
"
```

### Cell 4 — adapt the tile tree (CPU-bound, no GPU yet)

```bash
!bfbench run-prepared configs/spacenet_khartoum_phase1.yaml --stop-after adapt
```

### Cell 5 — Phase 1 fine-tune (GPU, ~20–30 min)

```bash
!bfbench train configs/spacenet_khartoum_phase1.yaml
```

### Cell 6 — produce the Phase 1 comparison table

Three things in one cell. **`kaggle_phase1.py` orchestrates all of
them** — no need to run cells 5/6/7 from the old runbook separately:

```bash
# predict/vectorize/filter/eval with the fine-tuned ckpt + rescore
# the baseline on the SAME held-out test tiles + build the comparison
!python scripts/kaggle_phase1.py evaluate-finetuned \
    configs/spacenet_khartoum_phase1.yaml
!python scripts/kaggle_phase1.py evaluate-baseline-on-phase1-tiles \
    configs/spacenet_khartoum_phase1.yaml
!python scripts/kaggle_phase1.py build-comparison \
    configs/spacenet_khartoum_phase1.yaml
!cat runs/khartoum_prepare/eval/comparison_AOI_5_Khartoum.md
```

### Cell 7 — Phase 2a fine-tune (GPU, ~25–35 min)

```bash
!bfbench train configs/spacenet_khartoum_phase2a.yaml
```

### Cell 8 — Phase 2a eval (the `argmax_interior` wiring is in; a
3-class checkpoint without it now **errors loudly** before predict
runs rather than producing silent garbage)

```bash
!bfbench run-prepared configs/spacenet_khartoum_phase2a.yaml
!bfbench eval configs/spacenet_khartoum_phase2a.yaml
# Same comparison helper, pointed at the 2a config this time:
!python scripts/kaggle_phase1.py build-comparison \
    configs/spacenet_khartoum_phase2a.yaml 2>/dev/null \
    || python -c "print('build-comparison does not yet support Phase 2a configs; compare manually via scripts/compare_baseline_vs_finetuned.py')"
```

### Cell 9 — package parquets for download

```bash
!cd runs && zip -r /kaggle/working/khartoum_runs.zip khartoum_prepare 2>/dev/null
!ls -lh /kaggle/working/khartoum_runs.zip
# then use the Kaggle "Output" tab to download
```

---

## What to send back

- `runs/khartoum_prepare/train/training_manifest.json` (+ `best.pth`
  + `best.pth.sha256.json`)
- `runs/khartoum_prepare/train_phase2a/*` (same shape as above)
- `runs/khartoum_prepare/eval/results.parquet`
- `runs/khartoum_prepare/eval_baseline_rescored/results.parquet`
  (produced by `evaluate-baseline-on-phase1-tiles`)
- `runs/khartoum_prepare/eval/comparison_AOI_5_Khartoum.{json,md}`
- `runs/khartoum_prepare/block_split.json`

Paste those paths back in a reply or attach the zip; I'll land the
numbers into `docs/results.md` without typing any value by hand.
