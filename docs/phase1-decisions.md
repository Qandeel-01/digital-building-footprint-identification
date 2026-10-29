# Phase 1 — Domain adaptation: decisions

Scope: fine-tune the Netherlands Red Cross AlbuNet checkpoint
(trained on xBD, ~0.5 m/px) on SpaceNet 2 training blocks per AOI and
one pooled run, then re-evaluate on held-out geographic blocks. Every
headline number lives in a parquet (same tidy schema as Phase 2 / 3);
this doc only writes down the choices behind those numbers.

The code lives under `src/bfbench/train/` + `src/bfbench/train_cli.py`
and is wired into `bfbench train`. All configs are `configs/
spacenet_*_phase1.yaml`.

## Why fine-tune (reminder of the diagnosis)

Baseline micro-F1 at IoU 0.5 on the Phase 2 benchmark:

- Vegas 0.43, Paris 0.28, Shanghai 0.17, Khartoum 0.18.

Phase 1 diagnosis (`docs/phase1-diagnosis.md`) ruled out GSD
mismatch, border artifacts and chip-seam deduplication as the primary
cause. The remaining hypothesis is domain shift: xBD imagery is
visually quite different from SpaceNet (sensor, radiometric
calibration, scene composition). Fine-tuning is the direct test.

## Geographic block split

Spatially random tile splits leak information: adjacent tiles overlap
on the same ground, so a building touching a split boundary appears
in both train and test. The split is therefore block-based:

- **Equal-area CRS**: EPSG:6933 (same as every area/distance/density
  metric in `bfbench.data.equal_area`). Blocks live in metres, not
  degrees, so a 1 km block means 1 km everywhere on Earth.
- **Block size**: **1000 m**. Half the median SpaceNet 2 chip side
  (~450 m across) in each direction, so a block is 2–4 chips wide.
  Smaller blocks risk a building straddling the boundary; larger
  blocks waste data at per-AOI sample sizes we can afford.
- **Ratios**: **70 / 15 / 15** (train / val / test). Val is used
  only for early stopping, so a smaller share is tolerable; 15 %
  gives enough bootstrap power for the final test tables.
- **Assignment**: `sha256(seed, i, j)` → uniform `[0, 1)` → bucketed
  by ratio. The hash is defined, not Python's `hash()`, because the
  latter is randomised per process. Two runs with the same seed +
  config produce byte-identical assignments; this is pinned by the
  `assignment_hash` recorded in `training_manifest.json` and asserted
  by `tests/test_block_split.py`.
- **Zero-overlap test**: `tests/test_block_split.py::
  test_train_test_blocks_have_zero_polygon_overlap_area` builds the
  union of all train blocks, the union of all test blocks, intersects
  them in EPSG:6933 and asserts the intersection area is below 1 m²
  (1 ppm of a 1 km block). This is stronger than "no shared tile
  IDs" because it also rules out two blocks that *should* overlap
  silently being adjacent on different splits.

Non-goals explicitly ruled out: K-fold spatial CV (compute budget on
one 16 GB GPU does not permit it), stratified block sampling by
building density (adds complexity without ruling out any confound
the two-way split doesn't already handle for a baseline comparison).

## Training config (defaults — all overridable from YAML)

| field | default | reasoning |
|---|---|---|
| optimizer | **AdamW** | Standard for segmentation; adaptive LR tolerates the per-AOI dataset size differences. |
| learning rate | **1e-4** | 10× smaller than training-from-scratch; standard for fine-tuning. |
| weight decay | 1e-4 | Mild; the baseline already generalises. |
| scheduler | **cosine warm restarts** (T₀=5) | Lets the LR revisit the useful range after the model fits a plateau; common for seg models trained for ~30 epochs. |
| batch size | **4** | 512×512 RGB tiles × fp16 fits in ~10 GB with Albunet; leaves headroom. |
| epochs | **30** | Target 50 but with early stop (patience 5) on val loss; 30 is the typical realised number. |
| loss | **BCE + Dice (0.5 / 0.5)** | Standard binary-seg baseline; Dice handles class imbalance, BCE anchors calibration. |
| mixed precision | **fp16 via `torch.amp`** | Required to fit batch size 4 on 16 GB. CPU fallback transparently disables AMP. |
| augmentation | **none** | Deliberate. Fine-tuning gains should come from the SpaceNet domain, not from augmentation choices. Adding aug is a Phase 2 lever. |
| seed | **20260911** | Same seed used everywhere else in the pipeline for consistency. |

Phase 1 does not sweep hyperparameters on the test split. Any
hyperparameter decision made after seeing a test-split number counts
as tuning on the test split; the brief rules that out. The defaults
above are chosen from literature + a-priori memory sizing.

### Loss choice (expanded)

BCE on logits (via `binary_cross_entropy_with_logits`) is numerically
stable and matches the baseline's output head (single-channel
sigmoid). Soft Dice (`1 - (2∩ + s) / (|P| + |T| + s)`, s=1) over the
sigmoid outputs addresses class imbalance on sparse-building AOIs
(Khartoum). The 0.5 / 0.5 blend is deliberately boring: picking the
blend by eyeballing per-AOI losses is a very easy source of overfit.

### Mixed precision details

`torch.amp.autocast(device_type="cuda", dtype=torch.float16)` wraps
the forward + loss; the scaler scales the backward loss and unscales
before `optimizer.step`. On CPU the loop uses a `_NoopScaler` so unit
tests exercise the exact same code paths. `BatchNorm` keeps full
precision because Albunet's encoder is a resnet50 with BN layers;
fp16 BN is a known accuracy footgun.

## Model + checkpoint handling

- **Model**: imports `abd_model.nn.albunet.Albunet` unchanged.
  `build_albunet_from_checkpoint` loads the baseline `.pth`, strips
  `"module."` prefixes if present (upstream saved under DDP), and
  calls `load_state_dict(..., strict=True)`.
- **Checkpoint format on save**: a dict with the same keys the
  upstream `predict.py` reads (`nn`, `shape_in`, `shape_out`,
  `encoder`, `state_dict`, `model_version`, `uuid`, `loader`). The
  fine-tuned checkpoint drops into `bfbench run-prepared` with no
  code changes.
- **Checksum pinning**: `TrainingManifest.output_sha256` is computed
  AFTER the file is written and the sibling `best.pth.sha256.json`
  is produced. `write_training_manifest` refuses to write a manifest
  whose declared sha256 disagrees with the on-disk file; this is the
  same guard `scripts/fetch_weights.py` applies to the baseline.

## Pooled model

One training run on the concatenation of all four AOIs' train blocks;
early stopping on the Vegas val set (the parent AOI). The pooled
model is evaluated per-AOI on each AOI's test blocks and reported
alongside the four per-AOI models in the comparison table.

Rationale for a Vegas-anchored val: picking any one AOI as the val
set is somewhat arbitrary; Vegas is chosen because it has the richest
building density (most informative val signal) and the smallest
upstream domain gap, so val-loss movement signals real training
progress rather than noise.

## Evaluation

- The brief says "evaluate with the EXISTING eval harness on held-out
  blocks". We re-use `bfbench eval` unchanged. The eval reads
  `sampled_tiles.json` (= all 400 sampled tiles) and
  `block_split.json` (= the geographic split), and metric rows are
  tagged by tile; filtering to test tiles happens at comparison time.
- **Baseline comparison**: `scripts/compare_baseline_vs_finetuned.py`
  reads TWO eval parquets (one from the baseline, one from the fine-
  tuned checkpoint) on the SAME held-out tiles and emits a markdown
  table + JSON. For a fair comparison, the baseline must be
  **re-evaluated** on the Phase 1 test tiles; the Phase 2 benchmark
  numbers (0.43, 0.28, 0.17, 0.18) are on a different sampling and
  are kept in `README.md` for historical reference only.
- **Bootstrap CIs**: 2000 resamples with replacement, seed 0,
  percentile method, 95 %. Precision/recall/F1 and the rate metrics
  bootstrap the mean across tiles; pixel IoU and Chamfer bootstrap
  the median across tiles (the median is the point estimate in the
  Phase 2 table).

## GPU runbook (Kaggle / Colab)

See `scripts/kaggle_phase1.py` for the orchestration. One AOI, end-
to-end:

```
bfbench prepare   configs/spacenet_vegas_phase1.yaml
bfbench run-prepared configs/spacenet_vegas_phase1.yaml --stop-after adapt
bfbench train     configs/spacenet_vegas_phase1.yaml
bfbench run-prepared configs/spacenet_vegas_phase1.yaml
bfbench eval      configs/spacenet_vegas_phase1.yaml

# baseline, scored on the Phase 1 test tiles:
python scripts/kaggle_phase1.py evaluate-baseline-on-phase1-tiles \
    configs/spacenet_vegas_phase1.yaml

# comparison table:
python scripts/kaggle_phase1.py build-comparison \
    configs/spacenet_vegas_phase1.yaml
```

Repeat for Paris / Shanghai / Khartoum, then do the pooled run with
`configs/spacenet_pooled_phase1.yaml`.

## What is not here

- No benchmark numbers. Running the training is out of scope for the
  developer-facing CI; a GPU run on Kaggle populates the parquets.
  `docs/results.md` will be filled from those parquets, never typed
  in by hand.
- No scheduler ablation, no AMP vs fp32 comparison, no augmentation
  ablation. Phase 1 is "does fine-tuning help?" — scope stays narrow.
- No early-stop tuning per AOI. The patience of 5 is the same for
  all four runs plus the pooled one.
- No regularisation beyond weight decay (no dropout, no stochastic
  depth, no cutout). Those are additive levers for Phase 2 at the
  earliest.
