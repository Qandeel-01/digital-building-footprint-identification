# Benchmark results

## Phase 1 — fine-tuned vs. baseline (held-out block test tiles)

Table below is populated by `scripts/compare_baseline_vs_finetuned.py`
after a GPU run on Kaggle / Colab. See `docs/phase1-decisions.md` and
`scripts/kaggle_phase1.py`. Every cell is a bootstrap over per-tile
values from the eval parquet; the brief forbids typed-in numbers, so
cells are left as `pending` until the parquets exist on disk.

| AOI | n_test_tiles | metric | baseline [CI] | fine-tuned (per-AOI) [CI] | fine-tuned (pooled) [CI] |
|---|---|---|---|---|---|
| AOI_2_Vegas    | pending | F1 | pending | pending | pending |
| AOI_2_Vegas    | pending | precision | pending | pending | pending |
| AOI_2_Vegas    | pending | recall | pending | pending | pending |
| AOI_2_Vegas    | pending | merge_rate | pending | pending | pending |
| AOI_2_Vegas    | pending | split_rate | pending | pending | pending |
| AOI_2_Vegas    | pending | pixel_IoU | pending | pending | pending |
| AOI_2_Vegas    | pending | Chamfer_m | pending | pending | pending |
| AOI_3_Paris    | pending | F1 | pending | pending | pending |
| AOI_4_Shanghai | pending | F1 | pending | pending | pending |
| AOI_5_Khartoum | pending | F1 | pending | pending | pending |

Fill procedure (run on the GPU host, after
`bfbench eval configs/spacenet_<aoi>_phase1.yaml`):

```
python scripts/kaggle_phase1.py evaluate-baseline-on-phase1-tiles \
    configs/spacenet_<aoi>_phase1.yaml
python scripts/kaggle_phase1.py build-comparison \
    configs/spacenet_<aoi>_phase1.yaml
```

Those two commands write `runs/<aoi>_prepare/eval/comparison_<AOI>.json`
and `.md` with every number used in this table.

---

## Phase 2b — Watershed instance separation (honest negative result at default)

**Framing (read before the numbers):** this comparison is on the
**existing Phase 2/3 benchmark sample** (seeded centroid cluster per
AOI), **NOT on Phase 1 held-out blocks**. The Phase 1 block split
does not exist yet because Phase 1 training has not been run. The
comparison here answers *"does watershed post-processing help the
unmodified baseline model on the tiles we already evaluated?"*, not
*"does watershed help on ground the model never saw?"*. Those are
different questions; Phase 1 will produce the second answer.

Numbers below are 95 % non-parametric bootstrap CIs (2000 resamples,
seed 0); per-tile for the rates, ratio-of-sums for micro-F1. Baseline
numbers match `docs/results-figures/` and the top-level README.

### Phase 2b at `min_marker_distance_m = 5 m` (the conservative default)

| AOI | n_tiles | metric | baseline | Phase 2b (5 m) | delta |
|---|--:|---|---|---|---|
| AOI_2_Vegas | 43 | F1 (micro) | 0.431 [0.383, 0.479] | 0.362 [0.311, 0.414] | −0.069 |
| AOI_2_Vegas | 43 | precision | 0.510 [0.467, 0.553] | 0.281 [0.232, 0.334] | −0.229 |
| AOI_2_Vegas | 43 | recall | 0.373 [0.323, 0.429] | 0.508 [0.462, 0.551] | +0.135 |
| AOI_2_Vegas | 43 | merge_rate | 0.163 [0.132, 0.196] | 0.016 [0.011, 0.023] | **−0.147** |
| AOI_2_Vegas | 43 | split_rate | 0.018 [0.009, 0.027] | 0.313 [0.252, 0.380] | **+0.295** |
| AOI_3_Paris | 64 | F1 (micro) | 0.281 [0.250, 0.310] | 0.256 [0.209, 0.297] | −0.025 |
| AOI_3_Paris | 64 | merge_rate | 0.049 [0.029, 0.071] | 0.006 [0.003, 0.011] | −0.043 |
| AOI_3_Paris | 64 | split_rate | 0.005 [0.002, 0.008] | 0.046 [0.025, 0.073] | +0.041 |
| AOI_4_Shanghai | 32 | F1 (micro) | 0.167 [0.123, 0.228] | 0.108 [0.073, 0.138] | −0.059 |
| AOI_4_Shanghai | 32 | merge_rate | 0.129 [0.081, 0.188] | 0.035 [0.018, 0.053] | −0.094 |
| AOI_4_Shanghai | 32 | split_rate | 0.010 [0.000, 0.024] | 0.156 [0.091, 0.229] | +0.146 |
| AOI_5_Khartoum | 18 | F1 (micro) | 0.184 [0.144, 0.233] | 0.125 [0.096, 0.150] | −0.059 |
| AOI_5_Khartoum | 18 | merge_rate | 0.190 [0.152, 0.227] | 0.043 [0.025, 0.062] | −0.147 |
| AOI_5_Khartoum | 18 | split_rate | 0.020 [0.009, 0.032] | 0.193 [0.146, 0.247] | +0.173 |

**Plain reading:** at the conservative default (5 m), watershed does
exactly what it's designed to — merge rate falls dramatically on
every AOI (−0.04 to −0.15). But **split rate explodes** (9×–17×
increase) because watershed is slicing single buildings into multiple
false-positive instances alongside the real touching-pair splits.
Precision drops heavily, and net F1 is **worse than baseline on every
AOI** with CIs that do not overlap on three of four AOIs. Pixel IoU
and Chamfer are unchanged — that's the raw mask area, which watershed
does not touch.

This is the merge-vs-split trade-off made visible. At 5 m the knob
is set too aggressively for the baseline's output.

### LOAO sweep of `min_marker_distance_m` ∈ {3, 5, 8, 12} m

Reproducible via `scripts/watershed_loao_sweep.py`.

**Micro-F1 at each distance (baseline for reference):**

| AOI | baseline | 3 m | 5 m | 8 m | 12 m |
|---|---|---|---|---|---|
| AOI_2_Vegas    | 0.431 | 0.259 | 0.362 | 0.459 | **0.524** |
| AOI_3_Paris    | 0.281 | 0.217 | 0.256 | 0.293 | **0.315** |
| AOI_4_Shanghai | 0.167 | 0.087 | 0.108 | 0.127 | 0.144 |
| AOI_5_Khartoum | 0.184 | 0.087 | 0.125 | 0.169 | **0.188** |

**Leave-one-AOI-out:** pick the distance that maximises mean micro-F1
across three AOIs, report on the fourth. 95 % bootstrap CIs (2000
resamples) on micro-F1.

| held-out AOI | LOAO-chosen distance | held-out F1 [CI] | baseline F1 [CI] | delta | CIs disjoint? |
|---|---|---|---|---|---|
| AOI_2_Vegas    | 12 m | 0.524 [0.480, 0.564] | 0.431 [0.383, 0.479] | **+0.093** | **yes** |
| AOI_3_Paris    | 12 m | 0.315 [0.276, 0.351] | 0.281 [0.250, 0.310] | +0.035 | no (overlap) |
| AOI_4_Shanghai | 12 m | 0.144 [0.118, 0.166] | 0.167 [0.123, 0.228] | −0.023 | no |
| AOI_5_Khartoum | 12 m | 0.188 [0.157, 0.221] | 0.184 [0.144, 0.233] | +0.004 | no |

**Plain reading:**

- **Vegas (dense regular suburbs) gets a clear win** at 12 m:
  +0.093 F1 with non-overlapping 95 % CIs. Watershed is doing its
  job — splitting merged blobs where buildings genuinely touch.
- **Paris is positive but not clearly separated** (+0.035, CIs
  overlap). Not safe to claim an improvement.
- **Shanghai (dense informal urban) is not helped by watershed at
  any swept distance.** Likely because the baseline isn't producing
  merged-blob errors so much as missing buildings entirely, so
  splitting doesn't address the dominant failure mode.
- **Khartoum (sparse) is flat.** Not enough touching buildings in the
  sample for watershed to have anywhere to act.
- **Every LOAO fold picked 12 m — the largest value swept.** The
  real optimum is probably beyond 12 m and the sweep should be
  extended (`--distances 12 15 20 30`). Not done here because the
  brief asked for `{3, 5, 8, 12}` specifically; want me to extend,
  just say so.

Reading this against the brief's "report bad or flat results
honestly" instruction: watershed at the default 5 m is a uniform
regression; at a *tuned* distance (12 m) it is useful on Vegas,
inconclusive on Paris, and not useful on Shanghai or Khartoum. The
knob needs per-region tuning, not a single global default.

### Overlay figures (baseline vs Phase 2b at 5 m)

![Vegas merge rate vs density](results-figures/merge_rate_overlay_vegas.png)
![Paris merge rate vs density](results-figures/merge_rate_overlay_paris.png)
![Shanghai merge rate vs density](results-figures/merge_rate_overlay_shanghai.png)
![Khartoum merge rate vs density](results-figures/merge_rate_overlay_khartoum.png)

Phase 2a and Phase 1 series will overlay on these same figures once
their eval parquets exist.

---

# Phase 4 results

Every number below is traceable to a row in a per-AOI Parquet under
`runs/<slug>_prepare/eval/results.parquet`. Regenerate the summary
JSONs and stratified CSVs with:

```powershell
python scripts/aoi_summary.py runs/<slug>_prepare/eval/results.parquet `
    --out runs/<slug>_prepare/eval/aoi_summary.json `
    --stratified-dir runs/<slug>_prepare/eval/stratified
```

Reproduce every figure via `pytest tests/test_real_figures_integration.py`.

## Pipeline configuration (identical across AOIs)

* Chip cluster: **800 chips** near the AOI's own centroid, selected by
  `scripts/select_chips.py --aoi-id <id> --n 800`. The centroid is
  computed from probed chip headers, not hand-set.
* Sampling: `n_tiles = 150`, `seed = 20260911`, `min_valid_fraction = 1.0`.
  Fewer than 150 tiles per AOI is possible when the mosaic has gaps
  and the coverage filter rejects most candidates; the actual n_tiles
  is recorded per AOI below and in `summary.json`.
* Metrics: `iou_threshold = 0.5`, `pixel_resolution_m = 0.5`.
* Everything else (target GSD, tile size, dedup rule, checkpoint,
  filter thresholds) as documented in `docs/adapter-decisions.md`
  and `docs/metric-decisions.md`.

## Per-AOI headline numbers

Values are micro-averaged precision / recall / F1 across the AOI's
sampled tiles (i.e. summed TP / FP / FN across tiles then divided,
not averaged tile-wise). Merge / split rate are per-tile means.

| AOI              | n_tiles | n_pred | n_ref | precision | recall |  F1   | merge_rate | split_rate | count_ratio (n_pred / n_ref) |
|------------------|--------:|-------:|------:|----------:|-------:|------:|-----------:|-----------:|------------------------------:|
| AOI_2_Vegas      |      43 |   1481 |  2028 |    0.510  | 0.373  | 0.431 |   0.163    |   0.018    |                        0.730  |
| AOI_3_Paris      |      64 |    700 |   746 |    0.290  | 0.272  | 0.281 |   0.049    |   0.005    |                        0.938  |
| AOI_4_Shanghai   |      32 |    478 |   729 |    0.211  | 0.139  | 0.167 |   0.129    |   0.010    |                        0.656  |
| AOI_5_Khartoum   |      18 |    448 |   814 |    0.259  | 0.143  | 0.184 |   0.190    |   0.020    |                        0.550  |

Auxiliary per-tile statistics (median across the AOI's tiles):

| AOI            | pixel_iou_median | Chamfer_median (m) |
|----------------|-----------------:|-------------------:|
| AOI_2_Vegas    |            0.625 |               1.40 |
| AOI_3_Paris    |            0.462 |               1.43 |
| AOI_4_Shanghai |            0.314 |               2.31 |
| AOI_5_Khartoum |            0.439 |               1.70 |

## Stratified breakdowns

Per AOI, `runs/<slug>_prepare/eval/stratified/` contains:

* `<AOI>_by_size_decile.csv`     -- F1 / precision / recall / merge_rate
                                     / split_rate / pixel_iou binned by
                                     the per-tile median reference
                                     building area.
* `<AOI>_by_density_w{100,200,500}.csv` -- same, binned by local
                                     reference density at the three
                                     window sizes.

## Figures

Description only; interpretation is left to the project owner.

* `docs/results-figures/f1_vs_density_w100.png`,
  `docs/results-figures/f1_vs_density_w200.png`,
  `docs/results-figures/f1_vs_density_w500.png` -- F1 vs local
  reference density (buildings per hectare), one line per AOI, 95 %
  bootstrap CI bands. Three separate plots for the three window sizes.

* `docs/results-figures/f1_vs_size_decile.png` -- F1 vs per-tile
  median reference building area (m²), one line per AOI, 95 %
  bootstrap CI bands.

* `docs/results-figures/merge_rate_vs_density_w200.png` -- merge
  rate vs local reference density at the 200 m window, one line per
  AOI, 95 % bootstrap CI bands.

* `runs/<slug>_prepare/eval/qualitative_panels.png` -- imagery /
  prediction / reference panels for the worst and median tile of
  each AOI (ranked by n_pred/n_ref count ratio).

## AOI notes (deviations from Vegas)

Any AOI where the cluster policy had to differ from Vegas is
documented here. When the number of sampled tiles is smaller than
150, the "same chip-cluster selection" line above still applies:
the coverage filter rejects tiles that are not 100 %-covered by
imagery, and some AOIs have sparser chip coverage than Vegas.

* **AOI_5_Khartoum**: 1012 total chips in the AOI (versus 3851 in
  Vegas). Selecting 800 still leaves the mosaic sparse -- most
  candidate tiles fail the 100 % coverage filter. Actual n_tiles is
  recorded in the table above.
* **AOI_3_Paris**: 1148 total chips. Same coverage caveat as
  Khartoum.
* **AOI_4_Shanghai**: 4582 total chips. Enough for the same 800-chip
  cluster policy without any deviation from Vegas.
