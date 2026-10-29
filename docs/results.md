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
