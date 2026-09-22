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
| AOI_5_Khartoum   |  TBD    |        |       |           |        |       |            |            |                               |
| AOI_4_Shanghai   |  TBD    |        |       |           |        |       |            |            |                               |
| AOI_3_Paris      |  TBD    |        |       |           |        |       |            |            |                               |

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
