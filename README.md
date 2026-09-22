# building-footprint-benchmark

Reproducible harness for benchmarking an open-source building-footprint
detector (AlbuNet / U-Net + ResNet, trained on xBD) across four
SpaceNet 2 regions: **Vegas, Paris, Shanghai, Khartoum**.

> The pipeline runs end-to-end per AOI:
> `bfbench prepare` → `bfbench run-prepared` → `bfbench eval`.
> Every headline number in this README is one row (or an aggregation
> of rows) in a per-AOI Parquet under
> `runs/<slug>_prepare/eval/results.parquet`. Full per-AOI breakdowns
> and stratified CSVs live in [`docs/results.md`](docs/results.md).

## What it does, in one paragraph

The AlbuNet checkpoint that ships with
[rodekruis/automated-building-detection](https://github.com/rodekruis/automated-building-detection)
was trained on xBD imagery at ~0.5 m/px. This project fixes the input
GSD via a mosaic-based adapter, runs the model over a seeded, spatially
deterministic sample of tiles from each of the four SpaceNet 2 AOIs,
and computes per-tile instance-matching metrics (precision / recall /
F1, merge / split rate, boundary Chamfer / Hausdorff) plus
matching-independent metrics (pixel IoU, count error) with an
equal-area CRS assertion running through every area / distance /
density computation. Nothing is quoted anywhere without a Parquet row
behind it.

## Results

Micro-averaged precision / recall / F1 across sampled tiles per AOI
(IoU threshold 0.5, 800-chip cluster near AOI centroid, seed
`20260911`). Every number here is regenerable from the corresponding
Parquet via `scripts/aoi_summary.py`.

| AOI            | n_tiles | precision | recall |   F1  | count_ratio | merge_rate | split_rate | pixel_iou_median | Chamfer_median (m) |
|----------------|--------:|----------:|-------:|------:|------------:|-----------:|-----------:|-----------------:|-------------------:|
| AOI_2_Vegas    |      43 |    0.510  | 0.373  | 0.431 |     0.730   |   0.163    |   0.018    |            0.625 |               1.40 |
| AOI_3_Paris    |      64 |    0.290  | 0.272  | 0.281 |     0.938   |   0.049    |   0.005    |            0.462 |               1.43 |
| AOI_4_Shanghai |      32 |    0.211  | 0.139  | 0.167 |     0.656   |   0.129    |   0.010    |            0.314 |               2.31 |
| AOI_5_Khartoum |      18 |    0.259  | 0.143  | 0.184 |     0.550   |   0.190    |   0.020    |            0.439 |               1.70 |

Full per-AOI table, stratified breakdowns and interpretation notes in
[`docs/results.md`](docs/results.md).

### F1 vs local building density (per AOI, 200 m window)

Per-tile F1 binned by the AOI-local reference density (buildings per
hectare in a 200 m window around the tile centroid). 95 % bootstrap
confidence bands per bin.

![F1 vs density 200 m](docs/results-figures/f1_vs_density_w200.png)

Density-window robustness at 100 m and 500 m windows:
[`f1_vs_density_w100.png`](docs/results-figures/f1_vs_density_w100.png),
[`f1_vs_density_w500.png`](docs/results-figures/f1_vs_density_w500.png).

### F1 vs building-size decile

F1 binned by the per-tile median reference building area (m²).

![F1 vs size decile](docs/results-figures/f1_vs_size_decile.png)

### Merge rate vs local density

Merger events (one prediction covering >= 2 reference buildings)
scale with local density.

![merge rate vs density](docs/results-figures/merge_rate_vs_density_w200.png)

### Qualitative panels (imagery / prediction / reference)

Median and worst tile per AOI ranked by `n_pred / n_ref`. Cyan =
reference. Pink = prediction. Only tile identifiers and polygon
counts appear on the figures.

| Vegas | Paris |
| ----- | ----- |
| ![Vegas](docs/results-figures/qualitative_vegas.png) | ![Paris](docs/results-figures/qualitative_paris.png) |

| Shanghai | Khartoum |
| -------- | -------- |
| ![Shanghai](docs/results-figures/qualitative_shanghai.png) | ![Khartoum](docs/results-figures/qualitative_khartoum.png) |

## Provenance

- **What is mine**: everything under `src/bfbench/`, `configs/`,
  `tests/`, `scripts/`, `docs/`, `.github/`, plus the top-level project
  metadata. `LICENSE` is MIT (Copyright 2026 Qandeel Fatima).
- **What is upstream**: `external/abd/` is a git submodule pointing at
  [rodekruis/automated-building-detection](https://github.com/rodekruis/automated-building-detection)
  pinned to commit **`e1668a470b94252040f27d26098826c293fbb46d`**.
  Upstream is MIT-licensed to Ondrej Zacha, Wessel de Jong, Jacopo
  Margutti (Netherlands Red Cross). See
  [`NOTICE`](NOTICE) and [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

Nothing in `external/abd/` is authored or modified by this project. It
is installed as a pip-editable dependency at env-setup time.

Five small upstream patches under `patches/abd/` are applied at
setup time; each has been re-verified in isolation (see
[`docs/upstream-prs/`](docs/upstream-prs/) and
`docs/upstream-prs/verification.json`).

## Repro from a clean clone

Detailed reproduction steps live in
[`docs/reproduce.md`](docs/reproduce.md). Short version, on Windows
with miniconda3 and an NVIDIA GPU:

```powershell
git clone --recurse-submodules `
    https://github.com/Qandeel-01/digital-building-footprint-identification.git `
    building-footprint-benchmark
cd building-footprint-benchmark
conda env create -f environment.yml
conda activate bfbench
python scripts/apply_upstream_patches.py
pip install --no-deps -e ./external/abd/abd_model ./external/abd/abd_utils
pip install --no-deps -e .
python scripts/fetch_weights.py

# Per-AOI slice (Vegas shown; --aoi-id 3, 4, 5 + matching config for
# Paris, Shanghai, Khartoum):
python scripts/select_chips.py --aoi-id 2 --n 800   # pick a compact 800-chip cluster
bfbench prepare configs/spacenet_vegas.yaml         # download + mosaic + dedup + sample
bfbench run-prepared configs/spacenet_vegas.yaml    # adapt + predict + vectorize + filter
bfbench eval configs/spacenet_vegas.yaml            # per-tile metrics -> Parquet
python scripts/aoi_summary.py runs/vegas_prepare/eval/results.parquet `
    --out runs/vegas_prepare/eval/aoi_summary.json  # AOI micro-average P/R/F1
```

Figures are generated via pytest (bare-python matplotlib segfaults on
some Windows GDAL builds; pytest sets up the interpreter differently
and the same code works):

```powershell
pytest tests/test_real_figures_integration.py tests/test_qualitative_panels_all_aois.py
```

## Design decisions

Written up rather than embedded silently in code:

- [`docs/adapter-decisions.md`](docs/adapter-decisions.md) — GSD
  resampling, tile size, border handling, and the Phase-3 GSD sweep
  (which invalidated the Phase-2 table because that table had been
  run on a chip that was 73 % nodata).
- [`docs/metric-decisions.md`](docs/metric-decisions.md) — the
  Phase-4 spec calls (asymmetric matching invariants, centre-only
  rasterisation default, raster `pixel_iou` interface, required
  `window_m`, densified boundary sampling, and the analytic Chamfer
  correction from 3.125 m → 2.5 m).
- [`docs/phase1-diagnosis.md`](docs/phase1-diagnosis.md) — the
  6-predictions-vs-31-references investigation that led to the
  per-AOI mosaic layer.
- [`docs/reference-data.md`](docs/reference-data.md) — chip-seam
  dedup rule for the reference layer.
- [`docs/environment-notes.md`](docs/environment-notes.md) — every
  deviation from upstream's pinned deps, the Windows / NCCL story,
  and the re-pinning procedure.

## Layout

```
building-footprint-benchmark/
├── external/abd/            # upstream submodule (pinned)
├── src/bfbench/             # this project's code
│   ├── data/                # SpaceNet fetcher, mosaic, adapter,
│   │                        # dedup, sampling, summary, equal-area
│   ├── metrics/             # instance, pixel, count, boundary,
│   │                        # stratify -- all implemented
│   ├── prepare.py           # `bfbench prepare`
│   ├── run_prepared.py      # `bfbench run-prepared`
│   ├── eval.py              # `bfbench eval`
│   └── cli.py               # `bfbench` entry point
├── configs/                 # one YAML per AOI
├── scripts/                 # chip selection, weight fetch, figures,
│                            # verifier, patch applier
├── tests/                   # ~93 tests, incl. equal-area, metrics,
│                            # eval end-to-end, patch verification
├── docs/                    # decisions, results, upstream PR repros
├── patches/abd/             # five upstream patches
├── .github/workflows/       # ruff + pytest CI
├── environment.yml
├── pyproject.toml
├── LICENSE                  # MIT, Qandeel Fatima 2026
├── NOTICE                   # upstream attribution
└── CONTRIBUTIONS.md         # file-by-file: mine vs upstream
```

## Reporting rules

Any performance number in this repo must be reproducible from a
specific config + weight checksum + input dataset revision recorded in
the run manifest. Upstream reports IoU 0.79 on xBD; that is their
number on their data and does not appear here as ours.

Equal-area CRS is enforced by assertion (`bfbench.data.equal_area`)
inside every metric that touches area, distance or density. Sampling
is seeded and deterministic: the same config + seed produces
byte-identical tile IDs.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
