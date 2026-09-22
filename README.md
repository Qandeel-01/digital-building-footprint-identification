# building-footprint-benchmark

Reproducible harness for benchmarking an open-source building-footprint
detector (AlbuNet / U-Net + ResNet, trained on xBD) across multiple regions.

> **Phase 4 -- multi-AOI results with all metrics implemented.**
> Baseline scaffolding (Phase 1) + multi-AOI data layer, metric spec,
> and evaluation harness (Phase 2) + Vegas vertical slice (Phase 3) +
> four AOIs with instance-matching metrics (Phase 4).
>
> The pipeline runs end-to-end per AOI:
> `bfbench prepare` → `bfbench run-prepared` → `bfbench eval`.
>
> Every headline number below is one row (or an aggregation of rows) in
> `runs/<slug>_prepare/eval/results.parquet`. Full per-AOI breakdowns +
> stratified CSVs live in [`docs/results.md`](docs/results.md).

## Provenance

- **What is mine**: everything under `src/bfbench/`, `configs/`, `tests/`,
  `scripts/`, `docs/`, `.github/`, plus the top-level project metadata.
- **What is upstream**: `external/abd/` is a git submodule pointing at
  [rodekruis/automated-building-detection](https://github.com/rodekruis/automated-building-detection)
  pinned to commit **`e1668a470b94252040f27d26098826c293fbb46d`**.
  Upstream is MIT-licensed to Ondrej Zacha, Wessel de Jong, Jacopo Margutti
  (Netherlands Red Cross). See `NOTICE` and `CONTRIBUTIONS.md`.

Nothing in `external/abd/` is authored or modified by this project. It is
installed as a pip-editable dependency at env setup time.

## Repro from a clean clone

Detailed reproduction steps live in [`docs/reproduce.md`](docs/reproduce.md).
Short version, on Windows with miniconda3 and an NVIDIA GPU:

```powershell
git clone --recurse-submodules <this-repo-url> building-footprint-benchmark
cd building-footprint-benchmark
conda env create -f environment.yml
conda activate bfbench
python scripts/apply_upstream_patches.py
pip install --no-deps -e ./external/abd/abd_model ./external/abd/abd_utils
pip install --no-deps -e .
python scripts/fetch_weights.py

# Per-AOI vertical slice (Vegas shown; substitute --aoi-id 3/4/5 + the
# matching config for Paris/Shanghai/Khartoum):
python scripts/select_chips.py --aoi-id 2 --n 800   # pick a compact 800-chip cluster
bfbench prepare configs/spacenet_vegas.yaml         # download + mosaic + dedup + sample
bfbench run-prepared configs/spacenet_vegas.yaml    # adapt + predict + vectorize + filter
bfbench eval configs/spacenet_vegas.yaml            # per-tile metrics -> Parquet
python scripts/aoi_summary.py runs/vegas_prepare/eval/results.parquet \
    --out runs/vegas_prepare/eval/aoi_summary.json  # AOI micro-average P/R/F1
```

Upstream is vendored unchanged; one small patch is applied on top of the
pinned SHA to fix a Python syntax error that has been in
`abd_model.tools.vectorize` since 2020. See
[`docs/environment-notes.md`](docs/environment-notes.md).

## Layout

```
building-footprint-benchmark/
├── external/abd/            # upstream submodule (pinned)
├── src/bfbench/             # this project's code
│   ├── data/                # SpaceNet fetcher + tile adapter
│   └── cli.py               # top-level `bfbench` CLI
├── configs/                 # one YAML per AOI
├── scripts/                 # weight fetch, QA figure
├── tests/                   # adapter geometry tests
├── docs/                    # environment + adapter decisions
├── .github/workflows/       # ruff + pytest CI
├── environment.yml
├── pyproject.toml
├── LICENSE                  # MIT, Qandeel Fatima 2026
├── NOTICE                   # upstream attribution
└── CONTRIBUTIONS.md         # file-by-file: mine vs upstream
```

## Results

Micro-averaged precision / recall / F1 across sampled tiles per AOI
(IoU threshold 0.5, 800-chip cluster near AOI centroid, seed
`20260911`). Every number here is regenerable from the corresponding
Parquet via `scripts/aoi_summary.py`.

| AOI            | n_tiles | precision | recall |   F1  | count_ratio | merge_rate |
|----------------|--------:|----------:|-------:|------:|------------:|-----------:|
| AOI_2_Vegas    |      43 |    0.510  | 0.373  | 0.431 |     0.730   |   0.163    |
| AOI_3_Paris    |      64 |    0.290  | 0.272  | 0.281 |     0.938   |   0.049    |
| AOI_4_Shanghai |      32 |    0.211  | 0.139  | 0.167 |     0.656   |   0.129    |
| AOI_5_Khartoum |      18 |    0.259  | 0.143  | 0.184 |     0.550   |   0.190    |

Full per-AOI table, stratified breakdowns and figures in
[`docs/results.md`](docs/results.md).

Any performance number in this repo must be reproducible from a specific
config + weight checksum + input dataset revision recorded in the run
manifest. Upstream reports IoU 0.79 on xBD; that is their number on their
data and does not appear here as ours.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
