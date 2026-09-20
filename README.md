# building-footprint-benchmark

Reproducible harness for benchmarking an open-source building-footprint
detector (AlbuNet / U-Net + ResNet, trained on xBD) across multiple regions.

> **Phase 3 vertical slice (Vegas).** Baseline scaffolding (Phase 1) +
> multi-AOI data layer, metric spec, and evaluation harness (Phase 2) +
> real predictions on real Vegas mosaicked tiles (Phase 3).
>
> The pipeline runs end-to-end on the Vegas per-AOI mosaic:
> `bfbench prepare` → `bfbench run-prepared` → `bfbench eval`.
>
> Matching-independent metrics are implemented: `pixel_iou`, `count_error`
> (per-tile), `count_error_aoi`, `stratify_by_area`, `stratify_by_density`.
> The instance-level matching (`match_instances`, `instance_prf`) is
> reserved for the project owner and stubbed — the whole benchmark rests
> on their design.
>
> No headline performance number appears in this README; per-tile
> results live in the eval Parquet and its manifests. Numbers here
> would be premature until the matching functions are implemented.

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

# Phase 3 vertical slice (Vegas):
python scripts/select_vegas_chips.py --n 800    # pick a compact 800-chip cluster
bfbench prepare configs/spacenet_vegas.yaml     # download + mosaic + dedup + sample
bfbench run-prepared configs/spacenet_vegas.yaml    # adapt + predict + vectorize + filter
bfbench eval configs/spacenet_vegas.yaml            # per-tile metrics -> Parquet
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

## Reporting

Any performance number in this repo must be reproducible from a specific
config + weight checksum + input dataset revision recorded in the run
manifest. Upstream reports IoU 0.79 on xBD; that is their number on their
data and does not appear here as ours.

Headline performance numbers (F1, precision, recall, matching-IoU)
are deliberately not printed anywhere in this README or in
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) until the instance-matching
functions (`match_instances`, `instance_prf`) are implemented by the
project owner. Per-tile pixel-IoU and count-error values live in the
Parquet emitted by `bfbench eval`; consult those directly.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
