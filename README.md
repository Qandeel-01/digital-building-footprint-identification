# building-footprint-benchmark

Reproducible harness for benchmarking an open-source building-footprint
detector (AlbuNet / U-Net + ResNet, trained on xBD) across multiple regions.

> **Phase 1 scaffolding.** This repo currently sets up a verified baseline:
> upstream pinned as a submodule, a working conda env, a SpaceNet-2 adapter,
> and a one-command end-to-end run producing GeoJSON footprints. Evaluation
> metrics (IoU, F1, instance matching, merge/split) are intentionally out of
> scope for Phase 1 — those will be authored separately.

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
pip install -e ./external/abd/abd_model ./external/abd/abd_utils
pip install -e .
python scripts/fetch_weights.py
bfbench run configs/spacenet_vegas_sample.yaml
```

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

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
