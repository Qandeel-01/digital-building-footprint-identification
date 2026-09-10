# Contributions — mine vs. upstream, file by file

This project vendors [rodekruis/automated-building-detection](https://github.com/rodekruis/automated-building-detection)
at commit `e1668a470b94252040f27d26098826c293fbb46d` as a git submodule
at `external/abd/`. Nothing in that directory is authored by this
project; upstream is MIT-licensed to Ondrej Zacha, Wessel de Jong, and
Jacopo Margutti (Netherlands Red Cross). See [`NOTICE`](NOTICE).

Everything else in the tree is authored by Qandeel Fatima, MIT-licensed.

## Mine

| Path                                           | Purpose                                                                                      |
|------------------------------------------------|----------------------------------------------------------------------------------------------|
| `LICENSE`                                      | MIT, Copyright (c) 2026 Qandeel Fatima.                                                     |
| `NOTICE`                                       | Upstream attribution.                                                                       |
| `README.md`                                    | Provenance, repro steps, pinned SHA.                                                        |
| `CONTRIBUTIONS.md`                             | This file.                                                                                  |
| `pyproject.toml`                               | Package metadata + pinned direct deps + ruff/pytest config + `bfbench` script entry point.   |
| `environment.yml`                              | Conda env (conda-forge geospatial + torch cu128 from pip).                                   |
| `.gitignore`                                   | Weights, data, output, secrets.                                                              |
| `.env.example`                                 | Placeholder for optional env overrides. No secrets committed.                                |
| `.github/workflows/ci.yml`                     | ruff + pytest on push. CPU-only.                                                             |
| `configs/spacenet_vegas_sample.yaml`           | One-AOI pipeline config.                                                                     |
| `docs/environment-notes.md`                    | Every deviation from upstream requirements.txt, patch rationale, Windows NCCL note.          |
| `docs/adapter-decisions.md`                    | Rationale for GSD resampling + tile size + border handling.                                  |
| `docs/reproduce.md`                            | From-clean-clone runbook.                                                                    |
| `src/bfbench/__init__.py`                      | Version string.                                                                              |
| `src/bfbench/py.typed`                         | PEP 561 typing marker.                                                                       |
| `src/bfbench/data/__init__.py`                 | Package init, re-exports SpaceNetFetcher/AOI.                                                |
| `src/bfbench/data/spacenet.py`                 | Anonymous S3 fetcher for SpaceNet 2 (SN2_buildings).                                          |
| `src/bfbench/data/adapter.py`                  | SpaceNet PS-RGB -> abd XYZ tile tree, with configurable GSD + tile size + border handling.   |
| `src/bfbench/cli.py`                           | `bfbench run` / `bfbench verify` -- config-driven pipeline orchestrator.                     |
| `scripts/fetch_weights.py`                     | Google Drive download of `neat-fullxview-epoch75` with SHA-256 pinning.                      |
| `scripts/apply_upstream_patches.py`            | Idempotent applier for `patches/abd/*.patch` on top of the submodule SHA.                    |
| `scripts/predict_local.py`                     | Single-GPU predict shim (uses upstream Albunet/SemSeg unchanged; skips DDP/NCCL).            |
| `scripts/qa_figure.py`                         | 3-panel matplotlib sanity figure.                                                            |
| `patches/abd/0001-fix-vectorize-syntax.patch`  | Fix upstream `SyntaxError` at vectorize.py:56 (present since 2020).                          |
| `patches/abd/0002-fix-webcolors-css3-api.patch`| Modernize webcolors call (dropped constant CSS3_NAMES_TO_HEX).                               |
| `patches/abd/0003-fix-tiles-from-dir-windows.patch` | Fix `tiles_from_dir` regex on Windows paths.                                            |
| `patches/abd/0004-fix-filter-buildings-geopandas-api.patch` | Modernize geopandas `sjoin(op=)` + pandas `.append`.                                |
| `tests/__init__.py`                            | Empty (package marker for pytest -m).                                                        |
| `tests/test_adapter_geometry.py`               | Real assertions on GSD math, tile bounds, boundary alignment, pixel<->geographic round-trip. |

Everything under `runs/`, `weights/`, `data/`, `output/` is gitignored
and produced by running the pipeline; it is not committed.

## Upstream (submodule at `external/abd/`)

Pinned SHA: `e1668a470b94252040f27d26098826c293fbb46d`.
The two Python packages installed from this tree in editable mode are:

| Path (inside submodule)          | Purpose                                                                    |
|----------------------------------|----------------------------------------------------------------------------|
| `abd_model/src/abd_model/`       | AlbuNet + SemSeg loader + the `abd` CLI (predict, vectorize, train, ...).  |
| `abd_utils/src/abd_utils/`       | Image download / images-to-abd tile prep / filter-buildings CLI.           |
| `input/config.toml`              | The AlbuNet class/channel/model config bfbench references in its own YAML. |
| `input/AOI.geojson`              | Upstream's sample AOI (not used by this benchmark).                        |
| `README.md`                      | Upstream README (source of the Google Drive checkpoint link).              |
| `Dockerfile`                     | Upstream's container recipe (not used by this benchmark).                  |

No files inside `external/abd/` were modified in this repo's tree. Two
kinds of derived artifacts live outside the submodule:

- **Patches** (`patches/abd/*.patch`) applied on top at install time.
  See `docs/environment-notes.md` for the rationale of each.
- **Shim** (`scripts/predict_local.py`) that imports upstream classes
  and orchestrates them without touching upstream source.

## What is *not* in this repo

- No evaluation-metric code (IoU, F1, instance matching, merge/split
  rates). Phase 1 is scaffolding + baseline. Phase 2 (metrics) is
  intentionally reserved to be written separately.
- No performance number. Upstream reports IoU 0.79 on xBD; that is
  their number on their data and does not appear here as ours.
- No committed model weights, secrets, or downloaded SpaceNet chips.
