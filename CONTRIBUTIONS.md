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
| `configs/spacenet_vegas.yaml`                  | Full-AOI benchmark config (Phase 2, Vegas).                                                  |
| `configs/spacenet_paris.yaml`                  | Full-AOI benchmark config (Phase 2, Paris).                                                  |
| `configs/spacenet_shanghai.yaml`               | Full-AOI benchmark config (Phase 2, Shanghai).                                               |
| `configs/spacenet_khartoum.yaml`               | Full-AOI benchmark config (Phase 2, Khartoum).                                               |
| `docs/environment-notes.md`                    | Every deviation from upstream requirements.txt, patch rationale, Windows NCCL note.          |
| `docs/adapter-decisions.md`                    | Rationale for GSD resampling + tile size + border handling. Phase 3 GSD sweep replaces Phase 2. |
| `docs/metric-decisions.md`                     | Phase 3 spec calls (asymmetry, centre-only, raster pixel_iou, required window_m, ...).          |
| `docs/reproduce.md`                            | From-clean-clone runbook.                                                                    |
| `docs/phase1-diagnosis.md`                     | Task 0b: 6-vs-31 diagnosis with tile-scope denominator + nodata + prob-histogram evidence.    |
| `docs/reference-data.md`                       | Dedup rules for merging chip geojson_buildings across seams (`touching_union` default).       |
| `docs/upstream-prs/*.md`                       | One factual repro per upstream PR branch (Task 5).                                            |
| `src/bfbench/__init__.py`                      | Version string.                                                                              |
| `src/bfbench/py.typed`                         | PEP 561 typing marker.                                                                       |
| `src/bfbench/data/__init__.py`                 | Package init, re-exports SpaceNetFetcher/AOI.                                                |
| `src/bfbench/data/spacenet.py`                 | Anonymous S3 fetcher for SpaceNet 2 (SN2_buildings).                                          |
| `src/bfbench/data/adapter.py`                  | SpaceNet PS-RGB -> abd XYZ tile tree, with configurable GSD + tile size + border handling.   |
| `src/bfbench/data/equal_area.py`               | Equal-area CRS helper + `assert_equal_area_crs` used across metrics.                          |
| `src/bfbench/data/mosaic.py`                   | Per-AOI GDAL VRT + `tile_is_fully_covered` (Task 1b).                                        |
| `src/bfbench/data/reference.py`                | Chip-geojson merge + dedup on seams (Task 1c).                                               |
| `src/bfbench/data/sampling.py`                 | Deterministic seeded tile sampling (Task 1d).                                                |
| `src/bfbench/data/summary.py`                  | Per-AOI descriptive stats (Task 1e).                                                         |
| `src/bfbench/prepare.py`                       | `bfbench prepare`: fetch -> mosaic -> merge references -> sample tiles -> summarise.          |
| `src/bfbench/metrics/pixel.py`                 | `pixel_iou` (raster interface, centre-only default; Phase 3 implemented).                     |
| `src/bfbench/metrics/count.py`                 | `count_error` per-tile + `count_error_aoi` aggregate (Phase 3 implemented).                   |
| `src/bfbench/metrics/stratify.py`              | `stratify_by_area` / `stratify_by_density` (Phase 3 implemented, bootstrap CI).               |
| `src/bfbench/metrics/instance.py`              | `match_instances` / `instance_prf` / `merge_rate` / `split_rate` -- RESERVED stubs.           |
| `src/bfbench/metrics/boundary.py`              | `boundary_distance` -- RESERVED stub (depends on MatchResult).                                |
| `src/bfbench/eval.py`                          | `bfbench eval`: per-tile metric harness writing tidy Parquet.                                 |
| `src/bfbench/run_prepared.py`                  | `bfbench run-prepared`: adapt+predict+vectorize+filter on the mosaic + sampled tiles.         |
| `src/bfbench/cli.py`                           | `bfbench run` / `run-prepared` / `verify` / `prepare` / `eval`.                               |
| `scripts/fetch_weights.py`                     | Google Drive download of `neat-fullxview-epoch75` with SHA-256 pinning.                      |
| `scripts/apply_upstream_patches.py`            | Idempotent applier for `patches/abd/*.patch` on top of the submodule SHA.                    |
| `scripts/predict_local.py`                     | Single-GPU predict shim (uses upstream Albunet/SemSeg unchanged; skips DDP/NCCL).            |
| `scripts/qa_figure.py`                         | 3-panel matplotlib sanity figure.                                                            |
| `scripts/diagnose_phase1.py`                   | Task 0b reproducer for the 6-vs-31 finding.                                                  |
| `scripts/gsd_sensitivity.py`                   | Task 0c reproducer for the GSD sweep (Phase 2; INVALIDATED by Phase 3, kept for history).      |
| `scripts/gsd_sweep_v2.py`                      | Phase 3 GSD sweep on 20 fully-covered mosaicked tiles (supersedes gsd_sensitivity).            |
| `scripts/figures.py`                           | Figures driven by the eval Parquet; synthetic mode is unmistakably labeled FAKE (Task 4).    |
| `scripts/qualitative_panels.py`                | Real-mode 3-panel figure for worst/median tile by count ratio.                                 |
| `scripts/select_vegas_chips.py`                | Probe SpaceNet chip headers via /vsis3/ and pick a compact contiguous cluster.                 |
| `scripts/diagnose_prepare.py`                  | Nodata + dedup diagnostic against a prepared work_dir.                                         |
| `patches/abd/0001-fix-vectorize-syntax.patch`  | Fix upstream `SyntaxError` at vectorize.py:56 (present since 2020).                          |
| `patches/abd/0002-fix-webcolors-css3-api.patch`| Modernize webcolors call (dropped constant CSS3_NAMES_TO_HEX).                               |
| `patches/abd/0003-fix-tiles-from-dir-windows.patch` | Fix `tiles_from_dir` regex on Windows paths.                                            |
| `patches/abd/0004-fix-filter-buildings-geopandas-api.patch` | Modernize geopandas `sjoin(op=)` + pandas `.append`.                                |
| `patches/abd/0005-single-gpu-predict-no-nccl.patch` | Add `abd predict --single-gpu` (Task 5).                                                |
| `tests/__init__.py`                            | Empty (package marker for pytest -m).                                                        |
| `tests/test_adapter_geometry.py`               | Real assertions on GSD math, tile bounds, boundary alignment, pixel<->geographic round-trip. |
| `tests/test_equal_area.py`                     | Equal-area CRS assertion + `is_equal_area` classifier tests.                                 |
| `tests/test_mosaic.py`                         | VRT builder + `tile_is_fully_covered` tests.                                                 |
| `tests/test_reference_dedup.py`                | Seam-partial dedup rules on synthetic geometry.                                              |
| `tests/test_sampling.py`                       | Seeded-sample determinism + hash stability tests.                                            |
| `tests/test_summary.py`                        | Per-AOI descriptive-stats tests.                                                             |
| `tests/test_metrics.py`                        | Failing-on-clean-checkout metric tests with hand-computed expected values (Task 2).           |
| `tests/test_eval.py`                           | End-to-end eval harness tests: stub-mode Parquet, resumability, seed/hash manifest.           |
| `tests/test_figures.py`                        | Smoke tests for the synthetic-placeholder path of `scripts/figures.py`.                       |

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
