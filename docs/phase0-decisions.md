# Phase 0 — Reproducibility: decisions

Scope of this phase: a Dockerfile that captures the exact runtime
environment used to run the benchmark, a one-command target that
exercises the pipeline on a small sample, a CI job that builds the
image on every push, and upstream-PR drafts under
`docs/upstream-prs/` for the five patches in `patches/abd/`.

Nothing in this phase re-computes or re-reports any benchmark number.
The published AOI metrics in `README.md` and `docs/results.md` are
unchanged.

## Base image: `nvidia/cuda:12.8.0-runtime-ubuntu22.04`

**Why CUDA 12.8.** The target hardware is RTX 40/50-series Blackwell
(sm_120). Earlier CUDA tags do not include an sm_120 kernel, so
PyTorch would either silently fall back to CPU or crash at kernel
launch. See `docs/environment-notes.md` for the host-side rationale;
the Docker image must match.

**Why `runtime` and not `devel`.** We do not compile CUDA kernels in
this project. The geospatial native stack (GDAL, libproj, GEOS) and
PyTorch (cu128 wheel) both come in as prebuilt binaries from
conda-forge and PyPI. The `runtime` image is ~2 GB smaller than
`devel` and all our Python packages install without needing
`nvcc` / headers.

**Why Ubuntu 22.04.** Matches miniforge's primary Linux build and is
inside the support window of all pinned deps. 20.04 would also work
but is nearing end-of-life.

**Why not `cudnn-runtime`.** The PyTorch cu128 wheel bundles its own
cuDNN. Pulling in a second copy from the base image is dead weight
and risks ABI drift with the bundled one. We keep the base image
small and let the wheel own cuDNN.

## Env from `environment.yml` (single source of truth)

The Dockerfile `COPY`s `environment.yml` and runs
`conda env create -f environment.yml`. It does **not** re-pin any
version inside the Dockerfile. Reason: every version bump has to go
through `docs/environment-notes.md`, and we don't want two places to
update when a pin moves.

The env file is the first thing copied into the image after the base
packages, so the heavy `conda env create` layer caches on
`environment.yml` alone — unrelated source edits never bust it.

## Patch applier runs inside the image

The upstream `external/abd/` submodule is `COPY`-ed in with its
`.git/` directory intact, and `scripts/apply_upstream_patches.py` is
invoked. The applier shells out to `git -C external/abd apply`; this
is exactly the same code path used on the host (see
`docs/reproduce.md` step 2). One applier, one patch set, no drift.

The applier runs **before** `pip install -e ./external/abd/abd_model`
so the editable install picks up patched source. There is a pytest
assertion for this ordering in `tests/test_dockerfile.py`.

Alternative considered: ship a `patch -p1` fallback for the Docker
build to avoid needing `.git`. Rejected: the applier would have two
code paths depending on where it runs, and I'd have to keep both in
sync when upstream commit boundaries move. Keeping `.git` costs
~2 MB and keeps the applier single-path.

## Non-root runtime user

The image switches to a `bfbench` user (uid/gid 1000 by default,
overridable via build args). Bind mounts for `weights/` and `runs/`
would otherwise be written as root on the host.

## `docker-compose.yml` for the one-AOI target

The compose service `bfbench-smoke` runs
`scripts/docker_smoke.sh`, which executes
`bfbench prepare` → `run-prepared` → `eval` against
`configs/spacenet_vegas_smoke.yaml` (10 chips, Vegas). It mounts
`weights/` and `runs/` from the host so artifacts persist and the
`neat-fullxview-epoch75.pth` checkpoint (fetched by
`scripts/fetch_weights.py` with a pinned SHA-256) is reused across
runs.

The compose file reserves `driver: nvidia, capabilities: ["gpu"]`,
so running the stack on a GPU-less host fails fast with a clear
error from `docker compose` rather than crashing deep inside
PyTorch.

### Why this is a developer target, not a CI target

Running the smoke end-to-end needs:

1. The 300 MB `neat-fullxview-epoch75.pth` checkpoint. Google Drive
   does not expose a stable HTTP download for files that size, so
   `scripts/fetch_weights.py` depends on `gdown`'s interstitial
   handling, which has rate limits and intermittent 403s. CI should
   not take that dependency.
2. A GPU. GitHub-hosted runners do not have one; self-hosted runners
   with GPUs are out of scope for this phase.
3. Anonymous S3 access to `s3://spacenet-dataset`. That works in CI
   but together with (1) and (2) the overall run is >30 minutes and
   flaky on network.

CI therefore does **build-only**: `docker build` succeeds, the image
boots, `bfbench --help` prints. The smoke stack is reproducible
locally by any developer with a GPU and a fetched checkpoint.

## CI: `.github/workflows/docker-build.yml`

Separate from the existing lint-and-test workflow so the two can
fail independently and so lint feedback stays fast. Uses `docker/
build-push-action@v6` with `type=gha` cache (both `cache-from` and
`cache-to: mode=max`). First build is slow (~20 min); subsequent
builds that only touch Python source reuse the heavy env layer.

The image is not pushed to any registry from CI. Pushing requires
credentials and a decision about where the image lives; both are
out of scope here.

## Upstream PR drafts

Five branches already exist in `external/abd/` (see
`docs/upstream-prs/README.md`). The repro docs
(`docs/upstream-prs/0001.md` … `0005.md`) all follow the required
structure (**title**, **problem** with traceback, **fix** diff,
**regression test**, how this benchmark applies it). No new drafts
were authored in Phase 0; the audit confirmed the five existing
drafts cover every patch in `patches/abd/`.

**PRs are not opened from this repo.** The brief explicitly says the
maintainer will submit them.

## Limitations and non-goals

- The Dockerfile pins the base image tag but not its digest. If
  NVIDIA ever retags `12.8.0-runtime-ubuntu22.04` the image contents
  shift silently. If byte-for-byte reproducibility is required later,
  pin by `@sha256:...` and record the digest here.
- The smoke config fetches 10 live chips from SpaceNet. The pipeline
  is deterministic given the config + seed, but the raw bytes depend
  on the AWS S3 object version. SpaceNet does not change these
  objects in practice, but there's no cryptographic pin.
- Multi-arch builds are out of scope. The image is `linux/amd64`
  only. ARM64 hosts (Apple Silicon via Rosetta, Graviton) are not
  supported in Phase 0.
