# syntax=docker/dockerfile:1.7
#
# Reproducible image for the building-footprint-benchmark pipeline.
#
# Target: Linux x86_64 with an NVIDIA GPU exposed through the NVIDIA
# Container Toolkit. Matches the host env recipe in environment.yml
# (CUDA 12.8 / sm_120, Python 3.11, conda-forge geospatial stack,
# PyTorch cu128). See docs/phase0-decisions.md for the rationale.
#
# Build:
#     docker build -t bfbench:latest .
#
# Run (needs a GPU + bind-mounted weights/ and runs/):
#     docker run --rm -it --gpus all \
#         -v $(pwd)/weights:/workspace/weights \
#         -v $(pwd)/runs:/workspace/runs \
#         bfbench:latest bfbench --help
#
# Build-only in CI (no GPU required at build time).

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/opt/conda \
    ENV_NAME=bfbench

# ---------------------------------------------------------------------------
# OS deps: curl + ca-certs for miniforge; git for the patch applier (uses
# `git apply`); build-essential kept out intentionally -- conda-forge supplies
# every compiled lib we need (gdal, rasterio) as prebuilt binaries.
# ---------------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        bzip2 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Miniforge (conda + mamba, conda-forge default channel). Pinned installer
# URL for determinism; bump intentionally when conda-forge rolls over.
# ---------------------------------------------------------------------------
ARG MINIFORGE_VERSION=24.11.3-0
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
 && rm /tmp/miniforge.sh \
 && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

# ---------------------------------------------------------------------------
# Create the conda env from the single source of truth. This layer is the
# heaviest; it caches on environment.yml alone, so unrelated source edits
# never invalidate it.
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY environment.yml /workspace/environment.yml
RUN ${CONDA_DIR}/bin/conda env create -f /workspace/environment.yml \
 && ${CONDA_DIR}/bin/conda clean -afy

# From here on, the env's python is on PATH and is the default interpreter.
SHELL ["/bin/bash", "-lc"]

# ---------------------------------------------------------------------------
# Copy the upstream submodule and apply our patches on top of its pinned
# SHA. Done BEFORE installing abd_model / abd_utils so the editable installs
# see the patched source. The applier is idempotent and uses `git apply`,
# which is why external/abd/.git is kept in the context (see .dockerignore).
#
# `safe.directory '*'` guards against the "dubious ownership" check that
# newer git raises when the submodule was mounted/copied from a different
# uid than the current user.
# ---------------------------------------------------------------------------
COPY patches/ /workspace/patches/
COPY scripts/apply_upstream_patches.py /workspace/scripts/apply_upstream_patches.py
COPY external/abd/ /workspace/external/abd/
RUN git config --global --add safe.directory '*' \
 && python /workspace/scripts/apply_upstream_patches.py

# ---------------------------------------------------------------------------
# Install upstream packages editable, deps already satisfied by conda.
# ---------------------------------------------------------------------------
RUN pip install --no-deps -e /workspace/external/abd/abd_model \
 && pip install --no-deps -e /workspace/external/abd/abd_utils

# ---------------------------------------------------------------------------
# Install bfbench itself (editable, no deps).
# ---------------------------------------------------------------------------
COPY pyproject.toml /workspace/pyproject.toml
COPY src/ /workspace/src/
RUN pip install --no-deps -e /workspace

# ---------------------------------------------------------------------------
# Copy the rest of the tree (configs, scripts, tests, docs) last so these
# edits don't bust the heavy layers above.
# ---------------------------------------------------------------------------
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/
COPY tests/ /workspace/tests/
COPY docs/ /workspace/docs/
COPY Makefile /workspace/Makefile
COPY README.md CONTRIBUTIONS.md NOTICE LICENSE /workspace/

# ---------------------------------------------------------------------------
# Non-root runtime user. UID/GID 1000 matches the typical host developer
# account; override at build time if your bind mounts need a different uid.
# ---------------------------------------------------------------------------
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} bfbench \
 && useradd --uid ${USER_UID} --gid ${USER_GID} --shell /bin/bash --create-home bfbench \
 && chown -R bfbench:bfbench /workspace
USER bfbench

CMD ["bfbench", "--help"]
