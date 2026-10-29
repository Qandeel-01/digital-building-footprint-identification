#!/usr/bin/env bash
# scripts/docker_smoke.sh
#
# One-AOI end-to-end smoke run inside the bfbench Docker image. Invoked by
# `docker compose run --rm bfbench-smoke` (see docker-compose.yml).
#
# Preconditions on the host:
#   - a GPU exposed via the NVIDIA Container Toolkit
#   - weights/neat-fullxview-epoch75.pth present (via scripts/fetch_weights.py)
#
# Steps:
#   1. Verify the checkpoint is on disk (sha256 is checked by fetch_weights.py
#      on the host; here we just assert the file is there so we fail fast).
#   2. bfbench prepare   -> downloads chips, builds mosaic, merges refs, samples.
#   3. bfbench run-prepared -> adapts, predicts, vectorises, filters.
#   4. bfbench eval      -> writes per-tile parquet.
#
# Exit codes:
#   0 ok; 2 missing checkpoint; non-zero propagated from bfbench subcommands.

set -euo pipefail

CONFIG="${BFBENCH_SMOKE_CONFIG:-configs/spacenet_vegas_smoke.yaml}"
CHECKPOINT="${BFBENCH_WEIGHTS:-weights/neat-fullxview-epoch75.pth}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT" >&2
    echo "       run scripts/fetch_weights.py on the host before `docker compose run`." >&2
    exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found at $CONFIG" >&2
    exit 2
fi

echo "[smoke] config:     $CONFIG"
echo "[smoke] checkpoint: $CHECKPOINT"

echo "[smoke] step 1/3: bfbench prepare"
bfbench prepare "$CONFIG"

echo "[smoke] step 2/3: bfbench run-prepared"
bfbench run-prepared "$CONFIG"

echo "[smoke] step 3/3: bfbench eval"
bfbench eval "$CONFIG"

# Resolve the eval parquet from the work_dir in the config and print it so
# the caller can grep for it. We parse the work_dir with the same python the
# pipeline uses to avoid shelling out to yq.
WORK_DIR="$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['run']['work_dir'])")"
echo "[smoke] done. eval parquet: ${WORK_DIR}/eval/results.parquet"
