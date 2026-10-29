# Thin convenience wrappers around the Phase 0 reproducibility surface.
# Nothing here hides semantics; every target echoes the underlying command.
#
# Run `make help` for a one-screen summary.

SHELL := /bin/bash

IMAGE ?= bfbench:latest
SMOKE_CONFIG ?= configs/spacenet_vegas_smoke.yaml

.PHONY: help docker-build docker-smoke patches smoke lint test ci

help:  ## list targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

docker-build:  ## build the reproducible image (no GPU needed)
	docker build -t $(IMAGE) .

docker-smoke:  ## run one-AOI prepare->run-prepared->eval inside the image (needs GPU + weights/)
	docker compose run --rm bfbench-smoke

patches:  ## apply upstream patches on the host submodule (idempotent)
	python scripts/apply_upstream_patches.py

smoke:  ## one-AOI smoke run on the host (needs activated conda env + GPU + weights/)
	bfbench prepare $(SMOKE_CONFIG)
	bfbench run-prepared $(SMOKE_CONFIG)
	bfbench eval $(SMOKE_CONFIG)

lint:  ## ruff
	ruff check src tests scripts

test:  ## pytest, no gpu/network
	pytest -m "not gpu and not network" -q

ci: lint test  ## what CI runs: lint + non-gpu/non-network pytest
