"""Phase 0: static checks on the smoke-run surface (compose + Makefile +
entrypoint script).

Covers only what makes the one-AOI end-to-end target reproducible from a
clean clone:

- docker-compose.yml points at an existing smoke config,
- Makefile exposes the targets the brief and README reference,
- scripts/docker_smoke.sh is executable from bash and invokes the three
  bfbench stages in order.

Behavioural tests (actually running `docker compose up`) require Docker
and a GPU and are explicitly out of scope here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE = REPO_ROOT / "docker-compose.yml"
MAKEFILE = REPO_ROOT / "Makefile"
SMOKE_SH = REPO_ROOT / "scripts" / "docker_smoke.sh"
SMOKE_CFG = REPO_ROOT / "configs" / "spacenet_vegas_smoke.yaml"


# ---------------------------------------------------------------------------
# docker-compose
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compose_doc() -> dict:
    assert COMPOSE.is_file(), f"missing {COMPOSE}"
    with COMPOSE.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_compose_defines_smoke_service(compose_doc: dict) -> None:
    assert "services" in compose_doc
    assert "bfbench-smoke" in compose_doc["services"], (
        f"compose services: {list(compose_doc['services'].keys())}"
    )


def test_compose_references_existing_smoke_config(compose_doc: dict) -> None:
    env = compose_doc["services"]["bfbench-smoke"].get("environment", {}) or {}
    cfg_path = env.get("BFBENCH_SMOKE_CONFIG")
    assert cfg_path, "compose does not set BFBENCH_SMOKE_CONFIG"
    assert (REPO_ROOT / cfg_path).is_file(), (
        f"compose points at {cfg_path!r} which does not exist on disk"
    )
    assert (REPO_ROOT / cfg_path) == SMOKE_CFG, (
        "smoke should run against the committed smoke config, not an ad-hoc one"
    )


def test_compose_reserves_gpu(compose_doc: dict) -> None:
    svc = compose_doc["services"]["bfbench-smoke"]
    try:
        devices = svc["deploy"]["resources"]["reservations"]["devices"]
    except KeyError as e:
        pytest.fail(f"compose does not declare a GPU reservation: missing {e}")
    assert any(d.get("driver") == "nvidia" for d in devices), (
        f"no nvidia-driver device reservation: {devices!r}"
    )


def test_compose_bind_mounts_persist_artifacts(compose_doc: dict) -> None:
    volumes = compose_doc["services"]["bfbench-smoke"].get("volumes", [])
    mounted_targets = {v.split(":")[1] for v in volumes if ":" in v}
    assert "/workspace/weights" in mounted_targets, (
        "weights/ must be bind-mounted so the checkpoint persists across runs"
    )
    assert "/workspace/runs" in mounted_targets, (
        "runs/ must be bind-mounted so eval parquet is visible on the host"
    )


# ---------------------------------------------------------------------------
# Makefile
# ---------------------------------------------------------------------------

def test_makefile_exposes_expected_targets() -> None:
    assert MAKEFILE.is_file(), f"missing {MAKEFILE}"
    text = MAKEFILE.read_text(encoding="utf-8")
    for target in ("docker-build:", "docker-smoke:", "patches:", "lint:", "test:"):
        assert target in text, f"Makefile missing target {target!r}"


# ---------------------------------------------------------------------------
# scripts/docker_smoke.sh
# ---------------------------------------------------------------------------

def test_smoke_script_present_and_runs_three_bfbench_stages() -> None:
    assert SMOKE_SH.is_file(), f"missing {SMOKE_SH}"
    text = SMOKE_SH.read_text(encoding="utf-8")
    # Must invoke exactly the three stages, in order.
    prepare_idx = text.find("bfbench prepare")
    run_prep_idx = text.find("bfbench run-prepared")
    eval_idx = text.find("bfbench eval")
    assert -1 < prepare_idx < run_prep_idx < eval_idx, (
        "smoke script does not invoke prepare -> run-prepared -> eval in order "
        f"(indices: {prepare_idx}, {run_prep_idx}, {eval_idx})"
    )


def test_smoke_script_fails_fast_without_checkpoint() -> None:
    text = SMOKE_SH.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text, "smoke script must enable strict bash mode"
    # Must check the checkpoint path and exit non-zero if missing. The exact
    # path is BFBENCH_WEIGHTS or the default under weights/.
    assert "neat-fullxview-epoch75.pth" in text or "BFBENCH_WEIGHTS" in text
    assert "exit 2" in text, (
        "smoke script should fail fast (exit code 2) when the checkpoint "
        "is missing, not fall through into a GPU run that will OOM later"
    )
