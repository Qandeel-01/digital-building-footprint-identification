"""Phase 0 reproducibility: static checks on Dockerfile + .dockerignore.

These are not container-build tests (that would need Docker on the CI
runner and 2 GB of pulls). They assert invariants the Dockerfile MUST
preserve so the image stays reproducible:

- the CUDA base image matches the sm_120 story in environment-notes.md,
- the conda env comes from environment.yml rather than duplicated pins,
- upstream patches are applied BEFORE abd_model / abd_utils are installed
  (otherwise the editable install picks up unpatched source),
- the image runs as a non-root user,
- external/abd/.git is NOT excluded from the build context (the patch
  applier shells out to `git -C external/abd apply`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.is_file(), f"missing Dockerfile at {DOCKERFILE}"
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore_text() -> str:
    assert DOCKERIGNORE.is_file(), f"missing .dockerignore at {DOCKERIGNORE}"
    return DOCKERIGNORE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Base image: must be a CUDA 12.8+ runtime image. sm_120 (Blackwell) needs
# CUDA 12.8; earlier tags silently fall back to CPU or crash at kernel launch.
# ---------------------------------------------------------------------------

def test_base_image_is_cuda_12_8_or_newer(dockerfile_text: str) -> None:
    m = re.search(
        r"nvidia/cuda:(\d+)\.(\d+)\.\d+-(?:cudnn-)?runtime-ubuntu\d+\.\d+",
        dockerfile_text,
    )
    assert m, "Dockerfile does not FROM an nvidia/cuda:*-runtime-ubuntu* image"
    major, minor = int(m.group(1)), int(m.group(2))
    assert (major, minor) >= (12, 8), (
        f"CUDA base is {major}.{minor}; sm_120 requires >= 12.8 "
        "(see docs/environment-notes.md)"
    )


def test_base_image_is_runtime_not_devel(dockerfile_text: str) -> None:
    assert "runtime-ubuntu" in dockerfile_text, (
        "expected a *-runtime-* base image; devel images are much bigger and "
        "we don't compile CUDA kernels in this project"
    )
    assert "devel-ubuntu" not in dockerfile_text


# ---------------------------------------------------------------------------
# Env comes from environment.yml -- no duplicated pins in the Dockerfile.
# ---------------------------------------------------------------------------

def test_env_created_from_environment_yml(dockerfile_text: str) -> None:
    assert "COPY environment.yml" in dockerfile_text
    assert "conda env create -f /workspace/environment.yml" in dockerfile_text


# ---------------------------------------------------------------------------
# Patch applier runs before abd_model / abd_utils are installed. Otherwise
# the editable install registers the unpatched source and the patches never
# take effect at runtime.
# ---------------------------------------------------------------------------

def test_patches_applied_before_abd_install(dockerfile_text: str) -> None:
    apply_idx = dockerfile_text.find("apply_upstream_patches.py")
    abd_install_idx = dockerfile_text.find("external/abd/abd_model")
    assert apply_idx != -1, "apply_upstream_patches.py is never invoked"
    assert abd_install_idx != -1, "abd_model is never pip-installed"
    assert apply_idx < abd_install_idx, (
        "patches must be applied BEFORE pip install -e external/abd/abd_model; "
        "otherwise upstream source is installed un-patched"
    )


def test_abd_installed_before_bfbench(dockerfile_text: str) -> None:
    """abd_model + abd_utils must be installed before bfbench so that
    bfbench's editable install sees upstream already registered."""
    abd_idx = dockerfile_text.find("external/abd/abd_model")
    bfbench_idx = dockerfile_text.find("pip install --no-deps -e /workspace\n")
    assert abd_idx != -1 and bfbench_idx != -1
    assert abd_idx < bfbench_idx


# ---------------------------------------------------------------------------
# Non-root runtime user. Container should not run as root by default; weight
# downloads and parquet writes land on bind mounts and would otherwise be
# owned by root on the host.
# ---------------------------------------------------------------------------

def test_image_runs_as_non_root(dockerfile_text: str) -> None:
    user_lines = [
        ln.strip() for ln in dockerfile_text.splitlines() if ln.strip().startswith("USER ")
    ]
    assert user_lines, "Dockerfile does not switch away from root"
    final_user = user_lines[-1].split(maxsplit=1)[1].strip()
    assert final_user not in {"root", "0"}, (
        f"final USER is {final_user!r}; must be non-root"
    )


# ---------------------------------------------------------------------------
# external/abd/.git must stay in the context: the applier uses `git apply`.
# ---------------------------------------------------------------------------

def test_dockerignore_does_not_exclude_submodule_git(dockerignore_text: str) -> None:
    # It's fine to ignore the top-level .git/ (we never need it in the image);
    # but any pattern that also matches external/abd/.git would break the
    # patch applier. The simplest safe rule: no bare `**/.git` or `*.git`.
    offenders = []
    for raw in dockerignore_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("!"):
            continue
        if line in {"**/.git", "**/.git/", "*.git", "*.git/"}:
            offenders.append(raw)
        if line.startswith("external/abd"):
            offenders.append(raw)
    assert not offenders, (
        "these .dockerignore patterns would strip external/abd/.git, "
        f"breaking the patch applier: {offenders!r}"
    )


# ---------------------------------------------------------------------------
# Generated runtime artifacts must NOT be in the build context.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["weights/", "runs/", "data/", "output/"])
def test_dockerignore_excludes_runtime_dirs(dockerignore_text: str, path: str) -> None:
    patterns = {
        ln.split("#", 1)[0].strip() for ln in dockerignore_text.splitlines()
    }
    assert path in patterns, f".dockerignore does not exclude {path!r}"
