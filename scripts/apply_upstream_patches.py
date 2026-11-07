"""Apply the small set of upstream patches in `patches/abd/` on top of the
pinned submodule SHA. Idempotent: skips patches already applied.

Rationale for each patch is in the patch file's header comment.

Two code paths:

    1. **git-apply** (preferred): used when `external/abd/` is a real
       git repo (= the normal `git submodule update --init` path on a
       dev machine). Uses `git apply --check --reverse` to detect
       already-applied patches.
    2. **patch(1) fallback**: used when `external/abd/` is NOT a usable
       git repo -- notably inside the Phase 0 Docker image, where
       `.git` is a gitlink pointing at the host's `.git/modules/...`
       path and so dangles inside the container. Uses
       `patch -p1 -R --dry-run` to detect already-applied patches.

Both paths produce identical working-tree state. The git path is a
tiny bit faster and gives better error messages; the patch(1) path is
portable to any environment where `patch` is on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMODULE = ROOT / "external" / "abd"
PATCH_DIR = ROOT / "patches" / "abd"


def _submodule_is_usable_git_repo() -> bool:
    """True iff `git -C external/abd rev-parse HEAD` works. False means
    the directory is there but not a usable repo (e.g. dangling
    gitlink inside Docker), so we fall back to `patch(1)`."""
    r = subprocess.run(
        ["git", "-C", str(SUBMODULE), "rev-parse", "HEAD"],
        capture_output=True,
    )
    return r.returncode == 0


def _submodule_head_sha() -> str:
    out = subprocess.check_output(
        ["git", "-C", str(SUBMODULE), "rev-parse", "HEAD"], text=True
    )
    return out.strip()


# ---------------------------------------------------------------------------
# Path 1: git apply
# ---------------------------------------------------------------------------


def _already_applied_git(patch: Path) -> bool:
    # `git apply --check --reverse` succeeds iff the reverse patch would
    # apply cleanly, i.e. the forward patch was already applied.
    r = subprocess.run(
        ["git", "-C", str(SUBMODULE), "apply", "--check", "--reverse", str(patch)],
        capture_output=True,
    )
    return r.returncode == 0


def _apply_git(patch: Path) -> None:
    r = subprocess.run(
        ["git", "-C", str(SUBMODULE), "apply", str(patch)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"failed to apply {patch.name} via git:\n"
            f"{r.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Path 2: patch(1) fallback
# ---------------------------------------------------------------------------


def _already_applied_patch(patch: Path) -> bool:
    """`patch -p1 -R --dry-run` succeeds iff the forward patch is
    already applied (reverse would apply cleanly)."""
    with patch.open("rb") as fp:
        r = subprocess.run(
            ["patch", "-p1", "-R", "--dry-run", "--silent"],
            cwd=str(SUBMODULE), stdin=fp,
            capture_output=True,
        )
    return r.returncode == 0


def _apply_patch(patch: Path) -> None:
    """Forward-apply with `patch -p1`. `--forward` makes it refuse to
    re-apply a patch that's already in; combined with the dry-run
    reverse check above, this gives git-apply-equivalent idempotence."""
    with patch.open("rb") as fp:
        r = subprocess.run(
            ["patch", "-p1", "--forward", "--silent"],
            cwd=str(SUBMODULE), stdin=fp,
            capture_output=True,
        )
    if r.returncode != 0:
        raise RuntimeError(
            f"failed to apply {patch.name} via patch(1):\n"
            f"{r.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    if not SUBMODULE.is_dir():
        print(
            f"ERROR: submodule not initialised at {SUBMODULE}. "
            "Run `git submodule update --init --recursive` first.",
            file=sys.stderr,
        )
        return 2

    patches = sorted(PATCH_DIR.glob("*.patch"))
    if not patches:
        print("no patches to apply")
        return 0

    use_git = _submodule_is_usable_git_repo()
    if use_git:
        print(f"submodule HEAD: {_submodule_head_sha()}")
        print("backend: git apply")
        already, apply = _already_applied_git, _apply_git
    else:
        if shutil.which("patch") is None:
            print(
                "ERROR: external/abd is not a usable git repo and "
                "`patch` is not on PATH. Install it (apt-get install "
                "patch) or run from a dev checkout with the submodule "
                "initialised.",
                file=sys.stderr,
            )
            return 2
        print(
            "WARN: external/abd is not a usable git repo (likely a "
            "dangling submodule gitlink); falling back to patch(1)."
        )
        print("backend: patch(1)")
        already, apply = _already_applied_patch, _apply_patch

    applied, skipped = 0, 0
    for p in patches:
        if already(p):
            print(f"skip (already applied): {p.name}")
            skipped += 1
            continue
        print(f"applying {p.name}")
        apply(p)
        applied += 1

    print(f"done ({applied} applied, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
