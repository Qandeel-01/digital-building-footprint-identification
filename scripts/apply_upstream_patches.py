"""Apply the small set of upstream patches in `patches/abd/` on top of the
pinned submodule SHA. Idempotent: skips patches already applied.

Rationale for each patch is in the patch file's header comment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMODULE = ROOT / "external" / "abd"
PATCH_DIR = ROOT / "patches" / "abd"


def _submodule_head_sha() -> str:
    out = subprocess.check_output(
        ["git", "-C", str(SUBMODULE), "rev-parse", "HEAD"], text=True
    )
    return out.strip()


def _already_applied(patch: Path) -> bool:
    # `git apply --check --reverse` succeeds iff the reverse patch would
    # apply cleanly, i.e. the forward patch was already applied.
    r = subprocess.run(
        ["git", "-C", str(SUBMODULE), "apply", "--check", "--reverse", str(patch)],
        capture_output=True,
    )
    return r.returncode == 0


def _apply(patch: Path) -> None:
    r = subprocess.run(
        ["git", "-C", str(SUBMODULE), "apply", str(patch)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"failed to apply {patch.name}:\n{r.stderr.decode(errors='replace')}"
        )


def main() -> int:
    if not SUBMODULE.is_dir():
        print(
            f"ERROR: submodule not initialised at {SUBMODULE}. "
            "Run `git submodule update --init --recursive` first.",
            file=sys.stderr,
        )
        return 2

    sha = _submodule_head_sha()
    print(f"submodule HEAD: {sha}")

    patches = sorted(PATCH_DIR.glob("*.patch"))
    if not patches:
        print("no patches to apply")
        return 0

    applied, skipped = 0, 0
    for p in patches:
        if _already_applied(p):
            print(f"skip (already applied): {p.name}")
            skipped += 1
            continue
        print(f"applying {p.name}")
        _apply(p)
        applied += 1

    print(f"done ({applied} applied, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
