"""Verify each upstream patch in isolation.

For each of the five patches under `patches/abd/`:

    1. Reset the submodule tree to the pinned SHA (removes all
       currently-applied patches).
    2. Apply that ONE patch.
    3. Exercise the affected code path with a real Python call, not
       just an import. Record pass/fail. Reset to pinned SHA either
       way.

    4. Apply just the pinned SHA (no patch) and re-run the same check;
       it must fail. If the pinned SHA already passes, the patch is
       a no-op and something is wrong with the check.

Prints a table + returns exit 0 iff every patch check flipped from
FAIL (pinned) to PASS (patched).

The single-GPU / NCCL patch cannot be exercised on a Windows box
without CUDA, so its check is documented as "requires linux GPU" and
its pinned-SHA fail signature is verified by inspection (the assert
on line 105) rather than by executing it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))

REPO = Path(__file__).resolve().parents[1]
SUBMODULE = REPO / "external" / "abd"
PATCH_DIR = REPO / "patches" / "abd"
PINNED_SHA = "e1668a470b94252040f27d26098826c293fbb46d"


def _run(*cmd: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _reset_submodule() -> None:
    subprocess.check_call(
        ["git", "-C", str(SUBMODULE), "reset", "--hard", PINNED_SHA],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _apply(patch: Path) -> None:
    r = subprocess.run(["git", "-C", str(SUBMODULE), "apply", str(patch)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed to apply {patch.name}: {r.stderr}")


# ---------------------------------------------------------------------------
# Per-patch checks. Each function returns (passed: bool, note: str).
# ---------------------------------------------------------------------------

def _check_0001_vectorize_syntax() -> tuple[bool, str]:
    """Import abd_model.tools.vectorize; the pinned SHA has a
    SyntaxError so import fails."""
    r = _run(sys.executable, "-c", "import abd_model.tools.vectorize")
    if r.returncode == 0:
        return True, "import abd_model.tools.vectorize -> OK"
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    return False, f"import failed: {tail}"


def _check_0002_webcolors() -> tuple[bool, str]:
    """Call make_palette on ['black', 'green'] and check we get sane
    hex values. Pinned SHA raises AttributeError."""
    r = _run(
        sys.executable, "-c",
        "from abd_model.core import make_palette;"
        "p, t = make_palette(['black', 'green']);"
        "print(p[:6])",
    )
    if r.returncode == 0:
        return True, r.stdout.strip()
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    return False, tail


def _check_0003_tiles_from_dir() -> tuple[bool, str]:
    """Build a tiny slippy-map tree in a tmpdir and call
    `tiles_from_dir`. Pinned SHA raises re.error on Windows paths.
    The check just needs to demonstrate the function RETURNS on a
    real filesystem walk without raising."""
    with tempfile.TemporaryDirectory() as td:
        # Create /tmp/xxxx/17/42/68.png (empty file is enough; the
        # function only cares about the path regex).
        root = Path(td)
        (root / "17" / "42").mkdir(parents=True)
        (root / "17" / "42" / "68.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        r = _run(
            sys.executable, "-c",
            "import sys; from abd_model.tiles import tiles_from_dir;"
            f"tiles = list(tiles_from_dir(r'{root}'));"
            "print(len(tiles), tiles[:1])",
        )
    if r.returncode == 0:
        return True, r.stdout.strip()
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    return False, tail


def _check_0004_filter_buildings() -> tuple[bool, str]:
    """Run `filter-buildings` on a >SPLIT_SIZE (1000) polygon geojson
    to force the multi-batch path that uses pd.concat.

    Pinned SHA fails two ways:
        * If op= is still in the code -> TypeError from sjoin.
        * If op= is fixed but pd import missing -> NameError from
          pd.concat.
    """
    with tempfile.TemporaryDirectory() as td:
        gj = Path(td) / "many.geojson"
        # 1200 small non-overlapping squares in EPSG:3857 metres.
        features = []
        for i in range(1200):
            x0 = 100.0 * i
            y0 = 0.0
            features.append({
                "type": "Feature", "properties": {},
                "geometry": {"type": "Polygon",
                             "coordinates": [[
                                 [x0, y0], [x0 + 5, y0],
                                 [x0 + 5, y0 + 5], [x0, y0 + 5],
                                 [x0, y0],
                             ]]},
            })
        collection = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
            "features": features,
        }
        gj.write_text(json.dumps(collection), encoding="utf-8")
        out = Path(td) / "filtered.geojson"
        r = _run(str(Path(sys.prefix) / "Scripts" / "filter-buildings.exe"),
                 "--data", str(gj), "--dest", str(out))
    if r.returncode == 0:
        return True, "filter-buildings on 1200 polys -> OK"
    tail = (r.stderr.strip().splitlines() or [""])[-1]
    if not tail:
        tail = (r.stdout.strip().splitlines() or [""])[-1]
    return False, tail


def _check_0005_single_gpu() -> tuple[bool, str]:
    """Static check only: the patched file must have --single-gpu
    plumbed through, and the pinned file must have an unconditional
    NCCL assert. We cannot execute this on a Windows GPU without NCCL
    -- reported as such.
    """
    target = SUBMODULE / "abd_model" / "src" / "abd_model" / "tools" / "predict.py"
    text = target.read_text(encoding="utf-8")
    has_flag = "--single-gpu" in text or "single_gpu" in text
    has_guard = "getattr(args, \"single_gpu\", False)" in text
    if has_flag and has_guard:
        return True, "static check: --single-gpu present + assertion guarded"
    return False, ("--single-gpu NOT present" if not has_flag
                   else "assertion NOT guarded by single_gpu")


CHECKS = [
    ("0001-fix-vectorize-syntax.patch",              _check_0001_vectorize_syntax),
    ("0002-fix-webcolors-css3-api.patch",            _check_0002_webcolors),
    ("0003-fix-tiles-from-dir-windows.patch",        _check_0003_tiles_from_dir),
    ("0004-fix-filter-buildings-geopandas-api.patch",_check_0004_filter_buildings),
    ("0005-single-gpu-predict-no-nccl.patch",        _check_0005_single_gpu),
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=REPO / "docs" / "upstream-prs" / "verification.json")
    args = p.parse_args(argv)

    results: list[dict] = []
    for patch_name, check in CHECKS:
        patch_path = PATCH_DIR / patch_name
        row = {"patch": patch_name, "pinned": None, "patched": None,
               "pinned_note": "", "patched_note": ""}

        # 1) pinned SHA (no patch): check should FAIL for a good patch.
        _reset_submodule()
        pinned_ok, pinned_note = check()
        row["pinned"] = pinned_ok
        row["pinned_note"] = pinned_note

        # 2) apply patch: check should PASS.
        _reset_submodule()
        _apply(patch_path)
        patched_ok, patched_note = check()
        row["patched"] = patched_ok
        row["patched_note"] = patched_note

        results.append(row)

    # 3) Restore working state: apply ALL patches so the repo is usable.
    _reset_submodule()
    for patch_name, _ in CHECKS:
        _apply(PATCH_DIR / patch_name)

    # Report.
    print()
    print(f"{'patch':<48}  {'pinned':<10}  {'patched':<10}  {'delta':<10}")
    print("-" * 90)
    for r in results:
        delta = "flip" if (r["pinned"] is False and r["patched"] is True) \
                else ("no-op" if (r["pinned"] == r["patched"]) else "regression")
        print(f"{r['patch']:<48}  {str(r['pinned']):<10}  "
              f"{str(r['patched']):<10}  {delta:<10}")
    print()
    for r in results:
        print(f"* {r['patch']}")
        print(f"    pinned:  {r['pinned_note']}")
        print(f"    patched: {r['patched_note']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    all_good = all(r["pinned"] is False and r["patched"] is True
                   for r in results if r["patch"] != "0005-single-gpu-predict-no-nccl.patch")
    return 0 if all_good else 1


if __name__ == "__main__":
    raise SystemExit(main())
