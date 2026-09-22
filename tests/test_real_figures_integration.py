"""Integration test that also serves as the actual figure generator.

Bare `python scripts/real_figures.py ...` segfaults matplotlib on
this Windows/GDAL env (delay-load DLL 0xC06D007F). Running under
pytest sets up the interpreter differently and the same code works.
This test builds every real-mode figure into docs/results-figures/
whenever the four AOI parquets are on disk.

Skipped unless every parquet is present, so CI never depends on the
whole benchmark having been run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PARQUETS = [
    REPO / "runs" / "vegas_prepare"    / "eval" / "results.parquet",
    REPO / "runs" / "khartoum_prepare" / "eval" / "results.parquet",
    REPO / "runs" / "shanghai_prepare" / "eval" / "results.parquet",
    REPO / "runs" / "paris_prepare"    / "eval" / "results.parquet",
]

pytestmark = pytest.mark.skipif(
    not all(p.is_file() for p in PARQUETS),
    reason="Not all four AOI parquets are present.",
)


def test_generate_real_figures_all_aois():
    sys.path.insert(0, str(REPO / "scripts"))
    from real_figures import build_all_figures  # noqa: PLC0415
    out_dir = REPO / "docs" / "results-figures"
    outs = build_all_figures(PARQUETS, out_dir)
    for o in outs:
        assert o.is_file()
        assert o.stat().st_size > 5_000
