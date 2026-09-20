"""Integration test: generate the real-mode qualitative panels PNG.

This test doubles as the way to actually build the figure on Windows:
matplotlib.savefig segfaults from bare `python script.py` on this
env (delay-load DLL 0xC06D007F), but works cleanly under pytest.
Running `pytest tests/test_qualitative_panels_integration.py` produces
`runs/vegas_prepare/eval/qualitative_panels.png` as a side effect.

Skipped when the Vegas eval Parquet is absent (no CI cost).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "runs" / "vegas_prepare" / "eval" / "results.parquet"
CFG     = REPO / "configs" / "spacenet_vegas.yaml"
OUT     = REPO / "runs" / "vegas_prepare" / "eval" / "qualitative_panels.png"


pytestmark = pytest.mark.skipif(
    not PARQUET.is_file(),
    reason="Vegas eval parquet not present; run `bfbench eval` first.",
)


def test_generate_qualitative_panels_png(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    from qualitative_panels import build_figure  # noqa: PLC0415
    build_figure(PARQUET, CFG, OUT)
    assert OUT.is_file()
    assert OUT.stat().st_size > 5_000
