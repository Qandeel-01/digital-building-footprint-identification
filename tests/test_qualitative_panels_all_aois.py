"""Generate qualitative_panels.png for each AOI whose eval parquet is present.

Also doubles as figure generator (bare `python scripts/qualitative_panels.py`
segfaults matplotlib on this Windows env; pytest works).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AOIS = [
    ("configs/spacenet_vegas.yaml",    "runs/vegas_prepare"),
    ("configs/spacenet_khartoum.yaml", "runs/khartoum_prepare"),
    ("configs/spacenet_shanghai.yaml", "runs/shanghai_prepare"),
    ("configs/spacenet_paris.yaml",    "runs/paris_prepare"),
]


@pytest.mark.parametrize("cfg_rel,work_rel", AOIS,
                         ids=[c for c, _ in AOIS])
def test_generate_qualitative_panels(cfg_rel: str, work_rel: str):
    cfg = REPO / cfg_rel
    work = REPO / work_rel
    parquet = work / "eval" / "results.parquet"
    if not parquet.is_file():
        pytest.skip(f"{parquet} missing")

    sys.path.insert(0, str(REPO / "scripts"))
    from qualitative_panels import build_figure  # noqa: PLC0415
    out = work / "eval" / "qualitative_panels.png"
    build_figure(parquet, cfg, out)
    assert out.is_file()
    assert out.stat().st_size > 5_000
