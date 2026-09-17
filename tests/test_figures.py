"""Smoke tests for scripts/figures.py.

Verifies the synthetic placeholder path runs without matplotlib
raising, and that outputs are labeled as fake in the filename.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "figures.py"


def _run_script(argv: list[str]) -> None:
    """Run scripts/figures.py as a script with `argv`."""
    old_argv = sys.argv[:]
    sys.argv = ["figures.py", *argv]
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = old_argv


def test_figures_synthetic_mode_writes_all_four_placeholders(tmp_path):
    out = tmp_path / "figs"
    _run_script(["--synthetic", "--out", str(out)])
    expected = {
        "f1_vs_density_SYNTHETIC_PLACEHOLDER.png",
        "f1_vs_size_decile_SYNTHETIC_PLACEHOLDER.png",
        "merge_vs_density_SYNTHETIC_PLACEHOLDER.png",
        "qualitative_panels_SYNTHETIC_PLACEHOLDER.png",
    }
    got = {p.name for p in out.glob("*.png")}
    assert expected.issubset(got), (
        f"missing placeholder outputs. expected={expected}, got={got}"
    )


def test_figures_never_writes_unlabeled_output_in_synthetic_mode(tmp_path):
    """If any file in the output directory lacks the placeholder suffix
    while --synthetic was given, that is a bug -- someone could paste
    that image into a report thinking it was a result.
    """
    out = tmp_path / "figs"
    _run_script(["--synthetic", "--out", str(out)])
    unlabeled = [
        p for p in out.iterdir()
        if p.suffix == ".png" and "SYNTHETIC_PLACEHOLDER" not in p.name
    ]
    assert not unlabeled, (
        f"synthetic mode wrote UNLABELED output(s): {unlabeled}. "
        "This is exactly the mistake the labelling was supposed to prevent."
    )


def test_figures_real_mode_requires_results_file(tmp_path):
    with pytest.raises(SystemExit):
        _run_script(["--out", str(tmp_path / "figs")])
