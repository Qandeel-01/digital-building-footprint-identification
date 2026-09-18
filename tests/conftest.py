"""Test-suite bootstrap.

Set GDAL_DATA / PROJ_LIB before rasterio imports on Windows. Without
these, rasterio.transform calls into GDAL and the missing PROJ files
cause a Windows fatal exception (0xc06d007f) mid-test, killing the
entire pytest run. Setting these once in a conftest is the least
invasive place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if os.name == "nt":
    _prefix = Path(sys.prefix)
    os.environ.setdefault("GDAL_DATA", str(_prefix / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(_prefix / "Library" / "share" / "proj"))
    _lib_bin = _prefix / "Library" / "bin"
    if _lib_bin.is_dir() and str(_lib_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_lib_bin) + os.pathsep + os.environ.get("PATH", "")
