"""Test-suite bootstrap.

Two Windows-only native-layer workarounds, both scoped to the test
harness (never applied in production code), both no-ops on Linux:

1. **GDAL_DATA / PROJ_LIB**. rasterio.transform calls into GDAL; on
   Windows these env vars need to point at the conda env's data
   directories or GDAL raises a Windows fatal exception (0xc06d007f)
   mid-test and the whole pytest run dies.

2. **KMP_DUPLICATE_LIB_OK**. Phase 1 adds torch to the dependency set.
   torch (via MKL) and GDAL/rasterio each ship their own Intel
   OpenMP runtime on Windows; when both are loaded in the same
   process (which happens as soon as a Phase 1 test imports torch
   AND a Phase 3 test imports rasterio), the two libomp copies
   collide at shutdown and pytest exits with `Fatal Python error:
   Aborted` inside rasterio. KMP_DUPLICATE_LIB_OK=TRUE tells Intel
   OpenMP to tolerate a second copy. See
   docs/environment-notes.md for the longer story.

Setting these once in a conftest is the least invasive place -- the
module runs before any test module is imported, so the env vars are
in place before torch/rasterio's native layers load.
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

    # torch (MKL) + GDAL (bundled libomp) collision; see module docstring.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
