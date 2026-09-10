# Environment notes

This file explains every deviation from upstream `automated-building-detection`'s
`requirements.txt` files inside `external/abd/`. Upstream is pinned to
`e1668a470b94252040f27d26098826c293fbb46d` (see repo root README).

## Constraints on this project's env

- **Target OS:** Windows 11 (miniconda3). Everything must install without WSL.
- **Target GPU:** NVIDIA RTX 40/50-series (Blackwell, sm_120). This forces
  PyTorch built with CUDA >= 12.8; earlier cu wheels do not include the
  sm_120 kernel and would fall back to CPU or crash at kernel launch.
- **Python:** 3.11. Upstream targets 3.7 (see upstream README "conda create
  --name abdenv python=3.7"). 3.11 is the newest interpreter for which
  every dependency below has a Windows wheel, and is inside the support
  window of `torch==2.11.0`.
- **GDAL / rasterio / geopandas / pyproj / fiona / shapely** come from
  `conda-forge`, never from pip on Windows. Pip wheels are unreliable for
  the GDAL native stack on this OS.

## Deviations from upstream `abd_model/requirements.txt`

Upstream pins are ~2020-vintage and do not have wheels for Python >= 3.10.
Where a bump was required, the smallest bump that installs cleanly on the
target env was chosen.

| Upstream           | Ours              | Why                                                                                            |
|--------------------|-------------------|------------------------------------------------------------------------------------------------|
| `numpy==1.19.1`    | `numpy==2.1.3`    | 1.19 has no cp311 wheels; rasterio 1.4 requires numpy>=2 on Windows.                            |
| `Pillow==7.2.0`    | `pillow==11.1.0`  | 7.2 has no cp311 wheels; API surface used (`Image.open`, `.save`) is unchanged.                 |
| `rasterio==1.1.5`  | `rasterio==1.4.3` | 1.1 predates conda-forge's current GDAL 3.10 ABI. 1.4.x is the current stable series.           |
| `Shapely>=1.7.0`   | `shapely==2.0.6`  | Shapely 2 is a hard requirement of geopandas 1.0. API used (`box`, `shape`) is stable across.   |
| `torch==1.6.0`     | `torch==2.11.0`   | 1.6 lacks CUDA 12.8 / sm_120 support entirely. State-dict format is backward compatible for the AlbuNet weights used here (`torch.load` reads the pinned checkpoint on 2.11 without shape changes). |
| `torchvision==0.7.0` | `torchvision==0.26.0` | Paired with torch 2.11.0.                                                                    |
| `albumentations==0.4.6` | `albumentations==1.4.24` | 0.4.6 has no cp311 wheels. The 1.x API used by upstream's `abd_model.da.rgb` is compatible with 1.4; we deliberately stop at 1.4 (not 2.x) because 2.0 dropped several classes upstream uses. |
| `mercantile==1.1.5` | (unchanged)      | Still current.                                                                                  |
| `supermercado==0.1.1` | `supermercado==0.2.0` | 0.1.1 has no cp311 wheels; 0.2 keeps the `uniontiles.union` and `burntiles.burn` API upstream calls. |
| `psycopg2-binary==2.8.5` | `psycopg2==2.9.10` (conda-forge) | Only installed because `abd_model.tiles` does an unconditional `import psycopg2` at module import time. No database is used by this benchmark. |
| `python-dotenv==0.15.0` | `python-dotenv==1.0.1` | Bumped; only used by upstream's Bing-Maps downloader path, which is not invoked. |
| `webcolors==1.11.1` | `webcolors==24.11.1` | Bumped; used by upstream palette generation.                                                |
| `toml==0.10.1`     | `toml==0.10.2`    | Bug-fix bump.                                                                                    |
| `pyGeoTile==1.0.6` | (unchanged)      | Only used by upstream's Bing-Maps downloader path; kept importable.                              |

Not carried over: `osmium==3.0.1` (upstream OSM helpers not invoked by
this benchmark) and `scikit-build` (upstream had it as a build shim, not
needed at runtime).

## Upstream patch: `abd_model.tools.vectorize` syntax error

The pinned upstream SHA has a straight-up Python syntax error at
`abd_model/src/abd_model/tools/vectorize.py:56`:

```python
transform = rasterio.transform.from_bounds((*mercantile.bounds(...)), W, H)
#                                          ^^^^^^^^^^^^^^^^^^^^^^^^^
# SyntaxError: cannot use starred expression here
```

Tested on Python 3.10/3.11/3.13 — all reject this. `git log` on the file
shows it has been broken since the first upstream commit (Nov 2020); the
authors likely only ever invoked `abd predict` and never triggered the
`abd_model.tools.__main__` code path that imports every tool at startup.

We do NOT modify the submodule (that stays pinned to the SHA recorded in
the root README). Instead, `patches/abd/0001-fix-vectorize-syntax.patch`
is applied on top by `scripts/apply_upstream_patches.py`, which is called
as part of the reproduce steps in `docs/reproduce.md`. The patch is a
one-character-class change: remove the extra parens so it becomes
`from_bounds(*mercantile.bounds(...), W, H)`.

If upstream ever fixes this, drop the patch and bump the pinned SHA.

## Windows-specific runtime constraint (predict step)

`abd_model.tools.predict.main` does:

```python
assert torch.distributed.is_nccl_available()
mp.spawn(gpu_worker, ...)   # each worker: dist.init_process_group("nccl", ...)
```

`torch.distributed.is_nccl_available()` returns `False` on native Windows
because PyTorch's Windows build ships without NCCL. This means `abd predict`
will crash at that assertion on this machine regardless of our env.

Fixes considered and rejected for Phase 1:

1. **Modify upstream to fall back to `gloo`.** Rejected: brief says do not
   touch upstream.
2. **Monkey-patch backend at subprocess startup.** Working but fragile:
   would also need to short-circuit `mp.spawn` for single-GPU.
3. **Docker Desktop with NVIDIA Container Toolkit.** Requires Docker, which
   was not installed on the target machine at Phase 1 time.

Chosen: run the **predict** step inside WSL2 (Ubuntu, Linux PyTorch has
NCCL) or on any Linux host. `bfbench verify` warns explicitly if NCCL is
unavailable. The remaining stages (fetch, adapt, vectorize, filter) run
natively on Windows.

## Re-pinning procedure

If you need to bump a dependency, do the bump in `environment.yml`, then:

```powershell
conda env remove -n bfbench
conda env create -f environment.yml
conda activate bfbench
pip install -e ./external/abd/abd_model ./external/abd/abd_utils -e .
bfbench verify
pytest -m "not gpu and not network"
```

If any of that fails, revert and add a row to the table above explaining
why the bump was not viable.
