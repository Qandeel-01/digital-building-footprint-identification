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

## Upstream patches applied on top of the pinned SHA

The submodule pointer stays exactly on the recorded SHA. The following
patch files live under `patches/abd/` and are applied on top by
`scripts/apply_upstream_patches.py` (called during setup and idempotent).
Every patch was necessary to get the pipeline to run at all on the target
env; each is documented in its own file header.

| Patch                                          | What it fixes                                                                                       |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `0001-fix-vectorize-syntax.patch`              | Python `SyntaxError` in `abd_model/tools/vectorize.py:56` (`(*x)` not valid). Present since upstream's first commit (2020). Blocks every `abd <subcommand>` because `__main__` imports every tool at startup. |
| `0002-fix-webcolors-css3-api.patch`            | `webcolors.CSS3_NAMES_TO_HEX` was removed in webcolors 1.13 (2023). Swap to `webcolors.name_to_hex(name)` which has existed since 1.0. Called from `abd_model.core.make_palette`, hit as soon as `abd predict` starts. |
| `0003-fix-tiles-from-dir-windows.patch`        | `tiles_from_dir` builds a regex from `os.path.join(root, ...)`. On Windows the backslashes are interpreted as escape sequences (`re.error: bad escape \\i`). Fix: `re.escape(root)` and match `[/\\]` between (z, x, y). No behaviour change on POSIX. |
| `0004-fix-filter-buildings-geopandas-api.patch`| `abd_utils/filter_buildings.py` uses `gpd.sjoin(..., op='intersects')` (removed in geopandas 1.0, kwarg renamed to `predicate` in 0.10) and `DataFrame.append` (removed in pandas 2.0, replaced by `pd.concat([...])`). Pure API swap. |

If upstream ever merges any of these, drop the corresponding patch and
bump the pinned SHA.

## Windows single-GPU predict shim (bypassing NCCL)

`abd_model.tools.predict.main` does an unconditional
`assert torch.distributed.is_nccl_available()` and launches DDP workers
via `mp.spawn`, using the NCCL backend. NCCL is not available in
PyTorch's Windows build (see verify output above), so `abd predict`
cannot run natively on the target machine.

`scripts/predict_local.py` is a single-GPU, non-distributed drop-in
replacement for the predict step. It **uses upstream's model class,
loader, palette maker, and tile writer unchanged** (`Albunet`, `SemSeg`,
`make_palette`, `tile_label_to_file`) — it just runs the DataLoader on
one device and skips the DDP scaffolding. There is no forked model code
in this project's tree.

The rest of the pipeline (`abd vectorize`, `filter-buildings`) still
runs through the upstream CLIs via subprocess as originally designed.

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
