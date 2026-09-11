# Adapter decisions

The adapter converts SpaceNet 2 PS-RGB GeoTIFFs (native EPSG:32611/32633/etc.,
~30 cm/pixel) into the XYZ slippy-map tile tree that `abd predict`
consumes. Two decisions dominate the benchmark and must be defensible;
they are both configurable knobs on `AdapterConfig` in
`src/bfbench/data/adapter.py`.

## 1. GSD resampling: 0.30 m -> 0.50 m

**Decision.** The default `AdapterConfig.target_gsd_m = 0.5`. All imagery
is reprojected to EPSG:3857 at 0.5 m/pixel *before* tiling.

**Why.**

- The upstream AlbuNet checkpoint (`neat-fullxview-epoch75`) was trained
  on xBD imagery at ~0.5 m/pixel. The convolutional feature stack has an
  effective receptive field expressed in *pixels*; feeding it 0.3 m
  imagery makes every feature scale ~1.67x smaller in world units than
  the network was trained to detect. Empirically that produces
  over-fragmentation of large buildings and missed detection of small
  ones; more importantly, "how many buildings did we find" now depends
  on the input GSD in a way we cannot cleanly attribute to detector
  quality.
- Resampling downward (30 cm -> 50 cm) is done with cubic resampling.
  Cubic was chosen over nearest for continuity (predictions shouldn't
  see aliasing patterns), and over Lanczos to save a small amount of
  edge halo that the AlbuNet was not trained against.
- A percentile stretch (p2/p98) is applied when writing tiles because
  SpaceNet PS-RGB is 16-bit uint and the model expects 8-bit uint RGB.
  Fixed percentile stretch (not per-tile min/max) is deliberate: it
  keeps radiometry comparable across tiles from the same AOI.

**Alternatives considered.**

- **No resampling, predict at native 0.30 m.** Rejected: silently
  changes the effective RF; the benchmark stops being about the model
  and starts being about GSD.
- **Resample to the exact GSD of the closest XYZ zoom (which at z=18,
  lat=36 is ~0.60 m).** This is what falls out if you set
  `target_gsd_m=0.5, zoom_tolerance=0.20`. Documented and accepted; the
  0.10 m residual is small compared to the 30 cm -> 50 cm training
  mismatch it is being applied to fix.

**Anti-goal.** The adapter must never *upsample*. If a future dataset
comes in at 1 m/pixel we will not synthesise 0.5 m pixels; we will
reject the input.

### GSD sensitivity — empirical validation (Phase 2 Task 0c)

The Phase 1 decision is above defended by argument (matches xBD
training GSD). Task 0c validates it empirically by running the same
Vegas chip through the pipeline at three zooms while holding
`tile_size=512` and everything else fixed. Reproducer:
`python scripts/gsd_sensitivity.py`. Panel:
`runs/gsd_sensitivity/gsd_panel.png`, counts in `results.json`.

| Zoom | Per-pixel GSD | Tiles written | Filtered polygons |
|------|---------------|---------------|-------------------|
| 16   | 0.964 m       | 1             | 10                |
| 17   | 0.482 m       | 1             | **6** (default)   |
| 18   | 0.241 m       | 4             | 20                |

Reference (31 total on chip, 12 with centroid in the z=17 tile) is a
shared constant.

Interpretation, staying descriptive:

- The count more than triples between z=17 and z=18 — the detector's
  behaviour is *not* GSD-invariant on this material. Whatever
  metric is chosen in Phase 2 must be reported jointly with the
  effective GSD used.
- At z=16 (2x coarser than training) the model emits fewer polygons,
  and several are elongated / clearly merged neighbours — consistent
  with the receptive-field-in-pixels intuition.
- At z=18 (2x finer than training) the model emits ~2x the polygons,
  including plausible false positives (blob-like fragments in road
  areas visible in the panel). Consistent with the same intuition
  going the other way.
- z=17 minimises visual over- and under-fragmentation in this single
  chip, which is consistent with picking the training GSD. This is
  n=1; a Phase-2 sweep on 150 tiles per AOI is the real test.

The panel `gsd_panel.png` is committed as an artifact of this run
(gitignored under `runs/`); regenerate with the script above.

**Anti-goal restated.** Changing the input GSD changes the detector's
answer significantly. Any comparison of building-detection quality
across regions must fix the input GSD first, or report GSD alongside
every metric.

## 2. Tile size and border handling

**Decision.** `tile_size = 512`, `keep_borders = True` by default.

**Why 512.**

The pinned AlbuNet checkpoint records `shape_in = (C, 512, 512)`.
Anything else would either force a resize inside the predict loop
(silently changing the effective GSD again) or crash the network. This
is a property of the checkpoint, not a knob to tune.

**Why the XYZ tile pyramid.**

Upstream `abd predict` reads tiles from a slippy-map directory tree of
`mercantile.Tile(x, y, z)` and computes lat/lon for each mask via
`mercantile.bounds()` during vectorization. If we lay out tiles any other
way, `abd vectorize` produces a GeoJSON with wrong coordinates.

Our adapter picks the zoom whose native 256-pixel Web-Mercator tile GSD is
closest to `target_gsd_m`, but writes tiles at `tile_size = 512`. That is
consistent with the loader: the loader keys tiles by their `(x, y, z)`
coordinates (which are GSD-independent), not by their pixel size.

**Why `keep_borders=True` by default.**

`abd predict --metatiles --keep_borders` runs the model on every tile
including those at the AOI boundary. With `--metatiles` alone, upstream's
`tile_is_neighboured` filter drops any tile lacking all 8 neighbours on
disk. For a small AOI (one SpaceNet chip) this can drop *every* tile,
producing an empty prediction and an unhelpful "empty dataset" abort.

The adapter writes surrounding neighbour tiles into the same tree so
`--metatiles` has neighbours to read; `keep_borders=True` then keeps
edge tiles for prediction (they still get a metatile buffer, they just
lose some neighbours to zero-padding).

**Alternatives considered.**

- **`keep_borders=False`, drop border tiles.** Loses ~1 ring of tiles
  per AOI. Fine for large AOIs, terrible for single-chip runs.
- **Do not use metatiles at all.** Reduces edge artifacts sharply near
  building boundaries that fall across a tile edge; upstream's own docs
  recommend `--metatiles` for inference.

## What is *not* configurable

- The tile-tree directory schema (`<root>/<channel_name>/<z>/<x>/<y>.<ext>`)
  and the `cover.csv` layout are dictated by upstream and reproduced
  exactly.
- The `channel_name` defaults to `images` because upstream's `input/config.toml`
  hardcodes the same string. Changing it here without editing the abd
  config file will make predict fail to find any tiles.

## Notes on upstream we noticed but did not fix

- `abd_utils/src/abd_utils/tiles.py::TileCollection.deg2tile` is called
  with axis order `(lon, lat)` from `generate_tiles` while the function
  body is written for `(lat, lon)`. This affects upstream's Bing-Maps
  downloader path but not our adapter (which uses `mercantile` directly).
  Left unfixed per the "do not modify upstream" rule.
