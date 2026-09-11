# Phase 1 diagnosis: 6 predicted vs 31 reference polygons

At the end of Phase 1 the sample run on SpaceNet AOI_2_Vegas img101
produced 6 filtered predicted polygons against a 31-polygon SpaceNet
reference. This document diagnoses that gap. Reproducer:
`python scripts/diagnose_phase1.py`. Raw JSON at
`runs/spacenet_vegas_sample/diagnosis/summary.json`; PNGs alongside.

## Finding 1: 6 vs 31 was the wrong comparison

The SpaceNet reference file is per-chip. The prediction ran on one
z=17 slippy-map tile whose footprint only partially overlaps the chip.
The right denominator is the reference polygons whose centroid lies
inside the *tile* footprint, not the chip footprint.

| Denominator                                              | Value |
|----------------------------------------------------------|-------|
| Reference polygons with centroid inside the chip         | 31    |
| Reference polygons with centroid inside the tile (z=17)  | **12** |
| Predicted polygons (post filter-buildings)               | 6     |

The Phase 1 QA figure titles read "predicted (6 polys)" and "SpaceNet
reference (31 polys)". The 31 there was misleading. The honest first-
order recall estimate on this tile is **6/12 ≈ 0.5**. This is not a
metric — Phase 2 will compute one — but it removes the impression of
a ~5x shortfall.

## Finding 2: 73% of the tile is nodata

The SpaceNet Vegas chip is ~200 m on a side. One z=17 tile at
lat 36 spans ~247 m. The chip's own bbox is
`W=-115.3058, S=36.2055, E=-115.3040, N=36.2073` vs the tile bbox
`W=-115.3070, S=36.2044, E=-115.3043, N=36.2066`. Only the intersection
carries real imagery; the rest is zero-padded by the adapter.

```
nodata (all-zero pixel) fraction: 0.7331   (192169 / 262144 pixels)
valid pixels: 26.7%   (69975 / 262144)
```

This is the root cause of the low absolute prediction count and it is
a **coverage** issue, not a per-pixel accuracy issue. The Phase 2 data
layer switches to a per-AOI mosaic of contiguous chips so that a tile
is either fully covered or excluded (see Task 1b).

## Finding 3: the model is confident, not mushy

Raw sigmoid probability from AlbuNet on this tile, computed on **valid
(non-nodata) pixels only** (`docs` histogram
`runs/spacenet_vegas_sample/diagnosis/prob_hist_valid.png`):

| Statistic                              | Value  |
|----------------------------------------|--------|
| mean                                   | 0.178  |
| p50                                    | 0.006  |
| p90                                    | 0.991  |
| p99                                    | 1.001  |
| fraction of valid pixels > 0.5         | 16.8 % |
| fraction of valid pixels in [0.3, 0.7] | **1.5 %** |

Only 1.5 % of valid pixels sit in the uncertainty band. The output is
essentially bimodal — the model is committing to background or
building at pixel granularity. This is the "confident zeros" case
Phase 2's task 0b anticipated, i.e. a domain gap rather than input
mush. That framing is *directional*, not conclusive; a proper answer
needs a labelled test set and P/R at multiple thresholds, which is
Phase 2 metrics work.

## Finding 4: masking nodata at output does not change anything

To distinguish "the model hallucinates buildings on zero-padded regions"
(input problem) from "the model correctly ignores zero-padded regions"
(nothing to fix), the diagnostic script also re-thresholds with the
nodata mask applied at output time. Result:

```
positive-pixel counts after threshold >= 0.5:
  nodata zero-filled (Phase 1 behaviour):  11771
  nodata masked at output:                 11771
```

They match exactly. The model already outputs near-zero on zero-pad
regions (p50 of raw prob on ALL pixels including nodata is 0.006, and
the ALL-pixel mean is 0.052). Zero-padding is inert for this model —
the fix belongs at the coverage layer (per-AOI mosaic), not at the
prediction layer.

## What the evidence supports

The 6-vs-31 gap decomposes into:

1. **A reporting error** — 31 was the wrong denominator, 12 is right.
2. **A coverage-driven under-count** — 73 % of the tile is nodata; the
   model only has 27 % of the tile to work with.
3. **Not** an input contamination bug — nodata-masked vs zero-filled
   are identical.
4. **Possibly** a domain gap (Vegas/xBD radiometry vs SpaceNet PS-RGB
   stretching), but this is beyond Phase 1's evidence to call. Phase 2
   metrics on more AOIs and reference-matched pairs will confirm or
   refute.

## What we're fixing before Task 1

- **Task 1b** — per-AOI mosaic tiling, tiles fully covered or excluded.
  Kills the coverage problem globally.
- **Task 1c** — reference merged across chips + deduped on seams. Kills
  the per-chip reference confusion that produced the wrong denominator
  above.
- The prediction path itself is left alone; the diagnosis found no
  input bug to fix in it.
