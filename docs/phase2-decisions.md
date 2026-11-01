# Phase 2 — Separating touching buildings: decisions

The merge-rate figure for the baseline rises with building density
(see `docs/results-figures/merge_rate_vs_density_w200.png`). The
model's dominant failure in dense regions is **merging adjacent
buildings into one polygon**. Phase 2 directly targets this with two
independent approaches.

Both are implemented behind config flags; neither replaces the
other. The results overlay (`scripts/merge_rate_overlay.py`) draws
all four series (baseline / Phase 1 / 2a / 2b) on one figure so the
contribution of each change is readable.

## Phase 2a — 3-class target

### What is "the border class"?

Each pixel is one of:

    0  background
    1  interior         (inside a building)
    2  border contact   (inside a building AND within
                        `border_distance_m` of SOME OTHER building's
                        boundary)

**Border is a touching-pair feature only.** Isolated buildings
(nearest-neighbour polygon farther away than `border_distance_m`)
contain no class-2 pixels; their interior remains class 1 to the
polygon boundary. The point is for the model to learn edges that
*separate* adjacent buildings, not generic building edges.

The classic U-Net "border weighting" trick (uniform edge ring on
every building) is an alternative. Rejected for Phase 2 because the
brief phrasing is specifically **"separate touching buildings"** —
giving isolated buildings a border ring teaches the model to drop
edge pixels across the board, which costs recall on single-building
tiles without a corresponding gain on touching pairs.

### `border_distance_m` default (3 m) — rationale

- 1 m is below the native SpaceNet 2 chip GSD (~0.3 m) adapted to
  the model's 0.5 m/px — border strips at that scale are 1-2 px
  wide and the model's receptive field cannot localise them.
- 5 m consumes more of a mid-sized single-family house's interior
  than we want; on Vegas (~10 m house footprints) that is one third
  of the interior area labelled as border.
- 3 m is the Goldilocks: ~6 px wide at 0.5 m/px, enough signal to
  learn without eating the interior.

Overridable in each `configs/spacenet_<aoi>_phase2a.yaml` under
`train.border_distance_m`.

### Model-side change

- `BuildingTileDataset(target_mode="3class")` emits
  `(H, W) int64` class-index targets.
- `cross_entropy_3class_loss` replaces `bce_dice_loss`.
- `rebuild_final_conv_for_three_class` swaps Albunet's 1-channel
  final conv for a 3-channel one at training-run start, **warm-
  starting the interior channel from the pretrained binary filter**.
  The background and border channels get Kaiming init. Everything
  before the final conv is unchanged, so the fine-tune stays cheap.
- The saved checkpoint declares `shape_out = (3, 512, 512)`.
  `bfbench run-prepared` reads that key and (when
  `postprocess.method: argmax_interior`) carves class 2 out of class
  1 before invoking `abd vectorize`.

### Why `cross_entropy_3class_loss` and not Focal?

CE is the default partner for softmax. Focal would help if class 2
were severely imbalanced (which it is: typically <5 % of pixels),
but Focal + Dice on 3 classes is a measurably more fragile training
dynamic and comes with its own hyperparameter (`gamma`) that we would
have to tune on the test split, which is explicitly banned by the
brief. CE with optional class weights (`cross_entropy_3class_loss`
accepts `class_weights=`) is the honest default; **the per-AOI
configs leave weights uniform** and the per-tile loss curve goes in
`training_manifest.json` so a future revision can tune on validation
loss.

## Phase 2b — Watershed instance separation at inference time

### Algorithm

Pure post-processing on the per-tile binary mask (same mask the
baseline writes):

    1. Foreground = (binary > 0)
    2. elevation = EDT(foreground)       [or probability map if present]
    3. markers = peak_local_max(elevation, min_distance=k)
    4. labels = watershed(-elevation, markers=markers, mask=foreground)

The outputs are per-instance labels. Each label is polygonised
independently so that two formerly-touching components exported as
one polygon end up as two polygons in the output GeoJSON.

### `min_marker_distance_m` default (5 m) — rationale

- 2 m: over-segments single-family houses (Vegas), because the EDT
  has small ripples that register as separate peaks.
- 10 m: under-segments row houses and informal urban clusters
  (Shanghai) because two adjacent 5 m-wide footprints share a
  single peak.
- 5 m: splits touching pairs reliably on all four AOIs in a visual
  pre-check without over-splitting isolated houses.

Overridable in each `configs/spacenet_<aoi>_phase2b.yaml` under
`postprocess.min_marker_distance_m`.

### Why EDT (not probability map) by default?

Upstream `tile_label_to_file` writes argmax PNGs, not probability
maps, so a probability-driven watershed would need a modified
`scripts/predict_local.py` that dumps logits. That is a Phase 2b
follow-up, not a blocker. The EDT-based watershed is the standard
pattern for binary-mask instance separation (`skimage.segmentation.
watershed` on `-EDT`) and gets us the Phase 2b comparison without a
predict-side rebuild. When a probability map **is** present at
`<z>/<x>/<y>.prob.npy`, `watershed_split_mask` uses it in preference.

### No retraining required

Phase 2b runs on the Phase 1 fine-tuned binary checkpoints — same
model as Phase 1, different post-processing. Running Phase 2a and
Phase 2b in parallel teases apart how much of any Phase 2 gain is
training-side vs inference-side.

## Comparison figure

`scripts/merge_rate_overlay.py` reads up to four eval parquets per
AOI (baseline, Phase 1, Phase 2a, Phase 2b) and overlays merge rate
vs local density (200 m window) with per-bin 95 % bootstrap CIs. It
is tolerant of missing series (silently skips any parquet not on
disk) so you can regenerate the figure after each phase as parquets
land.

## What is deliberately *not* in Phase 2

- No augmentation ablation. Phase 2a could be helped by elastic /
  flip augmentations on the 3-class masks, but that is one more
  hyperparameter and the brief calls it out of scope.
- No probability-driven watershed by default. Requires a predict-
  side change; staged for a Phase 2b follow-up PR.
- No marker-sweep over `min_marker_distance_m`. Picking the value
  by eye on a pre-check tile set is honest; picking it by optimising
  on test-split merge rate is not. The YAML exposes the knob;
  nothing else does.
- No new metric. Phase 2's success/failure is read off the existing
  merge-rate + F1 metrics on the Phase 1 test blocks.
