# Reference-polygon layer (SpaceNet 2)

SpaceNet 2 ships one `geojson_buildings/*.geojson` per chip. Buildings
that cross a chip boundary appear in BOTH adjacent chips, clipped to the
chip footprint. Naively concatenating those files double-counts every
seam building, which corrupts per-tile counts and every metric that
depends on them.

## Dedup rules

`bfbench.data.reference.merge_reference` implements three rules,
selected via `DedupRule.method` (all configurable via the AOI YAML,
`reference.dedup.*`):

| Rule              | Merges                                                                  | Best for                                                     |
|-------------------|-------------------------------------------------------------------------|--------------------------------------------------------------|
| `touching_union`  | Buffer + `unary_union` + de-buffer                                      | Default. Reunites seam partials (their edges touch).         |
| `iou_threshold`   | Pairs with IoU >= threshold                                             | Strict alternative; does NOT merge seam partials (IoU = 0).  |
| `identity`        | Exact WKB match                                                         | Control / null rule; does NOT merge seam partials.           |

### `touching_union` (default): why 0.5 m buffer

- SpaceNet PS-RGB is ~30 cm/px, so a 0.5 m buffer is <2 pixels. That is
  small enough that two truly separate buildings within 0.5 m are
  extremely rare (their walls would be touching in real life).
- Buffer + de-buffer with a 0.5 m buffer stays inside the geometric
  precision of the raw labels (SpaceNet polygon coordinates are stored
  in double-precision degrees, ~1e-6 deg ≈ 0.1 m).
- Buffering is done in the equal-area CRS (default EPSG:6933) so
  "0.5 m" means metres regardless of latitude.

### Rejected: `centroid_distance`

A centroid-distance rule was considered and rejected. A building split
across a chip seam produces TWO clipped polygons whose centroids can be
50-100 m apart (each centroid moves toward the interior of its side of
the seam). No single distance threshold both catches those partials
AND leaves distinct-but-nearby buildings alone. `touching_union` uses
the fact that clipped partials *share a boundary*, which is a structural
property of chip clipping, not a distance heuristic.

## Reporting

For every prepared AOI, `bfbench prepare` writes
`<work_dir>/reference_dedup.json` with:

```json
{
    "method": "touching_union",
    "n_input": <total polygons before dedup>,
    "n_output": <total polygons after dedup>,
    "n_duplicates_removed": <difference>
}
```

Expected magnitudes on SpaceNet 2 train splits (order-of-magnitude
sanity checks; verify at Task 3 evaluation time):

| AOI      | Approx chips | Approx raw polys | Expected duplicates removed |
|----------|--------------|------------------|-----------------------------|
| Vegas    | 3,851        | ~150k            | a small % (seam density x building density) |
| Paris    | 1,148        | ~23k             | a small %                   |
| Shanghai | 4,582        | ~92k             | a small %                   |
| Khartoum | 1,012        | ~35k             | a small %                   |

If `n_duplicates_removed / n_input` comes out `> 5 %`, that's implausible
for SpaceNet chip geometry and probably indicates a bug: the buffer is
too aggressive, or the chip list overlaps in a way other than the seam.
If it comes out `= 0` on a rule that should merge, the buffer or IoU is
too strict.

## Equal-area CRS

Every area computation in `reference.py` (and in `summary.py`) goes
through `bfbench.data.equal_area.assert_equal_area_crs`. The default
is EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global). Do NOT compute area
in EPSG:4326 -- see `equal_area.py` docstring for the rationale.
