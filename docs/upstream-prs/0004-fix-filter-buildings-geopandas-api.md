# Upstream PR 0004 — modernise `filter_buildings` for geopandas 1.0 / pandas 2.0

**Branch:** `upstream-pr/fix-filter-buildings-geopandas-api`
**Base:** `e1668a470b94252040f27d26098826c293fbb46d`
**Affected file:** `abd_utils/src/abd_utils/filter_buildings.py`

## The problem

`filter-buildings` fails at the first `sjoin` on modern geopandas, and
later at the first `.append()` on modern pandas.

Two independent breakages, one PR because they're in the same file:

* `geopandas.sjoin(..., op="intersects")` — the `op=` kwarg was
  renamed to `predicate=` in geopandas 0.10 (2021), kept as a
  deprecated alias, and removed in geopandas 1.0 (2024).
* `DataFrame.append` — removed in pandas 2.0 (2023). The replacement
  is `pd.concat([...], ignore_index=True)`.

### Failing command

```
$ filter-buildings --data raw.geojson --dest filtered.geojson
```

### Traceback (geopandas)

```
Traceback (most recent call last):
  File ".../abd_utils/filter_buildings.py", line 26, in main
    df_sj = gpd.sjoin(gdf_, gdf_, how='left', op='intersects')
TypeError: sjoin() got an unexpected keyword argument 'op'
```

### Traceback (pandas, after the sjoin is patched but before the append is)

```
Traceback (most recent call last):
  File ".../abd_utils/filter_buildings.py", line 34, in main
    gdf = gdf.append(df_sj.copy(), ignore_index=True)
AttributeError: 'GeoDataFrame' object has no attribute 'append'
```

## The fix

Pure API-swap. Four `sjoin` call sites, one `.append`. See the branch
diff for the full patch.

```diff
-            df_sj = gpd.sjoin(gdf_, gdf_, how='left', op='intersects')
+            df_sj = gpd.sjoin(gdf_, gdf_, how='left', predicate='intersects')
...
-            gdf = gdf.append(df_sj.copy(), ignore_index=True)
+            gdf = pd.concat([gdf, df_sj.copy()], ignore_index=True)
```

No behaviour change: `predicate='intersects'` was the semantic of
`op='intersects'` from day one; `pd.concat([a, b], ignore_index=True)`
is documented as the direct replacement for `a.append(b,
ignore_index=True)`.

## Regression test

```python
import geopandas as gpd
from shapely.geometry import Polygon
from abd_utils.filter_buildings import main as filter_main

gdf = gpd.GeoDataFrame(geometry=[
    Polygon([(0,0),(1,0),(1,1),(0,1)]),          # 1 x 1 = 1 m2
    Polygon([(10,0),(20,0),(20,10),(10,10)]),    # 10 x 10 = 100 m2
], crs="EPSG:3857")
gdf.to_file("in.geojson", driver="GeoJSON")
# Runs to completion on the branch; raises TypeError/AttributeError on the pinned SHA.
filter_main(data="in.geojson", dest="out.geojson", crsmeters=3857, waterbodies=None, area=50)
```

## How this benchmark applies it

`patches/abd/0004-fix-filter-buildings-geopandas-api.patch`, applied
by `scripts/apply_upstream_patches.py`.
