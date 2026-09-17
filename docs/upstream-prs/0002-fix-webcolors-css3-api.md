# Upstream PR 0002 — replace removed `webcolors.CSS3_NAMES_TO_HEX`

**Branch:** `upstream-pr/fix-webcolors-css3-api`
**Base:** `e1668a470b94252040f27d26098826c293fbb46d`
**Affected file:** `abd_model/src/abd_model/core.py`

## The problem

`webcolors` dropped the `CSS3_NAMES_TO_HEX` module attribute in
release 1.13 (September 2023). Upstream's `make_palette` and
`check_color` both use it, so any code path that calls those
functions (`abd predict` hits `make_palette` at startup) fails.

### Failing command

```
$ abd predict --dataset ./tiles --checkpoint weights/x.pth --config config.toml --out ./masks
```

### Traceback

```
Traceback (most recent call last):
  File ".../abd_model/tools/predict.py", line 111, in main
    palette, transparency = make_palette([classe["color"] for classe in config["classes"]])
  File ".../abd_model/core.py", line 137, in make_palette
    hex_colors = [webcolors.CSS3_NAMES_TO_HEX[color.lower()] if color[0] != "#" else color for color in colors]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'webcolors' has no attribute 'CSS3_NAMES_TO_HEX'
```

## The fix

Swap the dict lookup for the `name_to_hex()` function that has existed
in `webcolors` since 1.0. Same input (a CSS3 colour name), same
output (a `#RRGGBB` hex string).

```diff
-    hex_colors = [webcolors.CSS3_NAMES_TO_HEX[color.lower()] if color[0] != "#" else color for color in colors]
+    hex_colors = [webcolors.name_to_hex(color.lower()) if color[0] != "#" else color for color in colors]
```

Applied identically in both `make_palette` (palette construction at
predict startup) and `check_color` (config validator).

## Regression test

```python
# python
from abd_model.core import make_palette
palette, transparency = make_palette(["black", "green"])
assert palette[:3] == [0, 0, 0]   # 'black' -> #000000
assert palette[3:6] == [0, 128, 0]  # 'green' -> #008000
```

Fails on the pinned SHA (AttributeError); passes on the branch.

## How this benchmark applies it

`patches/abd/0002-fix-webcolors-css3-api.patch`, applied by
`scripts/apply_upstream_patches.py`.
