# Upstream PR 0003 — make `tiles_from_dir` work on Windows paths

**Branch:** `upstream-pr/fix-tiles-from-dir-windows`
**Base:** `e1668a470b94252040f27d26098826c293fbb46d`
**Affected file:** `abd_model/src/abd_model/tiles.py`

## The problem

`tiles_from_dir` builds a regex from `os.path.join(root, "(?P<z>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+).+")`
and matches it against paths returned by `glob.glob(os.path.join(root,
"[0-9]*/[0-9]*/[0-9]*.*"))`. Two things break this on Windows:

1. `os.path.join(root, ...)` uses `\` as the separator. In a regex,
   `\i` is not a recognised escape, so the regex compiler raises.
2. The tile-path template hardcodes `/` between the (z, x, y) groups,
   so even if the regex compiled, it wouldn't match paths like
   `C:\...\images\17\42\68.png` (backslashes between groups).

### Failing command

```
> abd predict --dataset C:\bfbench\runs\tiles --checkpoint weights\x.pth ...
```

### Traceback

```
Traceback (most recent call last):
  File ".../abd_model/tools/predict.py", line 133, in main
    dataset = getattr(loader, chkpt["loader"])(config, ..., args.dataset, ...)
  File ".../abd_model/loaders/semseg.py", line NN, in __init__
    self.tiles = [t for t in tiles_from_dir(root)]
  File ".../abd_model/tiles.py", line 73, in tiles_from_dir
    tile_xyz = re.match(os.path.join(root, "(?P<z>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+).+"), path)
  File ".../lib/python3.11/re/__init__.py", line ..., in match
    return _compile(pattern, flags).match(string)
  ...
re.error: bad escape \i at position 3
```

(The exact position varies by drive letter; for a dataset under
`C:\images` it lands on `\i`.)

## The fix

Two changes, both localised to one `re.match` call:

```diff
-            tile_xyz = re.match(os.path.join(root, "(?P<z>[0-9]+)/(?P<x>[0-9]+)/(?P<y>[0-9]+).+"), path)
+            tile_xyz = re.match(
+                re.escape(os.path.join(root, "")) + r"(?P<z>[0-9]+)[\\/](?P<x>[0-9]+)[\\/](?P<y>[0-9]+).+",
+                path,
+            )
```

1. `re.escape(os.path.join(root, ""))` — escape everything in the
   prefix that the regex compiler would otherwise interpret.
2. `[\\/]` in the character class matches either `/` or `\`, so POSIX
   paths (forward slash) and Windows paths (backslash) both work.

## Regression test

```python
import os, tempfile, re
from pathlib import Path
from abd_model.tiles import tiles_from_dir

with tempfile.TemporaryDirectory() as d:
    (Path(d) / "17" / "42").mkdir(parents=True)
    (Path(d) / "17" / "42" / "68.png").write_bytes(b"")
    tiles = list(tiles_from_dir(d))
    assert len(tiles) == 1
    assert (tiles[0].z, tiles[0].x, tiles[0].y) == (17, 42, 68)
```

Fails with `re.error: bad escape` on the pinned SHA on Windows,
passes on all platforms after the fix.

## How this benchmark applies it

`patches/abd/0003-fix-tiles-from-dir-windows.patch`, applied by
`scripts/apply_upstream_patches.py`.
