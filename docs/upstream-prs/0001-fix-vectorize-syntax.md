# Upstream PR 0001 — fix Python SyntaxError in `abd_model.tools.vectorize`

**Branch:** `upstream-pr/fix-vectorize-syntax`
**Base:** `e1668a470b94252040f27d26098826c293fbb46d`
**Affected file:** `abd_model/src/abd_model/tools/vectorize.py`

## The problem

Any `abd` subcommand fails at import time on Python 3.6+. The error is
independent of what you actually run; it hits during
`abd_model.tools.__main__`'s module-level import of every tool.

### Failing command

```
$ abd predict --dataset ./tiles --checkpoint weights/x.pth --config config.toml --out ./masks
```

### Traceback

```
Traceback (most recent call last):
  File ".../miniconda3/envs/bfbench/Scripts/abd-script.py", line 33, in <module>
    sys.exit(load_entry_point('abd-model==1.0.0', 'console_scripts', 'abd')())
  File ".../site-packages/abd_model/tools/__main__.py", line 20, in main
    submodule = importlib.import_module("abd_model.tools.{}".format(module))
  File ".../miniconda3/envs/bfbench/lib/python3.11/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File ".../site-packages/abd_model/tools/vectorize.py", line 56
    transform = rasterio.transform.from_bounds((*mercantile.bounds(tile.x, tile.y, tile.z)), W, H)
                                               ^
SyntaxError: cannot use starred expression here
```

The line has been in upstream since the initial commit (2020). It has
never been valid Python: the parser rejects `(*iterable)` outside a
call site's argument list. Python 3.5 was strict about this and every
version since has kept the rule.

## The fix

Remove the redundant parentheses so the starred expression sits
directly in the argument list of `from_bounds`.

```diff
-        transform = rasterio.transform.from_bounds((*mercantile.bounds(tile.x, tile.y, tile.z)), W, H)
+        transform = rasterio.transform.from_bounds(*mercantile.bounds(tile.x, tile.y, tile.z), W, H)
```

No behaviour change; `from_bounds` still receives `(west, south, east,
north, W, H)`.

## Regression test

Rebuilding the failure is trivial: `python -c "import
abd_model.tools.vectorize"` reproduces the SyntaxError on the pinned
SHA and succeeds on the branch. That one-liner is the regression test.

## How this benchmark applies it

`patches/abd/0001-fix-vectorize-syntax.patch` is applied at env-setup
time by `scripts/apply_upstream_patches.py`. The submodule pointer
stays on the pinned SHA.
