# Upstream PR branches

Five branches sit in the local checkout of the upstream submodule
(`external/abd/`), each cutting off pinned SHA
`e1668a470b94252040f27d26098826c293fbb46d` and carrying exactly one
minimal fix. Together they cover every patch this benchmark applies
at env-setup time.

```
external/abd $ git branch --list "upstream-pr/*"
  upstream-pr/fix-filter-buildings-geopandas-api
  upstream-pr/fix-tiles-from-dir-windows
  upstream-pr/fix-vectorize-syntax
  upstream-pr/fix-webcolors-css3-api
  upstream-pr/single-gpu-predict-no-nccl
```

Each branch is one commit off the pinned SHA. The intent is to push
each as an individual PR to
[rodekruis/automated-building-detection](https://github.com/rodekruis/automated-building-detection).
This directory contains a plain factual repro for each fix: the
failing command, the traceback, and the fix. PR descriptions are
authored separately by the maintainer.

| Branch                                            | File                                                                        | Repro doc                                                            |
|---------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
| `upstream-pr/fix-vectorize-syntax`                | [`0001-fix-vectorize-syntax.md`](0001-fix-vectorize-syntax.md)              | Python SyntaxError in `abd_model.tools.vectorize` since 2020.        |
| `upstream-pr/fix-webcolors-css3-api`              | [`0002-fix-webcolors-css3-api.md`](0002-fix-webcolors-css3-api.md)          | `webcolors.CSS3_NAMES_TO_HEX` removed in webcolors 1.13 (2023).       |
| `upstream-pr/fix-tiles-from-dir-windows`          | [`0003-fix-tiles-from-dir-windows.md`](0003-fix-tiles-from-dir-windows.md)  | `re.error` in `tiles_from_dir` on Windows paths.                     |
| `upstream-pr/fix-filter-buildings-geopandas-api`  | [`0004-fix-filter-buildings-geopandas-api.md`](0004-fix-filter-buildings-geopandas-api.md) | `sjoin(op=)` + `DataFrame.append` removed in geopandas 1.0 / pandas 2.0. |
| `upstream-pr/single-gpu-predict-no-nccl`          | [`0005-single-gpu-predict-no-nccl.md`](0005-single-gpu-predict-no-nccl.md)  | `abd predict` asserts NCCL and unconditionally uses `mp.spawn`; blocks Windows entirely. |

## Regression tests

Upstream ships no pytest suite (there is no `tests/` directory in either
`abd_model/` or `abd_utils/`). Where a regression test is feasible
without introducing a whole new test infrastructure to upstream, it is
included in the branch as a self-contained script (see the individual
repro docs). Where the fix is a one-line API swap that would need
mocking the whole GDAL / geopandas surface to exercise -- that regression
lives in this repo's `tests/` instead of the upstream branch.

## Reproduction environment

Every traceback below was captured on the target Windows 11 / Python
3.11 / conda-forge env pinned in `environment.yml`. That environment is
newer than upstream's pinned deps by ~5 years, which is what surfaces
most of these problems.
