# Upstream PR 0005 — single-GPU `abd predict` without NCCL / DDP

**Branch:** `upstream-pr/single-gpu-predict-no-nccl`
**Base:** `e1668a470b94252040f27d26098826c293fbb46d`
**Affected file:** `abd_model/src/abd_model/tools/predict.py`

## The problem

`abd predict` unconditionally launches with `torch.distributed` on the
NCCL backend, via `mp.spawn`, so:

1. It fails on any platform where PyTorch was built without NCCL —
   notably every native Windows PyTorch build. PyTorch on Windows
   ships without NCCL and the `is_nccl_available()` guard trips at
   the top of `main`.
2. On single-GPU boxes (Linux or Windows) the multiprocessing +
   distributed scaffolding is pure overhead — the DataLoader,
   `DistributedSampler`, and `mp.spawn` all cost startup time and
   RAM for no gain.

### Failing command (Windows)

```
> abd predict --dataset .\tiles --checkpoint weights\neat-fullxview-epoch75.pth --config external\abd\input\config.toml --out .\masks
```

### Traceback (Windows / native)

```
Traceback (most recent call last):
  File ".../abd_model/tools/predict.py", line 105, in main
    assert torch.distributed.is_nccl_available(), "No NCCL support found. Check your PyTorch install."
AssertionError: No NCCL support found. Check your PyTorch install.
```

There is no easy workaround: PyTorch on Windows will not gain NCCL
without a major upstream re-vendoring, and switching backend to `gloo`
requires a code change here.

## The fix

Add a `--single-gpu` flag. When set:

* Skip the NCCL availability assertion.
* Bypass `mp.spawn`; call `gpu_worker(0, 1, ...)` directly on rank 0
  with `world_size=1`.

The default path (no flag) is unchanged: multi-GPU users still get
`mp.spawn` + NCCL, and single-GPU Linux users still get the current
behaviour if they choose.

```diff
     perf = parser.add_argument_group("Performances")
     perf.add_argument("--bs", ...)
     perf.add_argument("--workers", ...)
+    perf.add_argument(
+        "--single-gpu", action="store_true",
+        help="Run prediction on a single device without torch.distributed/NCCL. "
+             "Required on platforms without NCCL (e.g. Windows).",
+    )
...
     assert torch.cuda.is_available(), "No GPU support found. ..."
-    assert torch.distributed.is_nccl_available(), "No NCCL support found. ..."
+    if not getattr(args, "single_gpu", False):
+        assert torch.distributed.is_nccl_available(), "No NCCL support found. Check your PyTorch install (or pass --single-gpu)."
...
-    mp.spawn(gpu_worker, nprocs=world_size, args=(world_size, lock_file, args, config, dataset, palette, transparency))
+    if getattr(args, "single_gpu", False):
+        gpu_worker(0, 1, lock_file, args, config, dataset, palette, transparency)
+    else:
+        mp.spawn(gpu_worker, nprocs=world_size, args=(world_size, lock_file, args, config, dataset, palette, transparency))
```

Note that `gpu_worker` still calls `dist.init_process_group("nccl",
...)` internally. On a Linux single-GPU box that will just work if
NCCL is available; on Windows it still won't. The minimal fix here is
the flag; a follow-up PR can teach `gpu_worker` to skip DDP entirely
when `world_size == 1`. That is deliberately out of scope for this
one-fix-per-branch policy.

## What this project uses instead

Until upstream ships something equivalent, `scripts/predict_local.py`
in this repo is a single-GPU, no-DDP shim that reuses upstream's
`Albunet` / `SemSeg` / `tile_label_to_file` classes and skips the
distributed launch entirely. See `docs/environment-notes.md` for
context. That shim will be deleted when upstream merges this PR (or
an equivalent one that also removes the DDP wrap for `world_size=1`).

## Regression test

A full regression requires a working GPU + CUDA on Linux, so it
cannot run in this project's CI. On the target machine:

```
$ abd predict --single-gpu --dataset ./tiles --checkpoint ... --config ... --out ./masks
```

should reach the DataLoader loop instead of tripping the NCCL
assertion. On the pinned SHA it fails at the assertion; on the branch
it enters `gpu_worker(0, 1, ...)`.

## How this benchmark applies it

`patches/abd/0005-single-gpu-predict-no-nccl.patch`, applied by
`scripts/apply_upstream_patches.py`.
