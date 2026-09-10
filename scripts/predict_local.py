"""Single-GPU, non-distributed replacement for `abd predict`.

Motivation:
    Upstream `abd_model.tools.predict` unconditionally uses
    `torch.distributed` with the NCCL backend, which is not available in
    PyTorch's Windows build. On this project's target machine (Windows
    11, RTX 5060), `abd predict` fails at
        assert torch.distributed.is_nccl_available()
    We do NOT modify upstream to work around this.

What this shim does:
    It runs the *same* AlbuNet model on the *same* SemSeg tile-tree
    layout, using upstream's class definitions unchanged
    (`abd_model.nn.albunet.Albunet`, `abd_model.loaders.semseg.SemSeg`,
    `abd_model.tiles.tile_label_to_file`,
    `abd_model.core.load_config` / `make_palette`). No forked model
    code lives here.

    The only thing this shim replaces is the distributed launch
    scaffolding: it iterates the DataLoader on the single available GPU
    (or CPU) and writes label tiles via upstream's own writer.

Why not just patch upstream to use `gloo` on Windows?
    That would require modifying vendored source substantively (backend
    selection is one line but hides more state than a simple find/replace),
    and would silently change the behaviour of *upstream's* predict for
    any user running our submodule. A separate shim keeps the change
    scoped to this project.

Inputs mirror `abd predict`'s CLI flags 1:1 so the shim is a drop-in
substitute for that specific step of the pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mercantile
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from abd_model.core import load_config, make_palette
from abd_model.loaders.semseg import SemSeg
from abd_model.nn.albunet import Albunet
from abd_model.tiles import tile_label_to_file, tiles_from_csv

_LOG = logging.getLogger("bfbench.predict_local")


def _strip_ddp_prefix(state_dict: dict) -> dict:
    """Upstream trains with DistributedDataParallel, which wraps the model
    and prefixes every parameter name with 'module.'. Strip that so keys
    load into a bare Albunet.
    """
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)

    # torch.load with weights_only=False required because the checkpoint
    # dict contains non-tensor metadata (nn class name, encoder, shape_in,
    # shape_out, model_version, uuid).
    chkpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _LOG.info(
        "checkpoint uuid=%s nn=%s encoder=%s shape_in=%s shape_out=%s",
        chkpt.get("uuid"), chkpt["nn"], chkpt["encoder"],
        chkpt["shape_in"], chkpt["shape_out"],
    )
    assert chkpt["nn"].lower() == "albunet", (
        f"predict_local only supports Albunet checkpoints; got {chkpt['nn']!r}"
    )

    model = Albunet(chkpt["shape_in"], chkpt["shape_out"], chkpt["encoder"].lower())
    model.load_state_dict(_strip_ddp_prefix(chkpt["state_dict"]))
    model.to(device)
    model.eval()

    palette, transparency = make_palette([c["color"] for c in config["classes"]])

    cover = (
        list(tiles_from_csv(str(Path(args.cover).expanduser())))
        if args.cover
        else None
    )

    dataset = SemSeg(
        config,
        chkpt["shape_in"][1:3],
        args.dataset,
        cover=cover,
        mode="predict",
        metatiles=args.metatiles,
        keep_borders=args.keep_borders,
    )
    assert len(dataset) > 0, f"empty dataset directory: {args.dataset}"

    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    C, W, H = chkpt["shape_out"]
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, tiles in tqdm(loader, desc="predict_local", unit="batch"):
            images = images.to(device, non_blocking=True)

            if args.metatiles:
                # Match upstream gpu_worker's metatile quadrant strategy.
                # The SemSeg loader with metatiles=True hands us a buffered
                # (3W/2, 3H/2) crop centered on the target tile. We slide a
                # WxH window over it in 4 stride-W/2 positions and keep the
                # inner W/2 x W/2 quadrant of each prediction.
                N = images.shape[0]
                qs = int(W / 4)
                hs = int(W / 2)
                ts = int(W)
                probs = np.zeros((N, C, W, H), dtype=np.float32)
                q_slices = [
                    ((slice(0, ts), slice(0, ts)),   (slice(0, hs), slice(0, hs))),
                    ((slice(0, ts), slice(hs, None)),(slice(0, hs), slice(hs, None))),
                    ((slice(hs, None), slice(0, ts)),(slice(hs, None), slice(0, hs))),
                    ((slice(hs, None), slice(hs, None)), (slice(hs, None), slice(hs, None))),
                ]
                for src_slice, dst_slice in q_slices:
                    inp = images[:, :, src_slice[0], src_slice[1]]
                    out = model(inp).detach().cpu().numpy()
                    probs[:, :, dst_slice[0], dst_slice[1]] = out[:, :, qs:-qs, qs:-qs]
            else:
                probs = model(images).detach().cpu().numpy()

            for tile_xyz, prob in zip(tiles, probs, strict=True):
                x, y, z = (int(v) for v in tile_xyz)
                mask = np.zeros((W, H), dtype=np.uint8)
                for c in range(C):
                    mask += np.around(prob[c, :, :]).astype(np.uint8) * c
                tile_label_to_file(
                    str(out_dir),
                    mercantile.Tile(x, y, z),
                    palette,
                    transparency,
                    mask,
                )

    _LOG.info("wrote label tiles to %s", out_dir)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cover", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metatiles", action="store_true")
    p.add_argument("--keep_borders", action="store_true")
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers; keep at 0 on Windows to avoid "
                        "the 'MSVC' fork issue during unpickling.")
    args = p.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
