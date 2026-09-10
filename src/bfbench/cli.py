"""bfbench: top-level CLI that wires SpaceNet fetching + tile adaptation +
upstream `abd` subcommands into a single reproducible pipeline.

The upstream `abd` command is an argparse subcommand entry-point registered
by the `abd_model` package. It is not an importable function; we invoke it
via subprocess exactly as an operator would from the shell. This preserves
upstream's own env-var handling, distributed launcher, and exit codes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

from bfbench.data.adapter import AdapterConfig, adapt_spacenet_image
from bfbench.data.spacenet import SpaceNetAOI, SpaceNetFetcher

_LOG = logging.getLogger("bfbench")

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AOIConfig:
    provider: str
    aoi_id: int
    split: str
    image_ids: list[str]
    image_kind: str = "PS-RGB"


@dataclass(frozen=True)
class ABDConfig:
    config: Path
    checkpoint: Path
    vectorize_type: str = "Building"
    metatiles: bool = True


@dataclass(frozen=True)
class RunConfig:
    work_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    aoi: AOIConfig
    adapter: AdapterConfig
    abd: ABDConfig
    run: RunConfig


class ConfigError(RuntimeError):
    """Raised when a YAML pipeline config is invalid."""


def _load_config(path: Path) -> PipelineConfig:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping in {path}")

    for section in ("aoi", "adapter", "abd", "run"):
        if section not in raw:
            raise ConfigError(f"config {path} missing required section '{section}'")

    aoi_raw = raw["aoi"]
    if aoi_raw.get("provider") != "spacenet2":
        raise ConfigError(
            f"config {path}: aoi.provider must be 'spacenet2' "
            f"(got {aoi_raw.get('provider')!r})"
        )
    aoi = AOIConfig(
        provider="spacenet2",
        aoi_id=int(aoi_raw["aoi_id"]),
        split=str(aoi_raw.get("split", "train")),
        image_ids=list(aoi_raw["image_ids"]),
        image_kind=str(aoi_raw.get("image_kind", "PS-RGB")),
    )

    ad_raw = raw["adapter"]
    adapter_cfg = AdapterConfig(
        target_gsd_m=float(ad_raw.get("target_gsd_m", 0.5)),
        tile_size=int(ad_raw.get("tile_size", 512)),
        zoom=ad_raw.get("zoom"),
        zoom_tolerance=float(ad_raw.get("zoom_tolerance", 0.15)),
        keep_borders=bool(ad_raw.get("keep_borders", True)),
        channel_name=str(ad_raw.get("channel_name", "images")),
    )

    abd_raw = raw["abd"]
    abd_cfg = ABDConfig(
        config=Path(abd_raw["config"]).expanduser(),
        checkpoint=Path(abd_raw["checkpoint"]).expanduser(),
        vectorize_type=str(abd_raw.get("vectorize_type", "Building")),
        metatiles=bool(abd_raw.get("metatiles", True)),
    )

    run_cfg = RunConfig(work_dir=Path(raw["run"]["work_dir"]).expanduser())

    return PipelineConfig(aoi=aoi, adapter=adapter_cfg, abd=abd_cfg, run=run_cfg)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _fetch_step(cfg: PipelineConfig) -> Path:
    """Ensure the requested SpaceNet images + geojson are on disk. Returns the
    directory containing the fetched files.
    """
    fetcher = SpaceNetFetcher()
    aoi = SpaceNetAOI(aoi_id=cfg.aoi.aoi_id, split=cfg.aoi.split)
    fetcher.verify_access(aoi)
    out_dir = cfg.run.work_dir / "spacenet"
    out_dir.mkdir(parents=True, exist_ok=True)
    for iid in cfg.aoi.image_ids:
        fetcher.download_pair(aoi, iid, out_dir, image_kind=cfg.aoi.image_kind)
    return out_dir


def _adapt_step(cfg: PipelineConfig, spacenet_dir: Path) -> Path:
    """Tile each fetched image into abd's slippy-map layout. Returns the
    tile-tree root that `abd predict --dataset` should point at.
    """
    tiles_root = cfg.run.work_dir / "tiles"
    if tiles_root.exists():
        # Fresh run: clear stale tiles so cover.csv and disk agree.
        shutil.rmtree(tiles_root)
    tiles_root.mkdir(parents=True, exist_ok=True)

    for iid in cfg.aoi.image_ids:
        img_name = (
            f"SN2_buildings_{cfg.aoi.split}_AOI_{cfg.aoi.aoi_id}_"
            f"{SpaceNetAOI(cfg.aoi.aoi_id, cfg.aoi.split).city}_"
            f"{cfg.aoi.image_kind}_{iid}.tif"
        )
        src = spacenet_dir / cfg.aoi.image_kind / img_name
        if not src.is_file():
            raise FileNotFoundError(
                f"expected SpaceNet image not present after fetch: {src}. "
                "Delete the fetch cache and re-run, or check spacenet.py logs."
            )
        result = adapt_spacenet_image(src, tiles_root, config=cfg.adapter)
        _LOG.info(
            "adapted %s -> %d tiles at z=%d (native_gsd=%.3f m, target=%.3f m)",
            src.name, result.tiles_written, result.zoom, result.native_gsd_m,
            result.target_gsd_m,
        )
    return tiles_root


def _run_abd(args: list[str]) -> None:
    """Invoke the upstream `abd` CLI, streaming stdout/stderr through. Non-zero
    exit is fatal and reported with the full command for reproducibility.
    """
    _LOG.info("+ %s", " ".join(args))
    completed = subprocess.run(args, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"upstream command failed (exit={completed.returncode}): "
            f"{' '.join(args)}"
        )


def _predict_step(cfg: PipelineConfig, tiles_root: Path) -> Path:
    if not cfg.abd.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {cfg.abd.checkpoint}. Run "
            "`python scripts/fetch_weights.py` first."
        )
    if not cfg.abd.config.is_file():
        raise FileNotFoundError(
            f"abd config not found: {cfg.abd.config}. Point `abd.config` "
            "at the vendored toml (external/abd/input/config.toml) or a copy."
        )
    masks_dir = cfg.run.work_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cover = tiles_root / "cover.csv"

    argv = [
        "abd", "predict",
        "--dataset", str(tiles_root),
        "--checkpoint", str(cfg.abd.checkpoint),
        "--config", str(cfg.abd.config),
        "--cover", str(cover),
        "--out", str(masks_dir),
    ]
    if cfg.abd.metatiles:
        argv.append("--metatiles")
        if cfg.adapter.keep_borders:
            argv.append("--keep_borders")
    _run_abd(argv)
    return masks_dir


def _vectorize_step(cfg: PipelineConfig, masks_dir: Path) -> Path:
    raw_geojson = cfg.run.work_dir / "predicted_raw.geojson"
    _run_abd(
        [
            "abd", "vectorize",
            "--masks", str(masks_dir),
            "--type", cfg.abd.vectorize_type,
            "--config", str(cfg.abd.config),
            "--out", str(raw_geojson),
        ]
    )
    return raw_geojson


def _filter_step(cfg: PipelineConfig, raw_geojson: Path) -> Path:
    final_geojson = cfg.run.work_dir / "predicted.geojson"
    _run_abd(
        [
            "filter-buildings",
            "--data", str(raw_geojson),
            "--dest", str(final_geojson),
        ]
    )
    return final_geojson


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=False, help="YAML pipeline config."),
    stop_after: str = typer.Option(
        "filter",
        "--stop-after",
        help="Stop after this stage. One of: fetch, adapt, predict, vectorize, filter.",
    ),
) -> None:
    """Run the full adapt -> predict -> vectorize -> filter pipeline for one AOI."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    valid_stages = ["fetch", "adapt", "predict", "vectorize", "filter"]
    if stop_after not in valid_stages:
        raise typer.BadParameter(
            f"--stop-after must be one of {valid_stages}; got {stop_after!r}"
        )

    cfg = _load_config(config_path)
    cfg.run.work_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(cfg, config_path)

    spacenet_dir = _fetch_step(cfg)
    if stop_after == "fetch":
        return

    tiles_root = _adapt_step(cfg, spacenet_dir)
    if stop_after == "adapt":
        return

    masks_dir = _predict_step(cfg, tiles_root)
    if stop_after == "predict":
        return

    raw_geojson = _vectorize_step(cfg, masks_dir)
    if stop_after == "vectorize":
        return

    final = _filter_step(cfg, raw_geojson)
    typer.echo(f"final geojson: {final}")


def _write_manifest(cfg: PipelineConfig, config_path: Path) -> None:
    """Write a run-manifest so every artifact traces back to the config."""
    manifest = {
        "config_source": str(config_path.resolve()),
        "aoi": dataclasses.asdict(cfg.aoi),
        "adapter": {
            "target_gsd_m": cfg.adapter.target_gsd_m,
            "tile_size": cfg.adapter.tile_size,
            "zoom": cfg.adapter.zoom,
            "zoom_tolerance": cfg.adapter.zoom_tolerance,
            "keep_borders": cfg.adapter.keep_borders,
            "channel_name": cfg.adapter.channel_name,
            "resampling": cfg.adapter.resampling.name,
        },
        "abd": {
            "config": str(cfg.abd.config.resolve()),
            "checkpoint": str(cfg.abd.checkpoint.resolve()),
            "vectorize_type": cfg.abd.vectorize_type,
            "metatiles": cfg.abd.metatiles,
        },
        "python": sys.version,
        "env": {k: os.environ.get(k, "") for k in ("CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES")},
    }
    (cfg.run.work_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


@app.command()
def verify() -> None:
    """Verify that the upstream `abd` CLI and CUDA are reachable from this env."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    abd = shutil.which("abd")
    if not abd:
        typer.echo("FAIL: `abd` not on PATH; did you `pip install -e ./external/abd/abd_model`?")
        raise typer.Exit(code=2)
    typer.echo(f"OK  abd -> {abd}")

    try:
        import torch  # noqa: PLC0415  # runtime import: torch is heavy
    except ImportError as e:
        typer.echo(f"FAIL: torch import failed: {e}")
        raise typer.Exit(code=2) from e

    typer.echo(f"OK  torch {torch.__version__} cuda_build={torch.version.cuda}")
    typer.echo(f"{'OK  ' if torch.cuda.is_available() else 'WARN'} torch.cuda.is_available()={torch.cuda.is_available()}")
    if not torch.distributed.is_nccl_available():
        typer.echo(
            "WARN torch.distributed.is_nccl_available()=False. "
            "Upstream `abd predict` asserts NCCL and will fail on Windows. "
            "Run predict inside WSL/Linux/Docker; adapt+vectorize+filter run natively."
        )


if __name__ == "__main__":  # pragma: no cover
    app()
