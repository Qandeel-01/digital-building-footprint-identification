"""Training dataset: reads the ABD slippy-map tile tree + rasterises the
dedup'd reference polygons into per-tile binary masks on the fly.

Why reuse the ABD tile tree rather than resample from the raw mosaic:

The whole point of Phase 1 is to fine-tune the AlbuNet in the same
pre-processing regime it will be evaluated in. `bfbench run-prepared`
feeds the model exactly these PNGs (0.5 m/px, 512x512, resampled from
30 cm SpaceNet). Training on the same tiles rules out any train/eval
pre-processing drift (an easy and silent way to overstate gains).

For masks:

The dedup'd reference GeoDataFrame (from `bfbench prepare`) is the
ground truth across the whole AOI. We clip to each tile's WGS84 bbox,
reproject into the tile's local raster transform (built from the tile's
mercantile bounds in EPSG:3857 and the tile's pixel shape), and
`rasterize` with `all_touched=False`. The result is binary {0, 1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
from rasterio.features import rasterize as _rasterize
from rasterio.transform import from_bounds as _from_bounds
from shapely.geometry import box

from bfbench.data.adapter import WEBMERCATOR_EPSG, tile_bounds_webmercator

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a runtime dep
    _HAS_TORCH = False

    class Dataset:  # type: ignore[no-redef]
        """Minimal stand-in so the module imports in torch-less test envs."""

        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> Any:
            raise IndexError(idx)


@dataclass(frozen=True)
class TileSample:
    """One (tile, image path) pair the dataset will consume."""

    tile: mercantile.Tile
    image_path: Path


def _ref_in_mercator(reference_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject the reference into EPSG:3857 so per-tile rasterisation
    uses the same CRS as the tile's bounds.
    """
    if reference_gdf.crs is None:
        reference_gdf = reference_gdf.set_crs("EPSG:4326")
    return reference_gdf.to_crs(f"EPSG:{WEBMERCATOR_EPSG}")


def rasterise_reference_to_tile(reference_gdf_mercator: gpd.GeoDataFrame,
                               tile: mercantile.Tile,
                               tile_size: int) -> np.ndarray:
    """Return a {0, 1} HxW uint8 mask for the given tile.

    `reference_gdf_mercator` must already be in EPSG:3857 so this
    function doesn't repay the per-tile reprojection cost in a tight
    training loop.
    """
    w, s, e, n = tile_bounds_webmercator(tile)
    tile_poly = box(w, s, e, n)
    clipped = reference_gdf_mercator[reference_gdf_mercator.intersects(tile_poly)]
    out = np.zeros((tile_size, tile_size), dtype=np.uint8)
    if len(clipped) == 0:
        return out
    shapes = [(g, 1) for g in clipped.geometry if g is not None and not g.is_empty]
    if not shapes:
        return out
    transform = _from_bounds(w, s, e, n, tile_size, tile_size)
    mask = _rasterize(shapes, out_shape=(tile_size, tile_size),
                      transform=transform, fill=0,
                      all_touched=False, dtype="uint8")
    return mask


def _read_image(path: Path) -> np.ndarray:
    """Read an RGB tile as a (3, H, W) float32 array scaled to [0, 1].

    rasterio handles both PNG and GeoTIFF; the ABD tile tree uses PNG
    but the tests may use GeoTIFF. Values beyond 3 bands are ignored.
    """
    with rasterio.open(path) as src:
        arr = src.read(out_dtype="float32")
    if arr.shape[0] > 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.concatenate([arr, arr, arr], axis=0)
    arr = arr / 255.0
    return arr.astype(np.float32)


class BuildingTileDataset(Dataset):
    """Torch Dataset for building-segmentation fine-tuning.

    Each item is (image (3,H,W) float32 in [0,1], mask (1,H,W) float32
    in {0,1}).

    Parameters
    ----------
    tiles
        List of mercantile tiles to iterate over. Order is preserved.
    tiles_root
        Root of the ABD slippy-map tree, e.g. `<work_dir>/tiles/`. The
        dataset expects `tiles_root/<channel_name>/<z>/<x>/<y>.<ext>`.
    reference_gdf
        Dedup'd reference polygons. CRS may be anything pyproj can
        parse; the dataset reprojects to EPSG:3857 at construction.
    channel_name
        ABD channel sub-directory. Defaults to the upstream config's
        `images`.
    tile_size
        Pixel side length. Must match the AlbuNet checkpoint's shape_in.
    extension
        File extension for image reads. Defaults to ".png"; GeoTIFF
        works too for tests.
    """

    def __init__(self,
                 tiles: list[mercantile.Tile],
                 tiles_root: Path,
                 reference_gdf: gpd.GeoDataFrame,
                 *,
                 channel_name: str = "images",
                 tile_size: int = 512,
                 extension: str = ".png") -> None:
        if not _HAS_TORCH:  # pragma: no cover - trivial guard
            raise ImportError(
                "BuildingTileDataset requires torch. Install torch==2.11.0 "
                "(see environment.yml) to use the training entry points."
            )
        self._tiles = list(tiles)
        self._tiles_root = Path(tiles_root)
        self._channel = channel_name
        self._tile_size = int(tile_size)
        self._extension = extension
        self._ref_mercator = _ref_in_mercator(reference_gdf)
        # Pre-resolve the image paths; fail fast if any tile is missing.
        self._samples: list[TileSample] = []
        for t in self._tiles:
            p = (self._tiles_root / channel_name / str(t.z) / str(t.x)
                 / f"{t.y}{extension}")
            if not p.is_file():
                raise FileNotFoundError(
                    f"BuildingTileDataset: tile image missing at {p}. "
                    "Run `bfbench run-prepared` with --stop-after adapt "
                    "to generate the tile tree first."
                )
            self._samples.append(TileSample(tile=t, image_path=p))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        s = self._samples[idx]
        img_np = _read_image(s.image_path)                 # (3, H, W)
        mask_np = rasterise_reference_to_tile(
            self._ref_mercator, s.tile, self._tile_size,
        ).astype(np.float32)                               # (H, W)
        img = torch.from_numpy(img_np)
        mask = torch.from_numpy(mask_np).unsqueeze(0)      # (1, H, W)
        return img, mask


__all__ = [
    "BuildingTileDataset",
    "TileSample",
    "rasterise_reference_to_tile",
]
