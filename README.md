# Digital Building Footprint Identification

Automated building detection using deep learning for rapid, large-scale building footprint mapping from very-high-resolution satellite imagery. Produces GeoJSON vector output ready for GIS integration, supporting disaster response and urban analysis workflows.

## Overview

| Property | Detail |
|---|---|
| Input | VHR satellite imagery (≤ 0.5 m/pixel) — Bing Maps or custom raster |
| Output | Building footprints in GeoJSON format |
| Architecture | AlbuNet — U-Net encoder-decoder with ResNet/ResNext/WideResNet encoder |
| Training dataset | xBD (disaster dataset) |
| Best model performance | IoU 0.79 · MCC 0.75 |

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning | PyTorch |
| Geospatial I/O | GDAL · Rasterio · Shapely |
| Image tiling | OSM slippy map tile scheme |
| Containerization | Docker + NVIDIA Container Toolkit |

## Setup

### Requirements

- GPU with ≥ 8 GB VRAM
- NVIDIA drivers + CUDA Toolkit

For image download: a [Bing Maps API key](https://docs.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key)

### Using Docker

```bash
docker pull rodekruis/automated-building-detection
docker run --name automated-building-detection -dit \
  -v <path-to-your-workspace>:/workdir \
  --ipc=host --gpus all -p 5000:5000 \
  rodekruis/automated-building-detection
docker exec -it automated-building-detection bash
```

### Manual Setup

```bash
conda create --name abdenv python=3.7
conda activate abdenv

cd abd_utils && pip install .
cd ../abd_model && pip install .
```

## End-to-End Example

Using a small Dutch town (Giethoorn) as the example AOI. Swap in any GeoJSON polygon to run detection on a different area.

```bash
# 1. Add your Bing Maps key to abd_utils/src/abd_utils/.env

# 2. Download imagery tiles
download-images --aoi input/AOI.geojson --output bing-images

# 3. Convert to model input format
images-to-abd --images bing-images/images --output abd-input

# 4. Download pre-trained model → place in input/

# 5. Run detection
abd predict \
  --config input/config.toml \
  --dataset abd-input \
  --cover abd-input/cover.csv \
  --checkpoint input/neat-fullxview-epoch75.pth \
  --out abd-predictions \
  --metatiles --keep_borders

# 6. Vectorize (raster → polygons)
abd vectorize \
  --config input/config.toml \
  --type Building \
  --masks abd-predictions \
  --out abd-predictions/buildings.geojson

# 7. Clean up geometry
filter-buildings \
  --data abd-predictions/buildings.geojson \
  --dest abd-predictions/buildings-clean.geojson
```

## Pre-trained Model

**neat-fullxview-epoch75** — AlbuNet trained on the xBD dataset for 75 epochs.

| Metric | Score |
|---|---|
| IoU (Jaccard Index) | 0.79 |
| MCC (Matthews Correlation Coefficient) | 0.75 |

## Project Structure

```
.
├── abd_model/     # Model training and inference framework
├── abd_utils/     # Image download and preprocessing utilities
├── ada_tools/     # Additional analysis tools
├── neat_eo/       # Vectorization utilities
├── input/         # AOI, config, and model checkpoint files
├── run_detection.py
├── Dockerfile
└── README.md
```

## Credits

Built on [robosat](https://github.com/mapbox/robosat) and [robosat.pink](https://github.com/acannistra/robosat.pink).  
Original development: [Ondrej Zacha](https://github.com/ondrejzacha), [Wessel de Jong](https://github.com/Wessel93), [Jacopo Margutti](https://github.com/jmargutt)
