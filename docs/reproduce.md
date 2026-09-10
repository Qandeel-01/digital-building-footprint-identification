# Reproducing a run from a clean clone

Target environment: Windows 11, miniconda3, NVIDIA GPU (Blackwell / sm_120
or newer). CPU-only reproduction is possible for every stage *except*
`abd predict`.

## 1. Clone

```powershell
git clone --recurse-submodules <this-repo-url> building-footprint-benchmark
cd building-footprint-benchmark
```

If you forgot `--recurse-submodules`:

```powershell
git submodule update --init --recursive
```

Verify the submodule is on the pinned SHA:

```powershell
git -C external/abd rev-parse HEAD
# expected: e1668a470b94252040f27d26098826c293fbb46d
```

## 2. Create the conda env

```powershell
conda env create -f environment.yml
conda activate bfbench
```

Then install the two upstream packages from the submodule and this
project's package:

```powershell
pip install -e ./external/abd/abd_model ./external/abd/abd_utils
pip install -e .
```

## 3. Verify the toolchain

```powershell
bfbench verify
```

Expected output on the target machine:

- `OK  abd -> <path>\bfbench\Scripts\abd.exe`
- `OK  torch 2.11.0+cu128 cuda_build=12.8`
- `OK  torch.cuda.is_available()=True`
- `WARN torch.distributed.is_nccl_available()=False.` — expected on
  Windows; `abd predict` will need to run under WSL2. See
  [environment-notes.md](environment-notes.md).

## 4. Fetch the checkpoint

```powershell
python scripts/fetch_weights.py
```

This downloads `neat-fullxview-epoch75.pth` (~300 MB) into `weights/`.
The file's SHA-256 is recorded on first download; subsequent runs verify
against the recorded digest and refuse to run with a mismatched file.

## 5. Run the pipeline

Fetch + adapt stages run natively on Windows:

```powershell
bfbench run configs/spacenet_vegas_sample.yaml --stop-after adapt
```

Predict stage on Windows (see NCCL note above): run this block inside
WSL2 Ubuntu with the same env activated (or a Linux Docker container):

```bash
# inside WSL, from the repo root
conda activate bfbench   # or source your Linux env
bfbench run configs/spacenet_vegas_sample.yaml --stop-after predict
```

Vectorize + filter can then run either OS:

```powershell
bfbench run configs/spacenet_vegas_sample.yaml
```

Because each stage is idempotent w.r.t. its outputs on disk, re-running
`bfbench run` picks up where the previous stage left off.

## 6. QA figure

```powershell
python scripts/qa_figure.py `
    --image runs/spacenet_vegas_sample/spacenet/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img101.tif `
    --pred runs/spacenet_vegas_sample/predicted.geojson `
    --ref runs/spacenet_vegas_sample/spacenet/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img101.geojson `
    --out runs/spacenet_vegas_sample/qa.png
```

## What's mine vs. what's upstream

See [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md) for the file-by-file breakdown.
