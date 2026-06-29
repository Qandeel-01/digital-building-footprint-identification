import os
from abd_model import predict, vectorize
from abd_utils.src.abd_utils.filter_buildings import main as filter_buildings

# Verify required files exist
REQUIRED_FILES = [
    "input/config.toml",
    "input/neat-fullxview-epoch75.pth",
    "abd-input/cover.csv"
]

for file in REQUIRED_FILES:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing required file: {file}")

# Run detection pipeline
try:
    predict(
        config="input/config.toml",
        dataset="abd-input",
        cover="abd-input/cover.csv",
        checkpoint="input/neat-fullxview-epoch75.pth",
        out="abd-predictions",
        metatiles=True,
        keep_borders=True
    )
    
    vectorize(
        config="input/config.toml",
        type="Building",
        masks="abd-predictions",
        out="abd-predictions/buildings.geojson"
    )
    
    filter_buildings([
        "--data", "abd-predictions/buildings.geojson",
        "--dest", "abd-predictions/buildings-clean.geojson"
    ])
    
    print("\nSUCCESS! Results saved to: abd-predictions/")
    print("Final output: abd-predictions/buildings-clean.geojson")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("Check if:")
    print("- All input files exist")
    print("- You activated your conda environment")
    print("- Required packages are installed (torch, gdal, etc.)")