import os
from pathlib import Path

# Define the directory structure
folders = [
    "data/raw/modis",
    "data/raw/viirs",
    "data/raw/images",
    "data/processed",
    "data/weather",
    "src/image_module",
    "src/hotspot_module",
    "src/risk_module",
    "src/spread_module",
    "src/fusion",
    "src/utils",
    "notebooks",
    "app/backend",
    "app/frontend",
    "models"
]

# Define the initial empty files to create
files = [
    "src/image_module/vit_model.py",
    "src/image_module/preprocess.py",
    "src/hotspot_module/modis_loader.py",
    "src/hotspot_module/hotspot_detector.py",
    "src/risk_module/rf_model.py",
    "src/risk_module/weather_features.py",
    "src/spread_module/cnn_lstm.py",
    "src/spread_module/sequence_builder.py",
    "src/fusion/decision_engine.py",
    "src/utils/helpers.py",
    "notebooks/EDA.ipynb",
    "notebooks/experiments.ipynb",
    "requirements.txt",
    "main.py"
]

print("Creating project structure...")

# Create directories
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {folder}")

# Create empty files
for file in files:
    Path(file).touch(exist_ok=True)
    print(f"Created file: {file}")

print("\nSuccess! Project structure is ready.")