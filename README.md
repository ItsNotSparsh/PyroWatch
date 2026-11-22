# PyroWatch

## Install and run

Open Command Prompt in the project folder and create a virtual environment:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains only the packages needed to run the app. PyTorch is installed separately so you can choose CPU or GPU. For NVIDIA GPU setup, see the optional section below.

Start the app:

```cmd
python -m streamlit run app\app.py
```

## Optional: NVIDIA GPU acceleration

The standard requirements install works on CPU. If you have a CUDA-capable NVIDIA GPU and want PyTorch GPU acceleration, install the CUDA-enabled PyTorch packages **after** installing the requirements, while `.venv` is activated:

```cmd
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

This replaces the CPU-only PyTorch packages in that virtual environment; do not install a separate CPU build afterward. A current NVIDIA driver is required. Check that PyTorch sees the GPU with:

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

The app automatically uses CUDA when available and otherwise runs on CPU. The PyTorch CUDA wheel includes its CUDA runtime, so a separate CUDA Toolkit installation is generally not needed.

## API keys

Create a `.env` file in the project root for optional live weather and hotspot fetching:

```env
OPENWEATHER_KEY=your_openweather_key
NASA_FIRMS_KEY=your_nasa_firms_key
```

Do not commit `.env` or share its keys.

## Optional: retrain the Vision Transformer

If you want to retrain the image model, install the extra training dependencies in the active environment:

```cmd
python -m pip install -r requirements-training.txt
```

Place your image dataset under `data\raw\training_images` in class subfolders (for example, `fire` and `no_fire`), then run:

```cmd
python -m src.image_module.train_vit
```

Training can use the same CPU or CUDA-enabled PyTorch installation selected above. The Random Forest training script uses packages already listed in `requirements.txt` and can be run with `python -m src.risk_module.rf_model`.
