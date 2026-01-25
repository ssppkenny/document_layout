# Quick Reference: Pixi Environments

## Installation Commands

### CPU Environment (Default - Works on WSL without GPU)
```bash
pixi install
# or explicitly
pixi install -e default

# After installation, install the package
pixi run pip install -e .
```

### GPU Environment (Requires NVIDIA GPU + CUDA 12)
```bash
pixi install -e gpu

# After installation, install the package
pixi run -e gpu pip install -e .
```

## Usage

### Activate Environment
```bash
pixi shell
```

### Run Commands in Environment
```bash
# Without activating shell
pixi run python your_script.py
pixi run jupyter lab

# After activating shell
python your_script.py
jupyter lab
```

## Testing Installation

### Quick Test
```bash
pixi run python -c "import ocr_reflow; print('✓ Success!')"
```

### Comprehensive Test
```bash
pixi run python test_imports.py
```

## Checking Environment

### View Environment Info
```bash
pixi info
```

### List Packages
```bash
# In default environment
pixi list -e default

# In GPU environment
pixi list -e gpu
```

### Check PyTorch CUDA Status
```bash
# In activated shell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Switching Environments

### From CPU to GPU
```bash
pixi install -e gpu
pixi shell
```

### From GPU to CPU
```bash
pixi install -e default
pixi shell
```

## Troubleshooting

### Import Errors (torchvision)

If you get `RuntimeError: operator torchvision::nms does not exist`:

```bash
# Reinstall environment
rm -rf .pixi/envs/default
pixi install -e default
pixi run pip install -e .
```

### Package Not Found

If `import ocr_reflow` fails:
```bash
# Make sure package is installed
pixi run pip install -e .

# Verify installation
pixi run pip list | grep ocr-reflow
```

### Environment Already Exists
```bash
rm -rf .pixi/envs/default  # or gpu
pixi install -e default     # or gpu
pixi run pip install -e .
```

### Update Dependencies
```bash
pixi update
```

### Clean and Reinstall
```bash
rm -rf .pixi
pixi install
```

## Windows WSL Notes

- **Default environment** works without any NVIDIA drivers
- **GPU environment** requires CUDA drivers in WSL (not just Windows)
- Check CUDA in WSL: `nvidia-smi`
- Install CUDA in WSL: Follow [NVIDIA WSL CUDA Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

## Key Differences

| Feature | CPU Environment | GPU Environment |
|---------|----------------|-----------------|
| **Command** | `pixi install` | `pixi install -e gpu` |
| **CUDA Required** | No | Yes (CUDA 12) |
| **WSL Compatible** | Always | Only with CUDA drivers |
| **PyTorch Build** | CPU-only | CUDA-enabled |
| **torchvision Build** | CPU-only | CUDA-enabled |
| **Performance** | Slower | Faster with GPU |
| **Use Case** | Development, testing, no GPU | Production with GPU |

## Dependencies Installed

Both environments include:
- Python 3.12
- PyTorch 2.9.1+
- torchvision 0.19+
- NumPy, SciPy, Matplotlib
- JupyterLab, ipywidgets
- Shapely, Black
- python-doctr (PyPI)
- opencv-python (PyPI)
- doclayout-yolo (PyPI)

Plus your package: **ocr-reflow** (installed via `pip install -e .`)

