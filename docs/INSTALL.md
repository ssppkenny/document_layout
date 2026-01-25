# Installation Guide

This document provides detailed installation instructions for the `ocr-reflow` package.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Verification](#verification)
4. [Usage](#usage)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Recommended: 4GB+ RAM for processing large images
- Optional: CUDA-compatible GPU for faster processing

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

This method is ideal if you want to modify the code or contribute to the project.

```bash
# 1. Clone the repository
git clone <repository-url>
cd segmentation

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# 4. Install in editable mode with all dependencies
pip install -e .

# 5. Install with development dependencies (includes Jupyter)
pip install -e ".[dev]"
```

### Method 2: Using Pixi (For Complete Environment Management)

[Pixi](https://pixi.sh) is a package manager that handles the entire environment including system dependencies.

```bash
# 1. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone the repository
git clone <repository-url>
cd segmentation

# 3. Install dependencies via Pixi
# For CPU-only (default, works on WSL without NVIDIA GPU):
pixi install

# For GPU with CUDA support:
pixi install -e gpu

# 4. Install the package in the Pixi environment
pixi run pip install -e .

# 5. Verify installation
pixi run python -c "from ocr_reflow import process_document; print('Success!')"
```

#### Pixi Environments

The project provides two environments:

**CPU Environment (default):**
- Works on all systems, including Windows WSL without NVIDIA GPU
- No CUDA dependencies
- Command: `pixi install` or `pixi install -e default`

**GPU Environment:**
- Requires NVIDIA GPU with CUDA 12 drivers
- Includes PyTorch with CUDA support
- Command: `pixi install -e gpu`

**Windows WSL Users:** The default CPU environment works perfectly on WSL without any special configuration. Only use the GPU environment if you have CUDA drivers installed in WSL.

### Method 3: Install from PyPI (Future)

Once published to PyPI, you'll be able to install with:

```bash
pip install ocr-reflow
```

## Verification

After installation, verify that everything works:

```bash
# Test the installation
python test_package.py

# Test the CLI
ocr-reflow --help

# Test importing in Python
python -c "from ocr_reflow import process_document; print('Package loaded successfully!')"
```

## Usage

### Command Line

```bash
# Process a single image
ocr-reflow input.png

# Specify output filename
ocr-reflow input.png output.png

# Process multiple files
for file in images/*.png; do
    ocr-reflow "$file"
done
```

### Python Script

```python
from ocr_reflow import process_document
import cv2

# Process a document
result = process_document("document.png")

# Save the result
cv2.imwrite("output.png", result)
```

### Jupyter Notebook

```python
from ocr_reflow import process_document
import cv2
from matplotlib import pyplot as plt

# Process
result = process_document("document.png")

# Display
plt.figure(figsize=(12, 16))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

For more Jupyter examples, see [JUPYTER_GUIDE.md](JUPYTER_GUIDE.md) and the example notebook at `notebooks/example_usage.ipynb`.

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'ocr_reflow'`

**Solution**:
```bash
# Make sure you installed the package
pip install -e .

# Check if it's in the list
pip list | grep ocr-reflow

# If using Jupyter, restart the kernel
```

### Dependency Issues

**Problem**: Missing dependencies like `python-doctr` or `opencv-python`

**Solution**:
```bash
# Reinstall with all dependencies
pip install -e .

# Or install dependencies individually
pip install numpy opencv-python matplotlib scipy python-doctr shapely
```

### CUDA/GPU Issues

**Problem**: CUDA errors or GPU not being used

**Solution**:
The package works on CPU. GPU acceleration is optional and handled automatically by PyTorch. 

**If using Pixi:**
```bash
# Switch to CPU environment (works on all systems including WSL)
pixi install -e default
pixi shell

# Or for GPU support (requires NVIDIA GPU with CUDA 12)
pixi install -e gpu
pixi shell
```

**If using pip/venv:**
```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Windows WSL Users**: Use the default CPU environment unless you have CUDA drivers installed in WSL.

### Permission Errors (Linux)

**Problem**: Permission denied when installing

**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .

# Or use user installation
pip install -e . --user
```

### Pixi Environment Issues

**Problem**: Package not found after `pixi install`

**Solution**:
```bash
# Install the package explicitly in Pixi environment
pixi run pip install -e .

# Or use the full path
$(pixi shell-hook | grep CONDA_PREFIX | cut -d= -f2 | tr -d "'")/bin/pip install -e .
```

### File Not Found Errors

**Problem**: CLI says "File not found"

**Solution**:
```bash
# Use absolute paths
ocr-reflow /full/path/to/document.png

# Or navigate to the directory first
cd /path/to/images
ocr-reflow document.png
```

## Getting Help

If you encounter issues not covered here:

1. Check the [README.md](../README.md) for general documentation
2. Look at the example notebook: `notebooks/example_usage.ipynb`
3. Review the [JUPYTER_GUIDE.md](JUPYTER_GUIDE.md) for Jupyter-specific help
4. Open an issue on GitHub with:
   - Your Python version (`python --version`)
   - Your OS
   - Full error message
   - Steps to reproduce the problem

## Uninstallation

To remove the package:

```bash
# Using pip
pip uninstall ocr-reflow

# Also remove virtual environment if created
rm -rf venv/  # Linux/Mac
# or
rmdir /s venv  # Windows
```
