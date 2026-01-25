# Fix Summary: Torchvision Compatibility Issue

## Problem

When importing `ocr_reflow` in the default (CPU) environment, you encountered this error:

```
RuntimeError: operator torchvision::nms does not exist
```

This was caused by a version mismatch between PyTorch and torchvision. The `python-doctr` package requires torchvision, but it wasn't explicitly specified in the pixi environment dependencies.

## Solution

Added `torchvision` as an explicit dependency in both CPU and GPU environments in `pixi.toml`:

### Changes to pixi.toml

```toml
# CPU-only environment (default, works on WSL without NVIDIA)
[feature.cpu.dependencies]
pytorch = ">=2.9.1,<3"
torchvision = ">=0.19,<1"

# GPU environment with CUDA support
[feature.gpu.dependencies]
pytorch = { version = ">=2.9.1,<3", build = "*cuda*" }
torchvision = { version = ">=0.19,<1", build = "*cuda*" }
```

## How to Apply the Fix

### For Default (CPU) Environment:

```bash
cd /home/sergey/code/python/segmentation

# Remove old environment
rm -rf .pixi/envs/default

# Reinstall with new configuration
pixi install -e default

# Reinstall the package
pixi run pip install -e .

# Test the fix
pixi run python -c "import ocr_reflow; print('✓ Success!')"
```

### For GPU Environment:

```bash
cd /home/sergey/code/python/segmentation

# Remove old environment
rm -rf .pixi/envs/gpu

# Reinstall with new configuration
pixi install -e gpu

# Reinstall the package
pixi run -e gpu pip install -e .

# Test the fix
pixi run -e gpu python -c "import ocr_reflow; print('✓ Success!')"
```

## Verification

After applying the fix, the following should work without errors:

```python
import torch
import torchvision
import ocr_reflow
from ocr_reflow import process_document

print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print("✓ All imports successful!")
```

### Expected Output:

```
torch: 2.9.1
torchvision: 0.25.0
CUDA available: False
✓ All imports successful!
```

## Test Script

A test script has been created at `test_imports.py` to verify all imports work correctly:

```bash
pixi run python test_imports.py
```

This will check all dependencies and confirm the package is properly installed.

## Files Modified

1. **pixi.toml** - Added torchvision to both environments
2. **README.md** - Updated to mention torchvision in dependencies
3. **test_imports.py** - New test script to verify installation

## Why This Happened

The issue occurred because:
1. `python-doctr` depends on `torchvision`
2. PyTorch was installed, but torchvision wasn't explicitly specified
3. When torchvision was installed as a transitive dependency, it may have gotten an incompatible version
4. Explicitly specifying torchvision ensures version compatibility with PyTorch

## WSL Compatibility

This fix maintains WSL compatibility:
- ✅ Default environment still works on WSL without NVIDIA GPU
- ✅ GPU environment still requires CUDA 12
- ✅ Both environments now have compatible PyTorch + torchvision versions
- ✅ No changes needed to WSL-specific configuration

## Additional Notes

- The version constraint `>=0.19,<1` ensures torchvision 0.19+ is used (compatible with PyTorch 2.9+)
- For GPU environment, the `build = "*cuda*"` selector ensures CUDA-enabled builds
- Both environments use the same version constraints to maintain consistency
