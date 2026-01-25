# Windows WSL Compatibility

This document explains how the project is configured to work on Windows WSL (Windows Subsystem for Linux) without requiring an NVIDIA GPU.

## Overview

The project now supports two Pixi environments:
1. **CPU Environment** (default) - Works on all systems, including WSL without NVIDIA GPU
2. **GPU Environment** - For systems with NVIDIA GPU and CUDA 12 drivers

## Environment Configuration

### CPU Environment (Default)

**File**: `pixi.toml`

```toml
[feature.cpu.dependencies]
pytorch = ">=2.9.1,<3"
cpuonly = "*"

[environments]
default = ["cpu"]
```

**Installation**:
```bash
pixi install
# or explicitly
pixi install -e default
```

**Use Cases**:
- Windows WSL without NVIDIA GPU
- Systems without CUDA support
- Development environments where GPU is not needed
- CI/CD pipelines

### GPU Environment

**File**: `pixi.toml`

```toml
[feature.gpu.dependencies]
pytorch = { version = ">=2.9.1,<3", build = "*cuda*" }

[feature.gpu.system-requirements]
cuda = "12"

[environments]
gpu = ["gpu"]
```

**Installation**:
```bash
pixi install -e gpu
```

**Requirements**:
- NVIDIA GPU
- CUDA 12 drivers installed
- For WSL: CUDA drivers must be installed in WSL

## How It Works

### On WSL Without NVIDIA GPU

1. Pixi treats WSL as a native Linux environment (linux-64 platform)
2. The default CPU environment installs PyTorch with CPU-only support
3. The `cpuonly` package explicitly excludes CUDA dependencies
4. No CUDA system requirements are checked for the default environment

### On WSL With NVIDIA GPU

If you have NVIDIA GPU support in WSL:
1. Install CUDA drivers in WSL (separate from Windows drivers)
2. Use `pixi install -e gpu` to install the GPU environment
3. PyTorch will detect and use the GPU automatically

## Benefits

### Before (Single Environment)

- **Problem**: Required CUDA 13.0 system requirement for all installations
- **Impact**: Failed on WSL without NVIDIA GPU
- **Workaround**: Manual modification of `pixi.toml`

### After (Feature-Based Environments)

- **Solution**: Two separate environments with different requirements
- **Default**: CPU-only, works everywhere
- **Optional**: GPU support when available
- **No modification needed**: Works out of the box on WSL

## Switching Environments

### From CPU to GPU

```bash
# Install GPU environment
pixi install -e gpu

# Activate GPU environment
pixi shell

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### From GPU to CPU

```bash
# Install CPU environment
pixi install -e default

# Activate CPU environment
pixi shell

# Verify CPU mode
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Technical Details

### Package Selection

**CPU Environment**:
- `pytorch` package from conda-forge
- No CUDA-specific builds
- No CUDA system requirements

**GPU Environment**:
- `pytorch` package from conda-forge with CUDA build selector
- Build filter: `*cuda*` to select CUDA-enabled builds
- System requirement: `cuda = "12"`

### Environment Isolation

- Each environment has its own prefix: `.pixi/envs/default` and `.pixi/envs/gpu`
- Environments don't interfere with each other
- Can switch between environments without conflicts

## Testing

### Test CPU Environment

```bash
pixi install -e default
pixi run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CPU only: {not torch.cuda.is_available()}')
"
```

Expected output:
```
PyTorch version: 2.9.1 (or later)
CUDA available: False
CPU only: True
```

### Test GPU Environment (if CUDA available)

```bash
pixi install -e gpu
pixi run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

Expected output (with GPU):
```
PyTorch version: 2.9.1 (or later)
CUDA available: True
CUDA version: 12.x
GPU count: 1 (or more)
```

## Troubleshooting

### Environment Already Exists

If you get an error about existing environment:

```bash
# Remove existing environment
rm -rf .pixi/envs/default
# or
rm -rf .pixi/envs/gpu

# Reinstall
pixi install -e default
```

### CUDA Not Found in GPU Environment

1. Check CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Verify system requirements:
   ```bash
   pixi info
   ```

3. Switch to CPU environment if CUDA is not available:
   ```bash
   pixi install -e default
   ```

### Performance on CPU

CPU processing is slower than GPU but still functional:
- Small images (< 2MB): Acceptable performance
- Large images (> 5MB): May take several minutes
- Consider using GPU environment or cloud GPU for large batches

## References

- [Pixi Documentation](https://pixi.sh)
- [Pixi Features](https://pixi.sh/latest/features/environment/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [WSL CUDA Support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
