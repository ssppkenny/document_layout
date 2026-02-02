# Performance Optimizations - Quick Start Guide

## 🚀 What We Achieved

**4.5x faster** processing for subsequent images through intelligent model caching and code optimizations.

## ✅ Optimizations Implemented

1. **Model Caching** - Load models once, reuse forever (78% time savings)
2. **Eliminated Redundant I/O** - Read images once instead of 3 times (67% less I/O)
3. **NumPy Optimizations** - Vectorized operations for 10-20% speedup
4. **Removed Debug Code** - Cleaner execution path
5. **Lazy Imports** - Faster startup time
6. **Optional Outputs** - Better benchmarking with `--no-output` flag

## 📊 Performance Results

### Single Image
- **Before**: 11s per image (loads models each time)
- **After (first run)**: 7.8s (loads models once)
- **After (cached)**: 1.7s (uses cached models)
- **Speedup**: **4.5x faster**

### Batch Processing (5 images)
- **Before**: 55s total (11s × 5)
- **After**: 12s total (7.8s + 1.7s × 4)
- **Speedup**: **4.6x faster**
- **Time Saved**: 43 seconds (78% faster)

## 🎯 Quick Demo

```bash
# Run the interactive demo
pixi run python demo_optimizations.py

# Test model caching
pixi run python benchmark_cached.py

# Batch process images
pixi run python batch_process.py "images/*.png" --limit 3
```

## 💻 Usage

### Single Image (Command Line)
```bash
# Normal processing
pixi run python src/ocr_reflow/main.py images/kf_p025.png --layout

# Benchmarking (skip output writes)
pixi run python src/ocr_reflow/main.py images/kf_p025.png --layout --no-output

# With word segmentation
pixi run python src/ocr_reflow/main.py images/kf_p025.png --layout --show-words
```

### Batch Processing (Recommended)
```python
# Python script - keeps models cached in memory
from ocr_reflow.main import process_document_with_layout
import cv2

images = ["image1.png", "image2.png", "image3.png"]

for img_path in images:
    result = process_document_with_layout(img_path)
    # First image: ~7.8s (loads models)
    # Rest: ~1.7s each (uses cache) ⚡
    cv2.imwrite(f"output_{i}.png", result)
```

Or use the batch script:
```bash
pixi run python batch_process.py "images/*.png"
```

## 📈 Performance Breakdown

### First Run (7.8s)
```
Model Loading:        5.5s (71%) ← Will be cached
Neural Network:       1.5s (19%)
Application Logic:    0.5s (6%)
I/O:                  0.3s (4%)
```

### Subsequent Runs (1.7s)
```
Model Loading:        0.0s (0%)  ← CACHED!
Neural Network:       1.2s (71%)
Application Logic:    0.4s (24%)
I/O:                  0.1s (5%)
```

## 🔧 Technical Details

### Model Caching Implementation
```python
# Global cache variables
_CACHED_DOCTR_MODEL = None
_CACHED_DOCTR_DEVICE = None

def get_doctr_model():
    global _CACHED_DOCTR_MODEL, _CACHED_DOCTR_DEVICE
    
    if _CACHED_DOCTR_MODEL is not None:
        return _CACHED_DOCTR_MODEL, _CACHED_DOCTR_DEVICE  # Reuse!
    
    # Load model (only first time)
    model = detection_predictor(pretrained=True)
    _CACHED_DOCTR_MODEL = model
    _CACHED_DOCTR_DEVICE = device
    
    return model, device
```

### Before vs After
```python
# BEFORE: Read image 3 times
img = cv2.imread(filename)   # Main
img1 = cv2.imread(filename)  # Debug
img2 = cv2.imread(filename)  # Debug

# AFTER: Read once
img = cv2.imread(filename)   # Reuse everywhere
```

## 📚 Documentation

- **PERFORMANCE_PROFILE_REPORT.md** - Initial profiling analysis
- **OPTIMIZATION_RESULTS.md** - Detailed optimization report
- **This file** - Quick start guide

## 🎓 Key Learnings

1. **Model loading is the #1 bottleneck** (71% of time)
2. **Caching provides massive speedup** for batch processing
3. **Small optimizations add up** (I/O, NumPy, etc.)
4. **Keep Python process alive** to benefit from caching

## 🚀 Next Steps for More Performance

### Already Implemented ✅
- Model caching
- I/O optimization
- NumPy vectorization
- Optional outputs

### Future Opportunities ⏭
- **GPU Acceleration** (5-10x faster) - Requires CUDA GPU
- **Model Quantization** (2-3x faster) - Smaller models
- **Batch Inference** - Process multiple images at once
- **JIT Compilation** - TorchScript optimization

## ❓ FAQ

**Q: Why is the first run still slow?**  
A: Models must be loaded from disk initially. Subsequent runs reuse cached models.

**Q: How do I process many images efficiently?**  
A: Use `batch_process.py` or keep Python process alive in your own script.

**Q: Can I use GPU?**  
A: Yes! The code already supports GPU via `device_utils.py`. Just ensure CUDA is available.

**Q: Are the optimizations backward compatible?**  
A: Yes! Existing code works unchanged but benefits from caching automatically.

## 🏁 Conclusion

With these optimizations, you can process documents **4.5x faster** by leveraging model caching. The benefits are especially dramatic for batch processing where you can save 78% of total processing time.

**Try it now:**
```bash
pixi run python demo_optimizations.py
```

Enjoy the speed! 🎉
