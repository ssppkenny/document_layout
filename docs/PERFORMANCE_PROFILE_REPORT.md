# Performance Profiling Report

**Date**: February 2, 2026  
**Command**: `python src/ocr_reflow/main.py images/kf_p025.png --layout`  
**Total Execution Time**: 11.409 seconds  
**Total Function Calls**: 7,152,641 (6,977,561 primitive calls)

---

## Executive Summary

The application spent **11.4 seconds** processing a single image. The performance bottlenecks are distributed across several areas:

1. **Model Loading & Initialization** (~40-50% of time)
2. **Neural Network Inference** (~25-30% of time)
3. **I/O Operations** (~15-20% of time)
4. **Application Logic** (~5-10% of time)

---

## Top Performance Bottlenecks

### 1. **PyTorch Model Deserialization** (9.9 seconds cumulative)
- **Function**: `torch.serialization.py:persistent_load`
- **Impact**: 87% of total execution time
- **Problem**: Loading pre-trained models from disk is extremely slow
- **Recommendation**: 
  - Consider model caching/memoization
  - Use model quantization or lighter model formats
  - Pre-load models at startup if processing multiple images

### 2. **File I/O Operations** (1.7 seconds)
- **Function**: `_io.open_code` 
- **Calls**: 4,387 calls
- **Impact**: Heavy file reading during module imports and model loading
- **Recommendation**:
  - Minimize dynamic imports
  - Use compiled bytecode (.pyc) files
  - Consider using faster storage (SSD)

### 3. **Dynamic Library Loading** (1.2 seconds)
- **Function**: `_imp.create_dynamic`
- **Calls**: 208 calls
- **Impact**: Loading C extensions and compiled modules
- **Problem**: Unavoidable during first run but happens every execution
- **Recommendation**: Keep application running as a service for batch processing

### 4. **Neural Network Convolutions** (0.58 seconds self-time)
- **Function**: `torch.conv2d`
- **Calls**: 966 calls
- **Impact**: Core computation for image processing
- **Performance**: This is actually well-optimized
- **Note**: Running on CPU (not GPU)

### 5. **Tensor CPU Transfer** (0.57 seconds)
- **Function**: `torch._C.TensorBase.cpu()`
- **Calls**: 9 calls
- **Impact**: Converting tensors from device memory to CPU
- **Recommendation**: This suggests mixed device usage; ensure consistent device placement

### 6. **Image I/O** (0.24 seconds)
- **Function**: `imread` (12 calls) and `imwrite` (9 calls)
- **Impact**: Reading input image and writing output images
- **Recommendation**: Consider reducing output image writes if not needed

---

## Detailed Breakdown by Component

### Model Inference (DOCTR - Text Detection)
- **Forward passes**: 9 calls taking 0.998 seconds cumulative
- **Post-processing**: 9 calls taking 0.626 seconds
- **Convolution operations**: 966 calls, 0.577 seconds total
- **Batch normalization**: 636 calls, 0.140 seconds

### Layout Model (YOLO)
- **Forward passes**: 2 calls taking 0.767 seconds cumulative
- **Preprocessing and inference** well-optimized

### Application Logic
- **Word wrapping/reflow**: `create_page_with_word_wrapping` - 8 calls, 0.098 seconds
- **Divide & conquer algorithm**: `divide_conquer_4d` - 33 calls, 0.020 seconds
- **Red-blue dominance**: 781 calls, 0.005 seconds
- **Device detection**: 4 calls, 0.025 seconds

---

## Performance Issues Identified

### Critical Issues

1. **Model Loading Overhead** (Severity: HIGH)
   - **Problem**: Models are loaded from disk on every run
   - **Time**: ~10 seconds per execution
   - **Solution**: Implement model caching or run as persistent service

2. **No GPU Utilization** (Severity: HIGH)
   - **Problem**: Running on CPU instead of GPU
   - **Evidence**: All tensor operations use CPU
   - **Impact**: 5-10x slower than GPU execution
   - **Solution**: Ensure CUDA-enabled PyTorch and GPU availability

3. **Excessive Module Imports** (Severity: MEDIUM)
   - **Problem**: 4,387 file I/O operations for imports
   - **Impact**: ~1.7 seconds of overhead
   - **Solution**: Lazy imports, reduce dependency tree

### Minor Issues

4. **Multiple Image Writes** (Severity: LOW)
   - **Problem**: Writing 9 output images (108ms total)
   - **Solution**: Make output optional via command-line flag

5. **Tensor Device Transfers** (Severity: LOW)
   - **Problem**: 9 CPU transfers taking 0.57 seconds
   - **Solution**: Ensure consistent device placement throughout pipeline

---

## Performance by Execution Phase

### Phase 1: Initialization (0-3 seconds)
- Import dependencies: ~2.5 seconds
- Load DOCTR model: ~6 seconds  
- Load YOLO model: ~2 seconds

### Phase 2: Inference (3-11 seconds)
- Text detection (DOCTR): ~1.6 seconds
- Layout detection (YOLO): ~0.8 seconds
- Image preprocessing: ~0.2 seconds
- Post-processing: ~0.6 seconds

### Phase 3: Application Logic (11-11.4 seconds)
- Line segmentation: ~0.1 seconds
- Word wrapping: ~0.1 seconds
- Divide & conquer: ~0.02 seconds
- Output generation: ~0.15 seconds

---

## Optimization Recommendations

### Immediate Wins (Easy to Implement)

1. **Enable GPU Acceleration**
   - Expected speedup: 5-10x for inference
   - Implementation: Ensure CUDA toolkit and GPU-enabled PyTorch

2. **Model Caching**
   - Expected speedup: 10 seconds saved on subsequent runs
   - Implementation: Load models once, process multiple images

3. **Reduce Output Verbosity**
   - Expected speedup: 0.1 seconds
   - Implementation: Add flags to skip intermediate image outputs

### Medium-Term Improvements

4. **Model Quantization**
   - Expected speedup: 2-3x inference speed
   - Implementation: Use INT8 or FP16 models
   - Trade-off: Slight accuracy loss

5. **Batch Processing**
   - Expected speedup: Better throughput for multiple images
   - Implementation: Process images in batches

6. **JIT Compilation**
   - Expected speedup: 10-20% on repeated runs
   - Implementation: Use TorchScript for model compilation

### Long-Term Optimizations

7. **Lighter Models**
   - Consider MobileNet or EfficientNet backbones
   - Trade-off: Model size vs. accuracy

8. **Pipeline Optimization**
   - Parallelize independent operations
   - Use async I/O for image loading

9. **Service Architecture**
   - Run as persistent service (REST API or gRPC)
   - Eliminates initialization overhead

---

## Comparison to Baseline

### Current Performance
- Single image: 11.4 seconds
- Model loading: 87% of time
- Actual inference: 13% of time

### Expected Performance (Optimized)
- With GPU: ~1-2 seconds per image
- With model caching + GPU: ~0.5-1 second per image
- With all optimizations: ~0.3-0.5 seconds per image

**Potential Speedup**: 20-35x improvement possible

---

## Hotspot Functions

### Top 10 by Self-Time

1. `_io.open_code` - 1.713s (15%)
2. `_imp.create_dynamic` - 1.183s (10%)
3. `torch.serialization.persistent_load` - 0.613s (5%)
4. `torch._C.TensorBase.cpu` - 0.574s (5%)
5. `torch.conv2d` - 0.488s (4%)
6. `BufferedReader.read` - 0.423s (4%)
7. `torch._C.TensorBase.normal_` - 0.320s (3%)
8. `torch._C.TensorBase.to` - 0.293s (3%)
9. `marshal.loads` - 0.284s (2%)
10. `_library.custom_ops._register_to_dispatcher` - 0.232s (2%)

### Most Called Functions

1. `isinstance` - 670,970 calls
2. `len` - 569,490 calls
3. `getattr` - 308,075 calls
4. `str.startswith` - 305,348 calls
5. `list.append` - 265,110 calls

---

## Conclusion

The application's performance is dominated by **model loading overhead** (87% of execution time). The actual inference and application logic are reasonably efficient. The primary recommendation is to:

1. **Enable GPU acceleration** for 5-10x speedup on inference
2. **Implement model caching** to eliminate 10 seconds of overhead per run
3. **Run as a persistent service** for production use

With these optimizations, single-image processing time could be reduced from **11.4 seconds to ~0.5 seconds** - a **20x improvement**.
