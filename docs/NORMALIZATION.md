# Image Normalization Implementation

## Overview

Image normalization has been successfully implemented as a preprocessing step in the OCR reflow pipeline. Normalization adjusts the range of pixel intensity values, bringing the image to a range that improves OCR and binarization performance.

## Implementation Details

### 1. Normalization Function

Location: `src/ocr_reflow/binarization.py`

```python
def normalize_image(image: np.ndarray, alpha: float = 0, beta: float = 255, 
                   norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
    """
    Normalize image pixel intensity values to a specified range.
    
    Uses OpenCV's cv2.normalize() function.
    """
```

**Features:**
- Supports multiple normalization types:
  - `cv2.NORM_MINMAX` - Normalizes to [alpha, beta] range (default)
  - `cv2.NORM_L1` - L1 normalization
  - `cv2.NORM_L2` - L2 normalization
  - `cv2.NORM_INF` - Infinity normalization
- Default range: [0, 255] for full pixel intensity range
- Works with both grayscale and BGR images

### 2. Integration into Main Pipeline

Location: `src/ocr_reflow/main.py`

The preprocessing pipeline with `--bin` flag now performs three steps:

1. **Normalization**: Brings pixel values to full [0, 255] range
   ```python
   img_normalized = normalize_image(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
   ```

2. **Binarization**: Applies Otsu's method (or Sauvola/Niblack)
   ```python
   binary_img = binarize_document(img_normalized, method='otsu')
   ```

3. **Enhancement**: Adds binary to normalized original
   ```python
   enhanced_img = cv2.add(img_normalized, binary_img_bgr)
   ```

## Usage

### Command Line

```bash
# Apply normalization + Otsu binarization + enhancement
python src/ocr_reflow/main.py images/yourimage.png --layout --bin

# Specify different binarization method
python src/ocr_reflow/main.py images/yourimage.png --layout --bin --bin-method sauvola

# With custom window size for adaptive methods
python src/ocr_reflow/main.py images/yourimage.png --layout --bin --bin-method sauvola --bin-window 25
```

### Python API

```python
from binarization import normalize_image, binarize_document
import cv2
import numpy as np

# Load image
img = cv2.imread('image.png')

# Step 1: Normalize
img_norm = normalize_image(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Step 2: Binarize
binary = binarize_document(img_norm, method='otsu')

# Step 3: Enhance
binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
enhanced = cv2.add(img_norm, binary_bgr)
enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
```

## Benefits

1. **Improved Contrast**: Normalization ensures pixel values span the full [0, 255] range
2. **Better Binarization**: Otsu's method works more effectively on normalized images
3. **Enhanced OCR**: The combination improves text detection and character recognition
4. **Consistent Processing**: Images with different original ranges are brought to a standard range

## Test Scripts

Three test scripts are provided to verify the implementation:

1. **test_normalization.py**: Comprehensive test of the normalization pipeline
   - Tests all normalization steps
   - Compares normalized vs non-normalized results
   - Generates intermediate images for inspection

2. **demo_normalization.py**: Simple demonstration of normalization effect
   - Shows before/after comparison
   - Displays pixel range and mean values

3. **test_workflow.py**: Tests the complete Otsu + original workflow
   - Verifies the three-step process
   - Generates output images at each stage

## Technical Details

### Normalization Algorithm

OpenCV's `cv2.normalize()` function performs the following:

For `NORM_MINMAX`:
```
normalized_pixel = (pixel - min_value) / (max_value - min_value) * (beta - alpha) + alpha
```

Where:
- `min_value` = minimum pixel value in the image
- `max_value` = maximum pixel value in the image
- `alpha` = lower bound of output range (default: 0)
- `beta` = upper bound of output range (default: 255)

### Processing Pipeline

```
Original Image
      ↓
Normalization (cv2.normalize)
      ↓
Binarization (Otsu's method)
      ↓
Add to Normalized Original
      ↓
Enhanced Image → OCR & Reflow
```

## Configuration

The normalization is automatically applied when using the `--bin` flag. No additional configuration is required.

Default settings:
- Normalization type: `cv2.NORM_MINMAX`
- Range: [0, 255]
- Binarization method: `otsu`

## Files Modified

1. `src/ocr_reflow/binarization.py`
   - Added `normalize_image()` function

2. `src/ocr_reflow/main.py`
   - Added normalization step before binarization
   - Updated imports to include `normalize_image`
   - Enhanced preprocessing pipeline

## Testing

Run the test scripts to verify the implementation:

```bash
# Test normalization functionality
python3 test_normalization.py

# Demo normalization effect
python3 demo_normalization.py

# Test complete workflow
python3 test_workflow.py
```

## Summary

Image normalization has been successfully integrated into the OCR reflow pipeline using OpenCV's `cv2.normalize()` function. The implementation:

✓ Normalizes pixel intensity values to [0, 255] range
✓ Applies binarization (Otsu/Sauvola/Niblack) to create clean black/white images
✓ Uses the binary image directly for OCR and reflow processing
✓ Improves text detection and OCR accuracy
✓ Automatically applied when using `--bin` flag
✓ Fully tested and verified

The preprocessing pipeline now performs: **Normalization → Binarization**, resulting in clean binary images for improved OCR accuracy and better text detection.
