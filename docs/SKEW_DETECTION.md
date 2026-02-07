# Skew Detection and Correction

## Overview

The OCR Reflow package now includes automatic skew detection and correction based on the MCCSD (Modified Cross-Correlation Skew Detection) algorithm from the paper "A Robust Skew Detection Algorithm for Grayscale Document Image" by Ming Chen and Xiaoqing Ding, Tsinghua University.

Skew detection is automatically applied **before** any OCR or layout analysis processing, ensuring that all subsequent operations work on properly aligned text.

## Algorithm Description

The MCCSD algorithm:

1. **Uses Cross-Correlation**: Calculates both horizontal (HCC) and vertical (VCC) cross-correlation to detect text line patterns
2. **Handles Multiple Layouts**: Automatically detects whether text is horizontally or vertically oriented (useful for Chinese/Japanese documents)
3. **Region-Based Sampling**: Uses randomly selected regions for efficiency and robustness
4. **Peak Detection**: Identifies the skew angle from correlation peaks
5. **Auxiliary Peak Resolution**: Uses multiple distance parameters to resolve ambiguous peaks

### Key Parameters

- **d**: Distance between lines for correlation (default: 75 pixels)
- **s_range**: Range of shift values to test, determines detection range (default: 25, gives ±18.43°)
- **d_prime**: Alternative distance for auxiliary peak resolution (default: 50 pixels)
- **region_size**: Size of randomly selected regions (default: 150×150 pixels)
- **num_regions**: Number of regions to analyze (default: 9)

## Usage

### Automatic Usage (Integrated)

Skew correction is automatically applied when using the main processing functions:

```python
from ocr_reflow import process_document, process_document_with_layout

# Skew correction is applied automatically
result = process_document_with_layout("image.png", zoom_factor=2.5)
```

### Manual Usage

You can also use the skew detection functions directly:

```python
from ocr_reflow import detect_skew, detect_and_correct_skew, rotate_image
import cv2

# Load image
img = cv2.imread("document.png")

# Detect skew angle only
angle = detect_skew(img)
print(f"Detected skew: {angle:.2f}°")

# Detect and correct skew in one step
corrected_img, detected_angle = detect_and_correct_skew(img)
print(f"Corrected by {detected_angle:.2f}°")

# Or rotate manually with a known angle
rotated_img = rotate_image(img, angle=-2.5)
```

### Command Line

When using the command-line interface, skew correction is applied automatically:

```bash
# With layout analysis
python src/ocr_reflow/main.py input.png --layout

# Standard processing
python src/ocr_reflow/main.py input.png
```

### Test Script

A standalone test script is provided to visualize skew detection:

```bash
python testscripts/test_skew_detection.py input.png [output.png]
```

This creates:
- A deskewed image
- A side-by-side comparison showing original vs corrected

## Performance

- **Detection Speed**: Typically 0.5-3 seconds depending on image size and text coverage
- **Adaptive**: Automatically skips unsuitable regions (e.g., pictures) to speed up processing
- **Detection Range**: ±18.43° (configurable via s_range parameter)
- **Detection Accuracy**: 0.76° (determined by arctan(1/d))

## Technical Details

### Cross-Correlation Functions

**Vertical Cross-Correlation (VCC)**: Used for horizontal text
```
R_V(s) = Σ_x0 Σ_y I(x0, y) * I(x0 + d, y + s)
```

**Horizontal Cross-Correlation (HCC)**: Used for vertical text
```
R_H(s) = Σ_y0 Σ_x I(x, y0) * I(x + s, y0 + d)
```

### Total Variation

The algorithm calculates the total variation to determine which correlation function has distinct peaks:
```
ΔR(±S) = Σ |R(s+1) - R(s)|
```

The function with larger total variation indicates the dominant text orientation.

### Skew Angle Calculation

Once the primary peak s_p is found:
```
α = arctan(s_p / d)
```

## Implementation Notes

1. **Preprocessing**: Skew correction is applied BEFORE layout analysis and OCR
2. **Temporary Files**: The corrected image is saved temporarily for processing, then cleaned up automatically
3. **Grayscale Conversion**: The algorithm works on grayscale images; color images are converted automatically
4. **Background Color**: Rotation uses white background by default (configurable)
5. **Image Expansion**: The rotated image is expanded to fit all content without cropping

## Limitations

- Detection range: ±18.43° by default (can be increased by adjusting s_range)
- Requires clear text lines for accurate detection
- May not work well on documents with:
  - Very large skew angles (>20°)
  - No distinct text lines (e.g., pure graphics)
  - Very dense or overlapping text

## References

Chen, M., & Ding, X. "A Robust Skew Detection Algorithm for Grayscale Document Image." 
Department of Electronics Engineering, Tsinghua University, Beijing, China.

## See Also

- [Main Documentation](README.md)
- [Processing Flow Diagram](PROCESSING_FLOW_DIAGRAM.md)
- [Quick Reference](QUICK_REFERENCE.md)
