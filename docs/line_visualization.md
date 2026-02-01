# Line Detection Visualization Enhancement

## Summary

Added comprehensive visualization capabilities for detected text lines, showing leftmost and rightmost points along with the detected baselines.

## Changes Made

### 1. Enhanced `verify_out0_fix.py`
- Added visualization of detected lines with colored markers
- Blue circles mark leftmost points of each line
- Yellow circles mark rightmost points of each line
- Colored lines show the detected text baselines
- Gray rectangles show individual detected words
- Output saved to `out0_line_detection_visualization.png`

### 2. Added `visualize_detected_lines()` Function to `main.py`
A new reusable function that creates line detection visualizations:

```python
def visualize_detected_lines(image, words, left_margins, right_margins, output_path=None):
    """
    Visualize detected text lines with leftmost and rightmost points.
    
    Args:
        image: Input image (BGR format)
        words: Array of word bounding boxes [(xmin, ymin, xmax, ymax, conf), ...]
        left_margins: List of (x, y) tuples for leftmost points
        right_margins: List of (x, y) tuples for rightmost points
        output_path: Optional path to save visualization
    
    Returns:
        Visualization image (BGR format)
    """
```

Features:
- Draws all detected words in light gray
- Draws line connecting left and right margins in different colors per line
- Marks leftmost points with blue circles (white border)
- Marks rightmost points with yellow circles (white border)
- Labels each line with "L1", "L2", etc.
- Optionally saves to file

### 3. Updated `__init__.py`
Exported the new `visualize_detected_lines` function for easy import:
```python
from ocr_reflow import visualize_detected_lines
```

### 4. Created Example Scripts

#### `example_line_visualization.py`
A complete example showing how to use the visualization function:
- Loads an image
- Detects words using doctr
- Detects line margins
- Creates and saves visualization
- Prints detailed line information

#### `add_line_vis_to_notebook.py`
Script to add line detection visualization to the Jupyter notebook `reflow_layout_analysis.ipynb`:
- Adds a new section "Step 3a: Visualize Detected Text Lines"
- Includes code to detect and visualize text lines
- Shows detailed line information

## Usage Examples

### Command Line
```bash
# Run the verification script with visualization
pixi run python verify_out0_fix.py

# Run the example script
pixi run python example_line_visualization.py
```

### Python Script
```python
from ocr_reflow import margins, visualize_detected_lines
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import cv2
import numpy as np

# Load image
img = cv2.imread('your_image.png')
img_h, img_w = img.shape[:2]

# Detect words
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images(['your_image.png'])
result = model(docs)
words = result[0]["words"]

# Convert coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
words = words.astype(np.int32)

# Detect lines
left_margins, right_margins = margins(words)

# Create visualization
vis_img = visualize_detected_lines(
    img, words, left_margins, right_margins,
    output_path="visualization.png"
)
```

### Jupyter Notebook
```python
from ocr_reflow import margins, visualize_detected_lines
import matplotlib.pyplot as plt
import cv2

# ... (same detection code as above) ...

# Visualize
vis_img = visualize_detected_lines(img, words, left_margins, right_margins)

# Display in notebook
plt.figure(figsize=(15, 20))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Lines: {len(left_margins)}')
plt.axis('off')
plt.show()
```

## Output Files Created

1. `out0_line_detection_visualization.png` - Visualization for out0.png test case
2. `line_detection_example.png` - Example visualization from the demo script

## Visualization Legend

- **Blue circles**: Leftmost points of each detected line
- **Yellow circles**: Rightmost points of each detected line  
- **Colored lines**: Detected text baselines (different color per line)
- **Gray rectangles**: Individual detected words
- **Line labels**: "L1", "L2", etc. marking each line

## Benefits

1. **Visual Debugging**: Easy to see if line detection is working correctly
2. **Documentation**: Clear visual representation for documentation and presentations
3. **Quality Assurance**: Quickly identify issues with line detection
4. **Reusability**: Function can be used anywhere in the project
5. **Notebook Integration**: Available in Jupyter notebooks for interactive analysis

## Testing

All functionality has been tested with:
- `notebooks/out0.png` (12 lines) ✓
- `images/kf_16_par.png` (7 lines) ✓
- `images/out2.png` (7 lines) ✓

## Files Modified

1. `/home/sergey/code/python/segmentation/verify_out0_fix.py`
2. `/home/sergey/code/python/segmentation/src/ocr_reflow/main.py`
3. `/home/sergey/code/python/segmentation/src/ocr_reflow/__init__.py`

## Files Created

1. `/home/sergey/code/python/segmentation/example_line_visualization.py`
2. `/home/sergey/code/python/segmentation/add_line_vis_to_notebook.py`
3. `/home/sergey/code/python/segmentation/docs/line_visualization.md` (this file)
