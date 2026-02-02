# Layout-Based Document Reflow

## Overview

This module provides advanced document reflow capabilities with layout analysis. It can:

1. **Analyze document layout** to identify different content types:
   - Plain text
   - Titles
   - Figures (with captions)
   - Tables (with captions and footnotes)
   - Formulas (with captions)
   - Other content types

2. **Reflow text** while preserving:
   - Paragraph structure
   - Baseline alignment
   - Word spacing
   - Line breaks

3. **Preserve non-text elements** by:
   - Zooming figures, tables, and formulas
   - Centering them on the new page
   - Maintaining aspect ratios

## Usage

### Command Line

```bash
# Basic text-only reflow (original method)
python src/ocr_reflow/main.py input_image.png

# Layout-based reflow (recommended)
python src/ocr_reflow/main.py input_image.png --layout
```

### Python API

```python
from docs.main import process_document_with_layout

# Process a document with layout analysis
result = process_document_with_layout(
    filename="input_image.png",
    zoom_factor=2.5,  # Zoom factor for non-text elements
    new_page_width=2000  # Width of the output page
)

# Save the result
import cv2

cv2.imwrite("output_reflowed.png", result)
```

## How It Works

### 1. Layout Analysis

The system uses a YOLO-based layout detection model to identify different content types on the page:

```python
from ocr_reflow.layout import layout as analyze_layout

# Returns list of (box_geometry, box_type) tuples
layout_boxes = analyze_layout("input_image.png")
```

### 2. Content Type Processing

**Text Content (plain text, title):**
- Extracted character by character
- Baseline alignment detected
- Reflowed with word wrapping
- Paragraph indentation preserved

**Non-Text Content (figures, tables, formulas):**
- Extracted as complete boxes
- Zoomed by `zoom_factor`
- Centered on the new page
- Resized if too wide

### 3. Box Pairing

Related content is automatically paired:
- Figures with their captions → `"figure_and_caption"`
- Formulas with their captions → `"isolate_formula_and_caption"`
- Tables with captions and footnotes → `"table_and_caption"`

### 4. Layout Order

Boxes are sorted by:
1. Y-coordinate (top to bottom)
2. X-coordinate (left to right)

This ensures natural reading order on the output page.

## Configuration Parameters

### `process_document_with_layout()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | required | Path to input image |
| `zoom_factor` | float | 2.5 | Scaling factor for non-text elements |
| `new_page_width` | int | 2000 | Width of output page in pixels |

### Text Reflow Parameters

The `create_page_with_word_wrapping()` function (called internally) supports:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `left_margin` | int | 50 | Left margin in pixels |
| `right_margin` | int | 50 | Right margin in pixels |
| `top_margin` | int | 50 | Top margin in pixels |
| `bottom_margin` | int | 50 | Bottom margin in pixels |
| `line_spacing` | int | 20 | Extra spacing between lines |
| `paragraph_spacing_factor` | float | 2.0 | Multiplier for paragraph spacing |

## Box Types

The layout analyzer detects these types:

| ID | Type | Processing |
|----|------|-----------|
| 0 | `title` | Reflowed as text |
| 1 | `plain text` | Reflowed as text |
| 2 | `abandon` | Zoomed and placed |
| 3 | `figure` | Paired with caption, zoomed |
| 4 | `figure_caption` | Paired with figure |
| 5 | `table` | Paired with caption/footnote, zoomed |
| 6 | `table_caption` | Paired with table |
| 7 | `table_footnote` | Paired with table |
| 8 | `isolate_formula` | Paired with caption, zoomed |
| 9 | `formula_caption` | Paired with formula |

## Examples

### Example 1: Academic Paper

```python
from docs.main import process_document_with_layout

# Process an academic paper with figures and formulas
result = process_document_with_layout(
    filename="paper_page.png",
    zoom_factor=3.0,  # Larger zoom for better formula visibility
    new_page_width=2400  # Wider page for academic content
)

import cv2

cv2.imwrite("paper_reflowed.png", result)
```

### Example 2: Book Page

```python
from docs.main import process_document_with_layout

# Process a book page with mostly text
result = process_document_with_layout(
    filename="book_page.png",
    zoom_factor=2.0,  # Smaller zoom for compact output
    new_page_width=1600  # Standard book page width
)

import cv2

cv2.imwrite("book_reflowed.png", result)
```

### Example 3: Batch Processing

```python
from docs.main import process_document_with_layout
import cv2
import glob

# Process all images in a directory
for image_path in glob.glob("input_pages/*.png"):
    print(f"Processing {image_path}...")
    result = process_document_with_layout(image_path)

    # Create output filename
    output_path = image_path.replace("input_pages", "output_pages")
    cv2.imwrite(output_path, result)
    print(f"Saved to {output_path}")
```

## Troubleshooting

### Issue: Text not detected

**Solution:** Check image quality and contrast. The OCR model works best with:
- High resolution (300+ DPI)
- Good contrast between text and background
- Minimal noise or artifacts

### Issue: Layout boxes overlap incorrectly

**Solution:** Adjust the confidence threshold in `layout.py`:

```python
det_res = model.predict(
    image_path,
    imgsz=1024,
    conf=0.3,  # Increase from 0.2 for stricter detection
    device=device
)
```

### Issue: Non-text elements too large/small

**Solution:** Adjust the `zoom_factor` parameter:

```python
# For smaller elements
result = process_document_with_layout(filename, zoom_factor=3.5)

# For larger elements
result = process_document_with_layout(filename, zoom_factor=1.5)
```

### Issue: Page too narrow/wide

**Solution:** Adjust the `new_page_width` parameter:

```python
# For mobile-friendly output
result = process_document_with_layout(filename, new_page_width=1200)

# For desktop reading
result = process_document_with_layout(filename, new_page_width=2400)
```

## Performance Considerations

- **Layout detection:** ~1-2 seconds per page (GPU)
- **Text detection:** ~0.5-1 second per text box (GPU)
- **Character extraction:** ~0.1-0.2 seconds per line
- **Image composition:** ~0.1 second

**Total:** Approximately 2-5 seconds per page on GPU, 10-30 seconds on CPU.

### Optimization Tips

1. **Use GPU if available** - The YOLO layout model is significantly faster on GPU
2. **Batch processing** - Process multiple pages in sequence to reuse loaded models
3. **Reduce image size** - Scale down very large images (>3000px) before processing
4. **Skip unnecessary content** - Filter out box types you don't need

## Advanced Usage

### Custom Layout Processing

```python
from ocr_reflow.layout import layout as analyze_layout
from docs.main import process_document_with_layout

# Get layout boxes
layout_boxes = analyze_layout("input.png")

# Filter to only process text and figures
filtered_boxes = [
    (box_geom, box_type)
    for box_geom, box_type in layout_boxes
    if box_type in ["plain text", "title", "figure_and_caption"]
]

# Custom processing...
```

### Adjusting Text Reflow Settings

```python
from ocr_reflow.reflow import create_page_with_word_wrapping
from docs.main import find_rects, margins, Letter
import cv2
import numpy as np

# ... extract lines and letters as in process_document_with_layout ...

# Create custom reflowed page
result = create_page_with_word_wrapping(
    all_lines,
    box_img,
    zoom_factor=2.5,
    new_page_width=2000,
    left_margin=100,  # Wider margins
    right_margin=100,
    top_margin=75,
    bottom_margin=75,
    line_spacing=30,  # More line spacing
    paragraph_spacing_factor=3.0,  # Larger paragraph breaks
    background_color=(255, 255, 240)  # Cream background
)
```

## Contributing

When modifying the layout processing pipeline:

1. Test with various document types (books, papers, magazines)
2. Verify box pairing works correctly (figures with captions, etc.)
3. Check that sorting maintains reading order
4. Ensure non-text elements are properly centered
5. Validate that page expansion works for long documents

## License

See LICENSE file in the project root.
