# Layout-Based Reflow - Quick Reference Card

## Command Line Usage

```bash
# Basic usage (layout-based processing)
python src/ocr_reflow/main.py input.png --layout

# Old method (text-only)
python src/ocr_reflow/main.py input.png
```

## Python API

```python
from docs.main import process_document_with_layout

# Default settings
result = process_document_with_layout("input.png")

# Custom settings
result = process_document_with_layout(
    "input.png",
    zoom_factor=3.0,  # Default: 2.5
    new_page_width=2400  # Default: 2000
)

# Save output
import cv2

cv2.imwrite("output.png", result)
```

## What Gets Processed How

| Content Type | Processing Method |
|--------------|------------------|
| Plain text   | ✍️ Reflowed with word wrapping |
| Titles       | ✍️ Reflowed with word wrapping |
| Figures      | 🔍 Zoomed 2.5x and centered |
| Tables       | 🔍 Zoomed 2.5x and centered |
| Formulas     | 🔍 Zoomed 2.5x and centered |
| Other        | 🔍 Zoomed 2.5x and centered |

## Common Adjustments

### Make non-text elements larger
```python
result = process_document_with_layout("input.png", zoom_factor=4.0)
```

### Make non-text elements smaller
```python
result = process_document_with_layout("input.png", zoom_factor=1.5)
```

### Wider output page
```python
result = process_document_with_layout("input.png", new_page_width=2400)
```

### Narrower output page
```python
result = process_document_with_layout("input.png", new_page_width=1600)
```

### Both together
```python
result = process_document_with_layout(
    "input.png",
    zoom_factor=3.0,
    new_page_width=2200
)
```

## Batch Processing

```python
import glob
from docs.main import process_document_with_layout
import cv2

for img_path in glob.glob("input/*.png"):
    result = process_document_with_layout(img_path)
    output_path = img_path.replace("input", "output")
    cv2.imwrite(output_path, result)
```

## Output Files

- `output_reflowed.png` - Main output (full resolution)
- `output_reflowed_preview.png` - Preview with matplotlib

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Elements too small | Increase `zoom_factor` |
| Elements too large | Decrease `zoom_factor` |
| Page too narrow | Increase `new_page_width` |
| Page too wide | Decrease `new_page_width` |
| Out of memory | Process smaller images |
| Text not detected | Check image quality/contrast |
| IndexError in margins() | Fixed - ensure using latest code version |

## Performance

- **GPU**: 2-5 seconds per page
- **CPU**: 10-30 seconds per page

## Documentation

- Quick Start: `docs/QUICKSTART_LAYOUT.md`
- Complete Guide: `docs/LAYOUT_REFLOW_GUIDE.md`
- Implementation: `docs/LAYOUT_INTEGRATION_SUMMARY.md`
- Flow Diagrams: `docs/PROCESSING_FLOW_DIAGRAM.md`

## Content Types Detected

0. title
1. plain text
2. abandon
3. figure
4. figure_caption
5. table
6. table_caption
7. table_footnote
8. isolate_formula
9. formula_caption

## Box Pairing

- Figure + Caption → `figure_and_caption`
- Formula + Caption → `isolate_formula_and_caption`
- Table + Caption + Footnote → `table_and_caption`

---

**Need help?** Check the full documentation in `docs/`
