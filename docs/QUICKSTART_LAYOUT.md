# Quick Start: Layout-Based Document Reflow

## Installation

Make sure you have all required dependencies:

```bash
# Using pixi (recommended)
pixi install

# Or using pip
pip install -r requirements.txt
```

## Basic Usage

### 1. Process a Single Image

```bash
# Using the command line
python src/ocr_reflow/main.py input_image.png --layout
```

This will create:
- `output_reflowed.png` - The reflowed document
- `output_reflowed_preview.png` - A preview image

### 2. Using Python API

```python
from docs.main import process_document_with_layout
import cv2

# Process document
result = process_document_with_layout("input_image.png")

# Save result
cv2.imwrite("output.png", result)
```

## What Gets Reflowed vs. Preserved?

### Reflowed (Character-level reflow with word wrapping):
- **Plain text** - Main body text
- **Titles** - Headers and titles

### Preserved (Zoomed and placed as-is):
- **Figures** (with captions) - Diagrams, photos, illustrations
- **Tables** (with captions and footnotes) - Data tables
- **Formulas** (with captions) - Mathematical equations
- **Other elements** - Abandon blocks, etc.

## Customization Options

### Adjust Zoom Level

```python
# Larger zoom for better visibility
result = process_document_with_layout(
    "input.png",
    zoom_factor=3.5  # Default is 2.5
)
```

### Change Output Page Width

```python
# Wider page for desktop reading
result = process_document_with_layout(
    "input.png",
    new_page_width=2400  # Default is 2000
)
```

### Both Together

```python
result = process_document_with_layout(
    "input.png",
    zoom_factor=3.0,
    new_page_width=2200
)
```

## Common Scenarios

### Scenario 1: Book Pages (Mostly Text)

```bash
python src/ocr_reflow/main.py book_page.png --layout
```

Default settings work well for book pages.

### Scenario 2: Academic Papers (Text + Figures + Formulas)

```python
from docs.main import process_document_with_layout

result = process_document_with_layout(
    "paper.png",
    zoom_factor=3.0,  # Make formulas more readable
    new_page_width=2400  # Wider for better layout
)
```

### Scenario 3: Magazine Articles (Complex Layouts)

```python
result = process_document_with_layout(
    "magazine.png",
    zoom_factor=2.5,
    new_page_width=2000
)
```

### Scenario 4: Batch Processing Multiple Pages

```python
from docs.main import process_document_with_layout
import cv2
import os
import glob

# Process all PNG files in a directory
input_dir = "input_pages"
output_dir = "output_pages"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image
for img_path in glob.glob(f"{input_dir}/*.png"):
    print(f"Processing: {img_path}")

    # Process the document
    result = process_document_with_layout(img_path)

    # Create output filename
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, filename)

    # Save result
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}\n")

print("Batch processing complete!")
```

## Comparing Old vs. New Method

### Old Method (Text-Only Reflow)

```bash
python src/ocr_reflow/main.py input.png
```

**Pros:**
- Simple and fast
- Works well for pure text documents

**Cons:**
- Ignores figures, tables, formulas
- May break on complex layouts

### New Method (Layout-Based Reflow)

```bash
python src/ocr_reflow/main.py input.png --layout
```

**Pros:**
- Handles all content types
- Preserves figures, tables, formulas
- Better for complex documents
- Automatic pairing of captions with content

**Cons:**
- Slightly slower (2-5 seconds vs. 1-2 seconds per page)
- Requires more memory

## Troubleshooting

### Problem: Output looks wrong

**Check:**
1. Image quality - Low resolution images may not work well
2. Contrast - Make sure text is clearly visible
3. File format - Use PNG or high-quality JPEG

### Problem: Figures/formulas too small

**Solution:**
```python
result = process_document_with_layout(
    "input.png",
    zoom_factor=4.0  # Increase zoom
)
```

### Problem: Figures/formulas too large

**Solution:**
```python
result = process_document_with_layout(
    "input.png",
    zoom_factor=1.5  # Decrease zoom
)
```

### Problem: Page too narrow

**Solution:**
```python
result = process_document_with_layout(
    "input.png",
    new_page_width=2400  # Increase width
)
```

### Problem: Out of memory

**Solutions:**
1. Process smaller images
2. Close other applications
3. Use CPU mode (automatic fallback)

### Problem: Text not detected

**Check:**
1. Text contrast with background
2. Text size (very small text may not be detected)
3. Image resolution (300 DPI recommended)

## Performance Tips

1. **Use GPU if available** - Automatic, speeds up by 5-10x
2. **Batch process** - Process multiple files in one session to reuse models
3. **Optimize image size** - Resize very large images (>4000px) before processing

## Output File Locations

By default, files are saved in the current directory:

- `output_reflowed.png` - Main output
- `output_reflowed_preview.png` - Preview with matplotlib

To save to a specific location:

```python
result = process_document_with_layout("input.png")

# Save with custom name
cv2.imwrite("my_custom_output.png", result)
```

## Next Steps

- Read the full guide: `docs/LAYOUT_REFLOW_GUIDE.md`
- Check implementation details: `docs/LAYOUT_INTEGRATION_SUMMARY.md`
- Explore the code: `src/ocr_reflow/main.py`

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the error message
3. Try with a simpler/different image
4. Check the logs for detailed processing information

## Examples Directory

See the `examples/` directory for sample scripts demonstrating:
- Basic usage
- Custom configuration
- Batch processing
- Advanced options
