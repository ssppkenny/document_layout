# Binarization Implementation Summary

## Current Implementation (Updated)

The OCR reflow pipeline now uses **normalized + binarized images only**, without adding the binary result back to the original image.

## Processing Steps

When using the `--bin` flag, the following steps are performed:

### Step 1: Normalization
```python
img_normalized = normalize_image(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
```
- Brings pixel intensity values to full [0, 255] range
- Improves contrast and binarization quality

### Step 2: Binarization
```python
binary_img = binarize_document(img_normalized, method='otsu')
```
- Applies Otsu's automatic thresholding (default)
- Alternative methods: Sauvola, Niblack
- Creates clean black/white binary image

### Step 3: Use Binary Image
```python
binary_img_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
# Save and use binary_img_bgr for OCR and reflow
```
- Converts to 3-channel BGR for compatibility
- The binary image is used directly (not added to original)

## Usage

```bash
# Use Otsu binarization (default)
python src/ocr_reflow/main.py images/yourimage.png --layout --bin

# Use Sauvola with custom window size
python src/ocr_reflow/main.py images/yourimage.png --layout --bin --bin-method sauvola --bin-window 25

# Use Niblack
python src/ocr_reflow/main.py images/yourimage.png --layout --bin --bin-method niblack
```

## Binarization Methods

### Otsu (Default)
- **Type**: Global automatic thresholding
- **Pros**: No parameters needed, fast, works well for most documents
- **Best for**: Documents with good contrast
- **Output**: White text on black background (inverted from original)

### Sauvola
- **Type**: Adaptive local thresholding  
- **Parameters**: window_size (default: auto-calculated), k (default: 0.5)
- **Pros**: Handles varying lighting and contrast
- **Best for**: Historical documents, poor quality scans
- **Output**: Black text on white background

### Niblack
- **Type**: Adaptive local thresholding
- **Parameters**: window_size (default: auto-calculated), k (default: -0.2)
- **Pros**: Good for degraded documents
- **Note**: May introduce background noise
- **Output**: Black text on white background

## Pipeline Flow

```
Original Image (grayscale/color)
         ↓
   Normalization
   (cv2.normalize → [0, 255])
         ↓
   Binarization
   (Otsu/Sauvola/Niblack)
         ↓
   Binary Image (pure black/white)
         ↓
   OCR & Reflow Processing
```

## Key Features

✓ **Clean Binary Output**: Pure black/white images for optimal OCR
✓ **Automatic Normalization**: All images normalized to [0, 255] range
✓ **Multiple Methods**: Choose between Otsu, Sauvola, and Niblack
✓ **Seamless Integration**: Works with existing OCR and reflow pipeline
✓ **No Manual Tuning**: Otsu method requires no parameters

## Implementation Details

### File: `src/ocr_reflow/binarization.py`
- `normalize_image()` - Image normalization using cv2.normalize()
- `otsu_binarization()` - Global Otsu thresholding
- `sauvola_binarization()` - Adaptive Sauvola method
- `niblack_binarization()` - Adaptive Niblack method
- `binarize_document()` - Main function dispatching to specific methods

### File: `src/ocr_reflow/main.py`
- Integrated into `process_document_with_layout()`
- Controlled by `--bin` flag and related parameters
- Creates temporary file with binary image
- Automatic cleanup after processing

## Testing

Test scripts provided:
- `test_binarized_only.py` - Verify binary-only approach
- `test_normalization.py` - Test normalization pipeline
- `demo_normalization.py` - Simple normalization demo

Run tests:
```bash
python3 test_binarized_only.py
python3 test_normalization.py
python3 demo_normalization.py
```

## Example Output Characteristics

### With Otsu (Default):
- White text on black background
- Mean pixel value < 128 (mostly black background)
- Only 2 unique values: 0 and 255

### With Sauvola/Niblack:
- Black text on white background  
- Mean pixel value > 128 (mostly white background)
- Only 2 unique values: 0 and 255

## Benefits

1. **Improved OCR Accuracy**: Binary images provide clearer text boundaries
2. **Faster Processing**: Binary images are simpler to process
3. **Consistent Output**: Normalization ensures consistent input quality
4. **Reduced File Size**: Binary images compress well
5. **Better Text Detection**: Clear foreground/background separation

## When to Use

**Use binarization (`--bin`) when:**
- Document has good to moderate quality
- Text needs to be clearly separated from background
- OCR accuracy is critical
- Processing speed is important

**Skip binarization (no `--bin`) when:**
- Document has colored text or diagrams
- Preserving grayscale information is important
- Images already have excellent quality

## Technical Notes

- Normalization uses `cv2.NORM_MINMAX` by default
- Otsu automatically determines optimal threshold
- Sauvola and Niblack use local adaptive thresholds
- Window size auto-calculated based on image width (~2%)
- Binary output converted to BGR for pipeline compatibility

## Summary

The binarization feature provides a robust preprocessing pipeline:
- **Normalize** → standardize pixel range
- **Binarize** → create clean binary image
- **Use** → binary image goes directly to OCR

This approach ensures optimal OCR accuracy while maintaining simplicity and speed.
