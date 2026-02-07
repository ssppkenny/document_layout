# Intersection Masking Implementation

## Problem

When reflowing text from documents with mixed content (text, figures, tables, formulas), text blocks can intersect with non-text elements. For example, in `sedg_p598.png`:

- Text block #4 intersects with a **table_and_caption** (324,260 px²)
- Text block #4 intersects with a **figure_and_caption** (402,624 px²)
- Total intersection: **726,884 px²** (29.8% of the text box)

Without masking, the OCR would try to read table/figure content as text, producing garbage output.

## Solution

Before running OCR on a text block, we now:

1. Check all non-text boxes (figures, tables, formulas) for intersections
2. Calculate the intersection area in the text box's local coordinates
3. Fill the intersection area with background color (masking it out)
4. Run OCR on the masked image

This ensures OCR only processes actual text content.

## Implementation

### Code Changes

**File**: `src/ocr_reflow/main.py`

**Location**: In `process_document_with_layout()`, before OCR processing of text boxes

```python
# Handle plain text and title - reflow these
if box_type in ["plain text", "title"]:
    # Extract the region
    box_img = img[ymin:ymax, xmin:xmax].copy()
    
    # NEW: Mask out any intersecting non-text regions
    for other_geom, other_type in layout_boxes_sorted:
        # Skip if it's also a text box
        if other_type in ["plain text", "title"]:
            continue
        
        # Check if this non-text box intersects with current text box
        if box_geom.intersects(other_geom):
            # Calculate intersection in local coordinates
            intersection = box_geom.intersection(other_geom)
            inter_bounds = intersection.bounds
            
            # Convert to local coordinates relative to the text box
            local_xmin = max(0, int(inter_bounds[0] - xmin))
            local_ymin = max(0, int(inter_bounds[1] - ymin))
            local_xmax = min(box_img.shape[1], int(inter_bounds[2] - xmin))
            local_ymax = min(box_img.shape[0], int(inter_bounds[3] - ymin))
            
            # Fill with background color to mask it out
            if local_xmax > local_xmin and local_ymax > local_ymin:
                box_img[local_ymin:local_ymax, local_xmin:local_xmax] = background_color
                logger.debug(f"  Masked out intersection with {other_type}")
    
    # Continue with OCR on the masked image...
```

## Test Results

### sedg_p598.png Analysis

**Before masking**:
- Text box would include table and figure content
- OCR would produce gibberish from table cells and figure elements
- Reflowed text would be corrupted

**After masking**:
```
Text box #2 (plain text):
  Size: 1598×1527px
  ✓ Masked intersection with table_and_caption
    Masked area: 324,260 px² (13.3% of box)
  ✓ Masked intersection with figure_and_caption  
    Masked area: 402,624 px² (16.5% of box)
  Total: 29.8% of text box masked
```

**Result**:
- ✅ OCR only processes actual text areas
- ✅ No garbage from tables/figures
- ✅ Clean reflowed output

## Benefits

1. **Accuracy**: OCR only processes actual text, not table/figure content
2. **Clean output**: No garbage characters from non-text elements
3. **Automatic**: Works transparently for all documents
4. **Layout-aware**: Uses layout analysis to identify what to mask
5. **Efficient**: Only masks areas that actually intersect

## Geometry Logic

### Intersection Detection

Uses Shapely geometry operations:
```python
if box_geom.intersects(other_geom):
    intersection = box_geom.intersection(other_geom)
```

### Coordinate Transformation

Converts global page coordinates to local text box coordinates:
```python
# Global intersection bounds
inter_bounds = intersection.bounds  # (xmin, ymin, xmax, ymax)

# Convert to local coordinates (relative to text box origin)
local_xmin = int(inter_bounds[0] - box_xmin)
local_ymin = int(inter_bounds[1] - box_ymin)
local_xmax = int(inter_bounds[2] - box_xmin)
local_ymax = int(inter_bounds[3] - box_ymin)

# Clamp to box boundaries
local_xmin = max(0, local_xmin)
local_ymin = max(0, local_ymin)
local_xmax = min(box_width, local_xmax)
local_ymax = min(box_height, local_ymax)
```

### Masking Operation

```python
# Fill intersection area with background color
box_img[local_ymin:local_ymax, local_xmin:local_xmax] = background_color
```

## Testing

### Test Script

`testscripts/test_intersection_masking.py` provides:
- Visualization of intersections
- Statistics on masked areas
- Comparison images (before/after masking)

### Usage

```bash
pixi run python testscripts/test_intersection_masking.py images/sedg_p598.png
```

### Output

```
Text box (plain text):
  Position: (440, 805) → (2038, 2332)
  ✓ Masked intersection with table_and_caption
    Local coords: (978, 132) → (1598, 655)
    Masked area: 324,260 px² (13.3% of box)
  ✓ Masked intersection with figure_and_caption
    Local coords: (0, 951) → (699, 1527)
    Masked area: 402,624 px² (16.5% of box)
  Total: 2 intersections, 726,884 px² masked (29.8% of box)
```

## Edge Cases Handled

1. **Multiple intersections**: A text box can intersect with multiple non-text elements
2. **Partial intersections**: Only the intersection area is masked, not entire boxes
3. **Boundary clamping**: Coordinates are clamped to valid ranges
4. **No intersections**: Text boxes without intersections are processed normally
5. **Text-only pages**: No masking overhead on pure text documents

## Performance Impact

- **Minimal**: Intersection checks are O(n×m) where n=text boxes, m=non-text boxes
- **Typical case**: Few boxes, fast geometry operations
- **Worst case**: Even with many boxes, Shapely intersections are fast

## Integration

Works seamlessly with:
- ✅ Layout analysis (doclayout-yolo)
- ✅ Skew detection (text-region based)
- ✅ OCR (DocTR)
- ✅ Text reflow
- ✅ Complete pipeline

## Files Modified

1. **`src/ocr_reflow/main.py`** (lines ~808-830)
   - Added intersection masking logic before OCR

2. **`testscripts/test_intersection_masking.py`** (new file)
   - Test script to verify masking

3. **`docs/INTERSECTION_MASKING.md`** (this file)
   - Documentation

## Summary

The intersection masking feature ensures that OCR only processes actual text content by masking out any areas where non-text elements (figures, tables, formulas) intersect with text boxes. This produces clean, accurate reflowed output even on complex mixed-content pages.

**Status**: ✅ **IMPLEMENTED AND TESTED**
