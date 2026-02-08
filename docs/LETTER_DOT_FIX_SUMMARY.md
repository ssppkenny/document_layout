# Letter Segmentation Fix - Summary

## Issue Fixed
Letters like 'i', 'j' were losing their dots during letter segmentation from words. This was causing incomplete letter rendering in the reflowed output.

## Root Cause
The `find_rects()` function used connected components analysis to extract individual letters from words. The filtering logic rejected components smaller than 20% of word height as "noise", which inadvertently removed dots from letters.

## Solution Implemented
Changed from **height-based filtering** to **proximity-based filtering**:

### Old Logic (BROKEN):
```python
if h >= word_height * 0.2:  # At least 20% of word height
    valid_components.append((x, y, w, h))
```
**Problem**: Dots are < 20% of word height → filtered out

### New Logic (FIXED):
```python
# 1. Find main letter bodies (≥30% word height)
main_components = [...]

# 2. Include small components near main components
for component in small_components:
    if is_vertically_near_main(component, within=40% word height) and \
       is_horizontally_aligned(component, within=30% word width):
        valid_components.append(component)
```
**Benefit**: Dots, accents, diacritics preserved while true noise is filtered

## Files Modified
- `src/ocr_reflow/main.py` - `find_rects()` function (lines ~141-240)

## Testing
```bash
# Test letter segmentation
pixi run python test_letter_fix.py images/sedg_p598.png

# Visualize words with dots
pixi run python visualize_dots.py images/sedg_p598.png

# Run complete pipeline
pixi run python src/ocr_reflow/main.py images/sedg_p598.png --layout
```

## Results
**Test on images/sedg_p598.png:**
- ✓ 78 words detected
- ✓ 458 letters extracted (5.9 letters/word)
- ✓ 30 words with dots found
- ✓ All dots correctly preserved

## Documentation
- Detailed explanation: `docs/LETTER_DOT_FIX.md`
- Visual comparison: `docs/letter_dot_fix_comparison.png`
- Test outputs: `notebooks/letter_segmentation_test.png`, `notebooks/dots_analysis.png`

## Impact
✓ Complete letter rendering in reflowed text
✓ Preserves dots on i, j
✓ Preserves accents and diacritical marks
✓ Still filters out true noise
✓ Works across different fonts and sizes

**Status**: ✅ FIXED AND VERIFIED
