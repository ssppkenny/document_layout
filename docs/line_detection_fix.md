# Line Detection Fix for out0.png

## Problem
The line detection algorithm was incorrectly detecting 9 lines instead of the expected 12 lines in `notebooks/out0.png`.

## Root Cause
The gap threshold used to separate lines was too large (50% of median word height). This caused the algorithm to merge separate lines that were close together.

## Solution
Changed the gap threshold from **0.5** (50%) to **0.42** (42%) of the median word height.

### Files Modified
1. `src/ocr_reflow/main.py` - Line 156
2. `diagnose_segmentation.py` - Line 77

### Code Change
```python
# Before:
gap_threshold = median_height * 0.5

# After:
gap_threshold = median_height * 0.42
```

## Verification
The fix was tested on three different test cases:

| Test Image | Expected Lines | Detected Lines | Status |
|------------|---------------|----------------|--------|
| notebooks/out0.png | 12 | 12 | ✓ PASS |
| images/kf_16_par.png | 7 | 7 | ✓ PASS |
| images/out2.png | 7 | 7 | ✓ PASS |

## Technical Details

### How Line Detection Works
1. Calculate median word height from all detected words
2. Filter out small words (subscripts/superscripts) that are < 70% of median height
3. Sort remaining words by center Y coordinate
4. Group words into lines based on Y-gaps
5. If gap between consecutive words > threshold, start a new line
6. For each line, find leftmost and rightmost words to define margins

### Why 0.42 Works
The threshold of 0.42 was determined empirically by testing different values:
- 0.40 → 13 lines (too many)
- 0.41 → 12 lines ✓
- 0.42 → 12 lines ✓ (with margin)
- 0.45 → 11 lines (too few)

A value of 0.42 provides a good balance that correctly separates lines while handling:
- Normal line spacing
- Varying font sizes
- Subscripts and superscripts
- Short lines

## Testing
Run the test suite to verify:
```bash
pixi run python test_line_detection.py
```

Or test individual images:
```bash
pixi run python diagnose_segmentation.py notebooks/out0.png
```

## Output Files
The diagnostic script creates several visualization files:
- `diagnostic_lines_before_merge.png` - Shows lines before merging
- `diagnostic_lines_after_merge.png` - Shows final merged lines
- `diagnostic_lines_comparison.png` - Side-by-side comparison
- `diagnostic_word_lines.png` - Legacy visualization

## Notes
- The threshold value may need fine-tuning for documents with significantly different formatting
- The current value works well for typical book paragraphs with standard line spacing
- Subscripts and superscripts are still filtered out (< 70% of median height) to avoid creating false lines
