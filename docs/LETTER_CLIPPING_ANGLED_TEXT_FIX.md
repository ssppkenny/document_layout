# Letter Clipping Fix for Angled Text

## Problem
When processing angled text (like line 24 in dvurog_p021.png), the last words on the line were being clipped - letters were cut off from the top and bottom.

## Root Cause
The word bounding boxes returned by the doctr detection model were too tight and didn't include all the letter pixels, especially for angled text. When letters were extracted using these word boxes:
- Letters were touching the TOP edge of word boxes
- Letters were touching the BOTTOM edge of word boxes
- Letters were touching the LEFT and RIGHT edges of word boxes

This meant we were starting with incomplete letter regions from the very beginning.

## Investigation Process
1. **Initial observation**: Line 24 letters appeared clipped in reflowed output
2. **First hypothesis**: Letter bounding boxes from connected components were wrong
   - Added padding to letter boxes (2 pixels) - helped slightly but didn't solve the problem
3. **Second hypothesis**: Connected components analysis was fragmenting angled letters
   - Added filtering for small components - helped remove noise but didn't fix clipping
4. **Root cause discovered**: Word boxes from doctr were too small
   - Created diagnostic script that showed all letters were touching word box edges
   - This proved the word boxes themselves were the problem

## Solution
Increased padding for word bounding boxes from doctr:
- Changed from: `+2` / `-2` pixels
- Changed to: `-5` / `+5` pixels (expand in all directions)
- Added bounds clamping to stay within image dimensions

### Modified Files
- `src/ocr_reflow/main.py`: Two locations where word coordinates are converted
  1. Line ~347: `process_document()` function
  2. Line ~509: `process_document_with_layout()` function

### Code Changes
```python
# Before:
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

# After:
words[:, 0] = (words[:, 0] * img_w).astype(np.int32) - 5  # left: expand left
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) - 5  # top: expand up
words[:, 2] = (words[:, 2] * img_w).astype(np.int32) + 5  # right: expand right
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) + 5  # bottom: expand down
# Clamp to image bounds
words[:, 0] = np.maximum(words[:, 0], 0)
words[:, 1] = np.maximum(words[:, 1], 0)
words[:, 2] = np.minimum(words[:, 2], img_w)
words[:, 3] = np.minimum(words[:, 3], img_h)
```

## Testing
Created several diagnostic scripts to verify the fix:
- `diagnose_out13.py`: Analyzes word and letter detection
- `debug_word_letters.py`: Shows connected components in detail
- `check_letter_clipping.py`: Checks if letters touch word box edges
- `test_out13_reflow.py`: Tests reflow on a single line
- `final_comparison.py`: Creates visual comparison of original vs reflowed

## Results
- **Before fix**: 32 letters detected but split incorrectly into 2 lines (7 + 25 letters)
- **After fix**: 32 letters detected correctly as 1 line
- **Clipping**: Letters no longer touch word box edges, full letter regions are captured

## Visualization Files Generated
- `out13_word_letter_detection.png`: Shows detected words (blue) and letters (red)
- `out13_reflowed_test.png`: Reflowed version of test line
- `out13_comparison.png`: Side-by-side comparison
- `final_comparison_last_words.png`: Close-up of last words to verify no clipping

## Impact
This fix ensures that:
1. All letter pixels are captured, even for angled text
2. No clipping occurs at the top or bottom of letters
3. Baseline calculations are more accurate
4. Reflowed text maintains letter integrity
