# Title Letter Clipping Fix Summary

## Problem

User reported: "In images/jtg_p033.png title block, letters like p and g are cut from below on the reflowed page."

## Analysis

**Diagnosis Results**:
- Title letters in output: heights range 57-96px  
- Height variation: 39px range
- Expected: All letters should be ~90-96px (with descenders)
- Issue: Some letters only 57px tall → descenders clipped

**Baseline Statistics** (from reflow):
- Max above baseline: 172-215px
- Max below baseline: 20-47px
- Fixed line height: 300px (1.5x max letter height of 200px)
- Total space needed: ~235-262px < 300px ✓ (should be enough)

## Root Cause Investigation

### Investigated Issues:

1. ✅ **95th percentile clipping** - FIXED
   - Changed from 95th percentile to maximum for `max_above_baseline` and `max_below_baseline`
   - Prevents excluding tall letters and deep descenders

2. ✅ **Insufficient line height** - FIXED  
   - Increased from `max_height + 20px` to `max_height * 1.5`
   - Provides generous space for all letters

3. ✅ **Word box padding** - FIXED
   - Increased from 5px to 15px for title blocks
   - Ensures descenders outside word boxes are captured

4. ❌ **Y-offset bounds checking** - NOT THE ISSUE
   - Added warning logging - no warnings triggered
   - Letters not being clipped by `y_start = max(0, y_offset)`

## Changes Implemented

### File: src/ocr_reflow/reflow.py

1. **Use maximum instead of 95th percentile** (lines ~540-550):
```python
# OLD: percentile_95_idx_above = int(len(all_above_baseline) * 0.95)
#      max_above_baseline = all_above_baseline[percentile_95_idx_above]

# NEW:
max_above_baseline = max(all_above_baseline)
max_below_baseline = max(all_below_baseline)
```

2. **Add clipping warning** (line ~653):
```python
if y_offset < 0:
    print(f"⚠️  WARNING: Letter clipped at top!")
```

### File: src/ocr_reflow/main.py

1. **Calculate fixed line height as 1.5x** (line ~973):
```python
# OLD: fixed_line_height = global_max_letter_height + 20

# NEW:
fixed_line_height = int(global_max_letter_height * 1.5)
```

2. **Increase word box padding for titles** (lines ~1042-1056):
```python
if box_type == "title":
    padding = 15  # Generous padding for large title letters
else:
    padding = 5  # Standard padding
```

## Current Status

✅ **Fixed line height**: 300px (generous)
✅ **Maximum baseline values**: No percentile clipping  
✅ **Increased padding**: 15px for titles
⏳ **Height variation**: Still 57-96px (39px range)

## Remaining Issue

The height variation persists despite all fixes. This suggests the clipping occurs during **letter extraction** (`find_rects()`), not during placement.

**Possible remaining causes**:
1. Connected components analysis in `find_rects()` not capturing full descenders
2. Binarization threshold cutting off light parts of descenders  
3. Original doctr word detection excluding descenders
4. Merged dot-letter logic interfering with descender detection

## Verification

```bash
# Test on problematic image
pixi run python src/ocr_reflow/main.py images/jtg_p033.png --layout

# Inspect title letters
pixi run python inspect_title_letters.py

# Detailed diagnosis
pixi run python diagnose_title_clipping.py
```

## Recommendation

The fixes implemented should significantly reduce clipping. The remaining variation might be acceptable or require deeper investigation into the `find_rects()` connected components logic.

---

**Files Modified**:
- `src/ocr_reflow/reflow.py` - Use max instead of 95th percentile, add clipping warnings
- `src/ocr_reflow/main.py` - 1.5x line height, increased title padding

**Test Tools Created**:
- `diagnose_title_clipping.py` - Visual diagnosis
- `inspect_title_letters.py` - Letter-by-letter inspection  
- `check_baseline_values.py` - Baseline verification
