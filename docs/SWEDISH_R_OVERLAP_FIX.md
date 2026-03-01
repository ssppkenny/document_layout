# Swedish Diacritics - Letter 'r' Overlap Fix

## Problem Identified
After user inspection of visualization, the issue was pinpointed:

**Both "Börja" and "inför" have the same problem:**
- Letter 'r' is too wide on the left side
- 'r' overlaps with 'ö' (the letter before it)

## Root Cause

The diacritic merging algorithm was merging ö dots with the WRONG letter:

**Example from "inför":**
- Raw components:
  - ö base at x=60
  - ö dot 1 at x=64
  - ö dot 2 at x=73
  - r at x=85

**What was happening:**
- The algorithm was merging ö dots with 'r' (x=85) instead of with 'ö' base (x=60)
- This created a merged component spanning x=71-103 (32px wide)
- Expected 'r': x=85-101 (16px wide)
- Result: 'r' extended 14 pixels too far left, overlapping with 'ö'

**Why it happened:**
Line 395 in main.py:
```python
is_horizontally_aligned = (horizontal_overlap > 0 or horizontal_gap < median_height * 0.5)
```

This allowed dots to merge with letters up to **50% of median_height** (~11px) away. The ö dot at x=73 and 'r' at x=85 have a 12px gap, which was just at the threshold, causing incorrect merging.

## Fix Applied

Changed horizontal alignment threshold from **50%** to **20%** of median_height:

```python
# Line 397-398
max_horizontal_gap = median_height * 0.2  # Much stricter (was 50%)
is_horizontally_aligned = (horizontal_overlap > 0 or horizontal_gap < max_horizontal_gap)
```

**Effect:**
- For typical text with median_height=22px:
  - Old threshold: 11px
  - New threshold: 4.4px
- Swedish ö dots must now be almost directly above 'o' (within ~4-5px)
- Prevents dots from merging with letters 10-15px away

## Previous Fixes (Still Active)

1. **Disabled horizontal letter merging** (line 301)
   - Prevents i+n+f from merging together

2. **Stricter vertical gap threshold** (line 380)
   - max_vertical_gap = median_height * 0.3 (was 100%)

3. **Merge with closest letter only** (lines 403-410)
   - Not all matching letters

## Test Results

### Before Final Fix:
- **inför**: 5 components, but 'r' overlapping 'ö'
- **Börja**: 7 components, but 'r' overlapping 'ö'

### After Final Fix:
- **inför**: 5 components ✓
- **Börja**: 5-7 components ~
- Overlap issue should be resolved

## Files Modified

**src/ocr_reflow/main.py**:
- Line 397-398: Changed horizontal gap threshold from 50% to 20%

## Verification

1. Run visualization:
   ```bash
   python visualize_both_swedish_words.py
   ```
   Check `/tmp/swedish_words_complete_visualization.png`
   
2. Run full reflow:
   ```bash
   python src/ocr_reflow/main.py images/gang_p023_lines1.png --layout
   ```
   Check `output_reflowed.png`

3. Look for:
   - ✓ No overlap between ö and r
   - ✓ Letters display complete without artifacts
   - ✓ Diacritics (dots) are part of their base letters

## Status

✅ Fix applied - horizontal alignment threshold reduced from 50% to 20%

**Please check the updated visualization and reflowed output to confirm the overlap is fixed.**

---
**Date:** 2026-02-28
**Issue:** Letter 'r' overlapping with 'ö' 
**Solution:** Stricter horizontal alignment for diacritic merging
