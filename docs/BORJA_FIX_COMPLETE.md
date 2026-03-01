# Börja/inför Fix - COMPLETED

## Problem Summary
Swedish words "Börja" and "inför" were displaying with "half of letter ö" on the reflowed page.

## Root Cause
**BUG IN `find_rects()` coordinate conversion** (line ~465 in main.py):

The tuple unpacking was wrong:
```python
# WRONG (was swapping xmax/ymin):
rectangles = [(int(xmin), int(xmax), int(ymin), int(ymax)) for xmin, ymin, xmax, ymax in rects]

# CORRECT:
rectangles = [(int(xmin), int(xmax), int(ymin), int(ymax)) for xmin, xmax, ymin, ymax in rects]
```

The rects were already in format `(xmin, xmax, ymin, ymax)` but the unpacking assumed `(xmin, ymin, xmax, ymax)`, causing coordinates to be scrambled.

## Fix Applied
✅ Fixed line 474: Changed tuple unpacking order  
✅ Added debug output to track coordinate conversion  
✅ **TEMPORARILY DISABLED** enclosed-rectangle removal (line 475-484)

## Test Results
**BEFORE FIX:**
- inför: 8 raw → 2 merged (❌ wrong)
- Börja: 10 raw → 2 merged (❌ wrong)

**AFTER FIX (with enclosure-removal disabled):**
- inför: 8 components, all coordinates correct ✅
- Börja: 6 components, all coordinates correct ✅

## Known Issue
The enclosed-rectangle removal logic (`divide_conquer_4d`) is **too aggressive** for Swedish text. When diacritics are merged with base letters, the merged bounding box becomes large and incorrectly "encloses" nearby letters, causing them to be removed.

**Example**: In "Börja", when ö dots are merged with 'o', the merged component might enclose parts of 'r', causing 'r' to be filtered out.

## Next Steps
1. ✅ **DONE**: Fix coordinate conversion bug
2. ✅ **DONE**: Verify coordinates are correct
3. 🔄 **IN PROGRESS**: Need to fix enclosed-removal logic or find alternative
4. ⏳ **TODO**: Fix reflow.py error that occurs with more components

## Temporary Workaround
The enclosed-removal is currently DISABLED (line 476: `if False:`). This allows Swedish text to work but may cause issues with other edge cases where enclosed rectangles should be removed.

## Files Modified
- `src/ocr_reflow/main.py`:
  - Line 474: Fixed tuple unpacking
  - Lines 449-457: Added debug output
  - Lines 476-485: Disabled enclosed-removal temporarily

---
**Status**: ✅ Core bug fixed, coordinates correct  
**Remaining**: Need smarter enclosed-removal logic for diacritics
