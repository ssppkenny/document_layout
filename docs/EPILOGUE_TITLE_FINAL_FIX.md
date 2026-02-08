# Epilogue Title Segmentation - RESOLVED

## Final Issue

**User Report**: "I can confirm that the Epilogue title block is not showing on the reflowed page, it is somehow not detected anymore"

## Root Cause Analysis

The Epilogue title was being **correctly detected and processed**, but was **skipped before letter extraction** due to a critical issue:

### The Problem Chain

1. **Word Merging**: Title blocks with multiple detected words (8 for "Epilogue") are merged into ONE box to prevent over-segmentation
   - Code: `words = np.array([[merged_xmin, merged_ymin, merged_xmax, merged_ymax]])`
   - Result: 8 words → 1 merged word box ✓

2. **Margins Function Failure**: The `margins()` function requires **at least 2 words** to detect lines
   - Code at line 423: `if len(entities) < 2: return [], []`
   - With 1 merged word: returns empty `left_margins` and `right_margins`

3. **Early Exit**: Empty margins triggered a skip
   - Code at line 1135: `if len(left_margins) == 0 or len(right_margins) == 0: continue`
   - **Result**: Title block skipped entirely, no letters extracted ❌

## Solution Implemented

Added **special handling for title blocks with single merged word**:

### Code Changes

**File**: `src/ocr_reflow/main.py` (lines ~1127-1200)

```python
# Special handling for title blocks with single merged word
if box_type == "title" and len(words) == 1:
    print(f"  [Title] Single merged word, processing directly")
    # Process the single word as one line
    wx1, wy1, wx2, wy2 = words[0][:4]
    line_letters = find_rects(box_img, [(wx1, wy1, wx2, wy2)])
    line_letters = sorted(line_letters, key=itemgetter(0))

    if len(line_letters) > 0:
        # ... baseline calculation ...
        letters = [
            Letter(l_xmin, l_ymin, l_xmax, l_ymax, baseline)
            for l_xmin, l_ymin, l_xmax, l_ymax in line_letters
        ]
        all_lines = [letters]
else:
    # Normal processing for multi-word blocks
    left_margins, right_margins = margins(words)
    # ... existing code ...
```

**Key Points**:
- Bypass `margins()` for single-word titles
- Extract letters directly from the merged word box
- Treat as one line with all letters
- Calculate baseline normally
- Continue to reflow processing

## Related Fixes Applied

### 1. Skew Detection Excludes Titles
**File**: `src/ocr_reflow/skew_detection.py` (line 287)

```python
# Filter to ONLY plain text boxes, EXCLUDE titles
text_only_boxes = [(geom, box_type) for geom, box_type in text_boxes
                   if box_type == "plain text"]  # Only plain text, not titles
```

**Reason**: Title fonts don't work well with skew detection; rotation corrupts letter shapes

### 2. Paragraph Detection Disabled for Titles
**File**: `src/ocr_reflow/reflow.py` (lines ~214-239)

```python
if is_title:
    # Title: no paragraph breaks, no indentation
    paragraph_starts = [0]
    paragraph_spacing = 0
    paragraph_indentations = {}
else:
    # Regular text: detect paragraphs
    paragraph_starts, avg_first_xmin = detect_paragraphs_and_spacing_from_lines(...)
```

**Reason**: Titles should be simple single-line text without paragraph analysis

### 3. Word Box Merging for Titles
**File**: `src/ocr_reflow/main.py` (lines ~1108-1117)

```python
if box_type == "title" and len(words) > 1:
    # Merge all word boxes into ONE
    merged_xmin = int(np.min(words[:, 0]))
    merged_ymin = int(np.min(words[:, 1]))
    merged_xmax = int(np.max(words[:, 2]))
    merged_ymax = int(np.max(words[:, 3]))
    words = np.array([[merged_xmin, merged_ymin, merged_xmax, merged_ymax]])
```

**Reason**: Prevents over-segmentation from doctr detecting decorative title text as multiple "words"

## Verification Results

**Terminal Output**:
```
[Title at y=2610] Detected 8 word(s)
  [Title] Merging 8 word boxes into one
  [Title] Merged box: (3, 12) → (366, 111)
  [Title] Single merged word, processing directly
  [Title] Extracted 8 letters from single word
  [Title] Extracted 1 lines with total 8 letters
  [Title] Reflowed page size: (300, 2000, 3), content_height: 249, placing at y=5940
  [Title] len(all_lines)=1
```

**Analysis**:
- ✅ 8 words detected (doctr output)
- ✅ Merged into 1 box (prevents over-segmentation)
- ✅ Single word processed directly (new special handling)
- ✅ 8 letters extracted (E-p-i-l-o-g-u-e)
- ✅ 1 line created
- ✅ Reflowed page generated (249px tall)
- ✅ Placed on final page at y=5940

## Complete Solution Summary

### Problems Solved
1. ✅ **Over-segmentation**: Disabled component splitting to prevent letters (p, i, g, u) being split vertically
2. ✅ **Skew corruption**: Excluded titles from skew detection to preserve letter structure  
3. ✅ **Word merging**: Merged multiple word boxes into one for titles
4. ✅ **Single word handling**: Added special path for single merged word (THIS FIX)
5. ✅ **Paragraph detection**: Disabled for titles (treat as simple single line)

### Expected Result
The "Epilogue" title should now appear correctly on the reflowed page with all 8 letters intact and properly positioned.

---

**Date**: February 8, 2026  
**Status**: ⚠️ **TITLE APPEARS BUT ROTATED**

All subsystems working correctly:
- Skew detection (excludes titles) ✓
- Word detection and merging (8→1) ✓
- Single word processing (new path) ✓
- Letter extraction (8 letters) ✓
- Reflow (249px content) ✓
- Page placement (y=5940) ✓
- **NEW ISSUE**: Title appears rotated on output page

### Current Investigation

**Latest Fix Applied**:
- Title blocks now extracted from `img_original` (before skew correction)
- Plain text still uses deskewed image

**Code**: `src/ocr_reflow/main.py` lines ~1043-1053
```python
if box_type == "title":
    box_img = img_original[ymin:ymax, xmin:xmax].copy()
    print(f"  [Title] Extracting from ORIGINAL image to avoid rotation")
else:
    box_img = img[ymin:ymax, xmin:xmax].copy()
```

**Note**: User reports page is NOT skewed, so rotation might be from:
1. Layout box coordinates mismatch between original and deskewed images
2. Individual letter images being rotated during extraction
3. Baseline angle calculation causing visual rotation

Need to verify exact nature of "rotation" - whole title block vs individual letters.
