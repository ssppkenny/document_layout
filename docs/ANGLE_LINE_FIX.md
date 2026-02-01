# Fix for Angled Line Detection - Line Baseline Preservation

## Problem
The line detection was not properly handling angled (non-horizontal) text lines. When text had a natural slope (e.g., leftmost letters higher than rightmost letters), the system was either:
1. Splitting the single line into multiple lines, or
2. Detecting it as one line but using an averaged Y-coordinate for both endpoints, creating an artificial horizontal line instead of following the actual text angle

This caused issues in the reflowed output where characters were not properly positioned relative to their baselines.

## Example
In `notebooks/out13.png`, there is a single line of text where:
- Leftmost word: x=73, y_center=14.5
- Rightmost word: x=692, y_center=28.0
- Angle: ~13.5 pixels of vertical difference across ~619 pixels horizontal distance

The old system would either split this into 2 lines or flatten it to a horizontal line.

## Solution

### 1. Updated Line Clustering Algorithm (`margins` function)
**Location:** `src/ocr_reflow/main.py`, lines 100-160

**Changes:**
- **Old approach:** Sorted words by center_y and grouped by Y-gaps
  - Problem: Failed for angled lines where consecutive words (by Y) are not consecutive horizontally
  
- **New approach:** Sort words by X-position and use vertical overlap to group
  - Words are sorted left-to-right by xmin
  - Two words belong to the same line if their Y-ranges overlap by at least 40% of the smaller height
  - Uses iterative expansion: once a word is added to a line, check if any remaining words overlap with ANY word in the current line
  - This naturally handles angled lines where far-apart words may not directly overlap but are connected through intermediate words

### 2. Preserved Actual Y-Coordinates
**Location:** `src/ocr_reflow/main.py`, lines 165-176

**Changes:**
- **Old code:**
  ```python
  # Calculate average center_y for this line
  avg_y = sum(w['center_y'] for w in line_words) / len(line_words)
  
  # Find leftmost word
  leftmost = min(line_words, key=lambda w: w['xmin'])
  left_margin.append((int(leftmost['xmin']), int(avg_y)))  # Uses avg_y!
  
  # Find rightmost word
  rightmost = max(line_words, key=lambda w: w['xmax'])
  right_margin.append((int(rightmost['xmax']), int(avg_y)))  # Uses avg_y!
  ```

- **New code:**
  ```python
  # Find leftmost word (minimum xmin)
  leftmost = min(line_words, key=lambda w: w['xmin'])
  # Use the actual center Y of the leftmost word (middle of its height)
  left_margin.append((int(leftmost['xmin']), int(leftmost['center_y'])))
  
  # Find rightmost word (maximum xmax)
  rightmost = max(line_words, key=lambda w: w['xmax'])
  # Use the actual center Y of the rightmost word (middle of its height)
  right_margin.append((int(rightmost['xmax']), int(rightmost['center_y'])))
  ```

**Result:** The line endpoints now use the actual center_y of the leftmost and rightmost words, preserving the natural angle of the text line.

## Verification

### Test Results

#### `notebooks/out13.png` (Angled line)
- **Before:** Detected as 2 lines or 1 horizontal line
- **After:** Correctly detected as 1 line with proper angle
  - Left margin: (73, 14)
  - Right margin: (692, 28)
  - Y difference: 14 pixels across 619 pixels (correct angle preserved)

#### `images/kf_16_par.png` (7 lines including subscripts)
- **Before:** Various detection issues
- **After:** Correctly detects 8 lines initially, merges subscript to get 7 final lines
  - All lines preserve their natural angles (-3.5 to +6.5 pixels Y difference)

#### `images/out2.png` (7 lines)
- **Before:** Various detection issues
- **After:** Correctly detects 7 lines with proper angles preserved

## Benefits

1. **Accurate baseline preservation:** Characters are now positioned on the reflowed page with correct vertical offsets relative to their baselines
2. **Handles angled text:** Works correctly even when text has significant slope
3. **Works with horizontal text:** Still correctly handles perfectly horizontal lines (Y difference ≈ 0)
4. **Maintains compatibility:** All existing test cases continue to pass

## Technical Details

### Overlap Calculation
For two words to be on the same line:
```python
overlap_top = max(word1['ymin'], word2['ymin'])
overlap_bottom = min(word1['ymax'], word2['ymax'])
overlap = max(0, overlap_bottom - overlap_top)
min_height = min(word1['height'], word2['height'])

# Requirement: overlap ≥ 40% of smaller word height
if overlap >= 0.4 * min_height:
    # Words are on the same line
```

This 40% threshold:
- Allows for moderate angle differences
- Filters out superscripts/subscripts (which typically don't overlap with main text)
- Handles slight vertical misalignments in OCR detection

### Integer Conversion
The final coordinates use `int()` which rounds down. For most cases, the 0.5 pixel precision loss is negligible, but for very precise applications, this could be changed to `round()` for nearest-integer rounding.

## Files Modified
- `src/ocr_reflow/main.py` - Updated `margins()` function

## Files Created for Diagnosis
- `diagnose_angle_issue.py` - Visualizes OLD vs NEW line detection methods
- `test_out13_clustering.py` - Debug script for clustering algorithm

## Date
February 1, 2026
