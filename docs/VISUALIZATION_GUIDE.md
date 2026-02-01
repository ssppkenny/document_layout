# Diagnostic Line Visualization - User Guide

## Overview

The `diagnose_segmentation.py` script now generates **comprehensive line detection visualizations** showing both before and after the merging process.

## Generated Visualization Files

When you run the diagnostic script, it creates **4 visualization files**:

### 1. `diagnostic_lines_before_merge.png` (63 KB)
**Shows:** Initial line detection BEFORE merging
- **Orange lines** - All 14 initially detected lines (including super/subscripts as separate lines)
- **Red rectangles** - All detected word bounding boxes
- **Blue circles** - Left margin points
- **Yellow circles** - Right margin points
- **L0, L1, L2...** - Line numbers labeled on each line

### 2. `diagnostic_lines_after_merge.png` (62 KB)
**Shows:** Final line detection AFTER merging
- **Green lines** (thicker) - The 7 final merged lines (CORRECT!)
- **Red rectangles** - All detected word bounding boxes
- **Blue circles** - Left margin points
- **Yellow circles** - Right margin points
- **L0, L1, L2...** - Final line numbers

### 3. `diagnostic_lines_comparison.png` (135 KB)
**Shows:** Side-by-side comparison of before and after
- **Left side:** Before merging (14 orange lines)
- **Right side:** After merging (7 green lines)
- **Headers:** Clear labels showing the difference
- **Great for:** Understanding what the merge algorithm does

### 4. `diagnostic_word_lines.png` (62 KB)
**Legacy file** - Same as `diagnostic_lines_after_merge.png` (for backward compatibility)

## How to Use

### Run the Diagnostic
```bash
# In pixi environment
pixi run python diagnose_segmentation.py

# The script will automatically generate all 4 visualization files
```

### What to Look For

#### In the "Before Merge" Image (Orange Lines)
- Look for **many short lines** with only 1-2 words
- These are typically **superscripts, subscripts, or punctuation**
- You should see lines very close together (< 20 pixels apart)

#### In the "After Merge" Image (Green Lines)
- Look for **fewer, more accurate lines**
- Each line should span the full width of the text
- Superscripts/subscripts should be integrated into main text lines
- Line count should match expected count (7 for kf_16_par.png)

#### In the Comparison Image
- **Left vs Right** shows the transformation
- See how 14 fragmented lines become 7 proper text lines
- Visual proof that the merge algorithm works correctly

## Example Output

For `images/kf_16_par.png`:

```
📊 Visualization BEFORE merging saved to: diagnostic_lines_before_merge.png
   - Orange lines: 14 detected lines
   - Red rectangles: detected words
   - Blue circles: left margins
   - Yellow circles: right margins

✅ Visualization AFTER merging saved to: diagnostic_lines_after_merge.png
   - Green lines: 7 merged lines (FINAL)
   - Red rectangles: detected words
   - Blue circles: left margins
   - Yellow circles: right margins

   Legacy file: diagnostic_word_lines.png (same as after_merge)

📊 Side-by-side comparison saved to: diagnostic_lines_comparison.png
   Shows both before (14 lines) and after (7 lines) merging
```

## Color Code Reference

| Color | Meaning |
|-------|---------|
| 🟠 **Orange lines** | Lines BEFORE merging (initial detection) |
| 🟢 **Green lines** | Lines AFTER merging (final, correct) |
| 🔴 **Red rectangles** | Detected word bounding boxes |
| 🔵 **Blue circles** | Left margin points |
| 🟡 **Yellow circles** | Right margin points |
| ⚪ **White text** | Line number labels (L0, L1, L2...) |

## Interpreting the Results

### Good Results (Expected)
```
Before: 14 lines → After: 7 lines
✅ Margin detection (merged) matches expected 7 lines!
```

**What you see:**
- Before image has many small orange lines for super/subscripts
- After image has 7 clean green lines spanning full text width
- Comparison shows clear reduction in line count

### Problem Indicators

If you see something different:

**Too many lines after merging (e.g., 9 or 10):**
- Some lines didn't merge properly
- Check the console output for merge operations
- May need to adjust `y_threshold` in merge_close_lines()

**Too few lines after merging (e.g., 5 or 6):**
- Over-aggressive merging
- Some distinct lines were incorrectly merged
- May need to reduce `y_threshold` or adjust merge criteria

## Technical Details

### Line Detection Process

1. **Word Detection** - doctr model finds all words
2. **Margin Detection** - `margins()` finds left/right endpoints of lines
3. **Visualization 1** - Save "before merge" image (orange lines)
4. **Line Merging** - `merge_close_lines()` merges super/subscripts
5. **Visualization 2** - Save "after merge" image (green lines)
6. **Comparison** - Create side-by-side image

### Merge Criteria (All in merge_close_lines function)

Lines are merged if they meet ANY of these criteria:

1. **Few Words:** One line has ≤3 words AND within 50px
2. **Very Close:** Lines are within 20px apart
3. **Height Difference:** One line has <70% the text height of the other

## Usage in Development

### Debugging Line Detection Issues

1. **Run diagnostic** on your test image
2. **Open comparison image** to see before/after
3. **Check console output** for merge decisions
4. **Adjust thresholds** if needed in the code

### Testing Changes

After modifying the merge algorithm:

```bash
# Run diagnostic to see the effect
pixi run python diagnose_segmentation.py

# Check all 4 visualization files
# Compare with previous results
```

### Creating Test Cases

You can use these visualizations to:
- Document test cases
- Show examples of good/bad segmentation
- Debug specific issues
- Verify fixes work correctly

## File Locations

All visualization files are saved in the project root:

```
segmentation/
├── diagnostic_lines_before_merge.png    (before: orange lines)
├── diagnostic_lines_after_merge.png     (after: green lines)
├── diagnostic_lines_comparison.png      (side-by-side)
└── diagnostic_word_lines.png            (legacy: same as after)
```

## Tips

### For Best Visualization Results

1. **High-resolution images** show details better
2. **Good contrast** helps see the lines clearly
3. **Zoom in** on the PNG files to see word boxes and line numbers
4. **Compare multiple images** to understand different document types

### When Sharing Results

- **Share the comparison image** to show the whole story
- Include the **console output** with line counts
- Mention the **image filename** being tested
- Note any **unusual characteristics** of the document

## Summary

The enhanced diagnostic script now provides:

✅ **Visual proof** that line merging works correctly  
✅ **Before/after comparison** to understand the algorithm  
✅ **Color-coded output** for easy interpretation  
✅ **Line numbers** to track specific lines  
✅ **Side-by-side view** for quick comparison  

This makes it much easier to verify that the line detection is working correctly and to debug any issues that arise with different document types.

---

**Next Steps:** Run `pixi run python diagnose_segmentation.py` and open the generated PNG files to see your line detection results!
