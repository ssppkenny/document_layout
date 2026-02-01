# Letter Clipping Fix for Angled Baselines - FINAL

**Date:** February 1, 2026  
**Issue:** Letters clipped at top/bottom on angled lines with paragraph indentation  
**Status:** ✅ FIXED

## Problem Summary

Letters at the end of angled lines (especially line 24 in dvurog_p021.png / out13.png) were being clipped, showing only 2-5px height instead of full height. The issue affected rightmost words on lines with ascending baselines.

## Root Cause

The baseline positioning calculation for angled lines didn't account for **paragraph indentation**. When a line starts with an indent:

1. Initial baseline calculation used `available_width` (1900px)
2. But the actual line started at `left_margin + paragraph_indent` (e.g., 50 + 192 = 242px)
3. So the first letter's baseline was already elevated by `slope × indent`
4. The baseline rise calculation was wrong, causing rightmost letters to be positioned too high
5. These letters went above the allocated line space and got clipped

### The Math

For an ascending line (slope > 0):
- Rightmost letter baseline: `baseline_y_at_start + slope × line_span`
- Rightmost letter top: `baseline_y_at_start + slope × line_span - max_above_baseline`

Without indent:
- line_span = letter_widths_sum
- Works correctly

With indent of 192px:
- Actual span = indent + letter_widths_sum  
- But calculation only used letter_widths_sum
- Missing: `slope × indent` ≈ 0.0223 × 192 ≈ 4.3px

While 4px doesn't sound like much, for small letters (dots, periods) with only 2-3px above baseline, this causes complete clipping!

## Solution

### Step 1: Calculate baseline using actual line content width

Changed from using `available_width` to summing actual letter widths:

```python
line_width = sum(item['scaled_width'] + item.get('space_before', 0) 
                 for item in line['letters'])
baseline_rise = scaled_baseline_slope * line_width
```

### Step 2: Account for paragraph indentation

After applying indent, recalculate baseline for ascending lines:

```python
if scaled_baseline_slope > 0 and paragraph_indent > 0:
    total_x_span = paragraph_indent + line_width
    baseline_rise = scaled_baseline_slope * total_x_span
    baseline_y_at_start = current_y + max_above_baseline_for_line - int(baseline_rise)
```

## Files Modified

**src/ocr_reflow/reflow.py:**

1. **Lines 633-645:** Initial baseline calculation uses actual line content width
2. **Lines 670-682:** Apply paragraph indent before letter placement  
3. **Lines 684-693:** Recalculate baseline accounting for indent

## Technical Details

### Baseline Positioning Formula

For ascending baseline (slope > 0):

```
baseline_y_at_start = current_y + max_above - baseline_rise
```

Where:
- `current_y` = top of allocated line space
- `max_above` = maximum space any letter needs above baseline
- `baseline_rise` = slope × total_horizontal_span

### Coordinate System

- Letter placement: `baseline_y_here = baseline_y_at_start + slope × (current_x - left_margin)`
- `current_x` starts at `left_margin` then gets indent added
- So first letter is at X = indent (relative to left_margin)
- Must include this in baseline_rise calculation

### Edge Cases Handled

1. **Small letters (dots, periods):**
   - Height 3-7px, only 2px above baseline
   - Most sensitive to baseline positioning errors
   - Now positioned correctly

2. **Paragraph indents:**
   - Can be 100-200px
   - Creates significant baseline elevation: slope × 200 ≈ 4-5px
   - Now accounted for

3. **Multiple paragraphs:**
   - Only first line of paragraph gets indent
   - Baseline recalculation only applied when indent > 0

## Verification

### Test: notebooks/out13.png
- Single angled line with slope 0.008924
- Paragraph indent: 192px
- **Before:** Rightmost letters clipped to 2-5px
- **After:** All letters render at full height

### Test: images/dvurog_p021.png
- Full page with 41 lines, including line 24
- 331 words total
- **Before:** Line 24 (and similar) had clipping
- **After:** All lines render correctly

## Additional Improvements Made

1. **Fragment filtering** (earlier fix):
   - Filter letters < 3×3px during extraction
   - Removes noise/dots that split from letters

2. **Line height calculation:**
   - Includes baseline_rise in allocated space
   - Ensures enough vertical room for angled lines

## Lessons Learned

1. **Paragraph formatting affects geometry:**
   - Indents change the coordinate space
   - Must recalculate after applying layout changes

2. **Small differences matter:**
   - 4-5px error is catastrophic for 7px tall dots
   - Edge cases (punctuation) are the canary

3. **Order of operations:**
   - Calculate baseline → Apply indent → Recalculate baseline
   - Can't finalize geometry until all layout is known

4. **Testing with real content:**
   - Synthetic tests passed but real pages failed
   - Real documents have indents, varied fonts, punctuation

---

**Status:** ✅ COMPLETE AND VERIFIED  
**Impact:** Critical fix for text quality  
**Complexity:** High - required understanding coordinate transforms and layout interactions
