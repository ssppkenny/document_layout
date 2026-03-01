# Capital J Duplication Fix

## Problem
Word "Jag" (W12) showed capital J repeated twice on the reflowed page.

## Root Cause
Capital J was being split into 2 connected components during component detection (likely due to font rendering or scanning artifacts). When horizontal letter merging was disabled (to fix Swedish i,n,f merging issue), these J parts remained separate and both were placed on the reflowed page, creating a duplicate J.

## Analysis
```
Raw components in "Jag":
- 5 components total
- First 2 components: J part 1, J part 2 (gap 0-2px)
- Next 3 components: a, g, (other)

Expected after processing:
- 3 components: J (merged), a, g
```

The J parts have:
- **Horizontal gap**: 0-2 pixels (touching or nearly touching)
- **Vertical overlap**: ~100% (perfectly aligned)

This is different from separate Swedish letters:
- **Horizontal gap**: 3-5+ pixels (clearly separate)
- **Vertical overlap**: ~100% but with larger gap

## Solution
Re-enabled horizontal letter merging with **VERY strict criteria**:

```python
# OLD: Disabled completely (caused J duplication)
if False and len(main_letters_to_merge) > 1:

# NEW: Enabled with strict 3px threshold
if len(main_letters_to_merge) > 1:
    # ...merging logic...
    if (horizontal_gap < 3 and  # ONLY 0-2px gaps (split letters)
        vertical_overlap > min_height * 0.7):  # Well aligned
        # Merge
```

### Key Changes:
1. **Horizontal gap threshold**: < 3px (absolute pixels, not percentage)
   - Catches split J parts (0-2px gap)
   - Avoids merging Swedish i,n,f (3-5px gaps)

2. **Vertical overlap**: > 70% of min height
   - Ensures components are well-aligned vertically

## Test Results

### Before Fix:
- **Jag**: 5 components → J appears twice ✗
- **inför**: 5 components (correct) ✓

### After Fix:
- **Jag**: 3 components → J appears once ✓
- **inför**: 5 components → i,n,f stay separate ✓

## Files Modified
- **src/ocr_reflow/main.py**, lines 301-349: Re-enabled horizontal merging with strict 3px threshold

## Verification
Check `output_reflowed.png`:
- [x] Word "Jag" shows only ONE capital J (not duplicated)
- [x] Swedish words "inför", "Börja" still display correctly
- [x] All diacritics (i/j dots, ö/ä dots, å rings) display correctly

---
**Date:** 2026-02-28  
**Issue:** Capital J duplicated in word "Jag"  
**Solution:** Re-enabled letter merging with strict 3px threshold for genuinely split letters
