# Swedish Diacritics Duplication Fix - Summary

## Problem Description

After the initial Swedish diacritics fix, words like "Börja" were appearing as "Böörja" and "inför" as "inföör" in the reflowed output. The diacritics (ö) were being duplicated.

## Root Cause

The bug was in the diacritic merging logic in `src/ocr_reflow/main.py` (find_rects function):

1. Diacritics were being grouped correctly (e.g., two dots of "ö")
2. The algorithm tried to find matching base letters below each diacritic group
3. **BUG**: If NO matching base letter was found, the diacritics were NOT marked as merged
4. Result: The ungrouped diacritics were added AGAIN as separate components
5. This caused each dot to appear as a separate character → duplication

### Code Flow Before Fix:
```python
for diacritic_group in diacritic_groups:
    # ... find matching base letters ...
    
    if matching_components:  # ← BUG: only marks as merged if match found
        # Merge and mark as used
        for dot_idx, _, _, _, _ in diacritic_group:
            merged_indices.add(dot_idx)
    # If no match, diacritics NOT marked as merged!

# Later: add non-merged components
for comp_idx, (x, y, w, h) in enumerate(valid_components):
    if comp_idx not in merged_indices:  # ← Diacritics added again!
        merged_components.append((x, y, w, h))
```

## Solution

Modified the merging logic to ALWAYS mark diacritic components as merged, regardless of whether a matching base letter is found:

```python
for diacritic_group in diacritic_groups:
    # ALWAYS mark diacritics as merged FIRST
    for dot_idx, _, _, _, _ in diacritic_group:
        merged_indices.add(dot_idx)
    
    if matching_components:
        # Merge diacritics with base letters
        # ... create merged component ...
    else:
        # No base letter found - create component from diacritic group alone
        # (handles isolated diacritics)
        # ... create component from group ...
```

This ensures diacritics are never added twice:
1. If base letter found → merged component created
2. If no base letter found → standalone diacritic component created  
3. In both cases → diacritics marked as merged, so not added again

## Test Results

### Before Fix:
- "Börja" → "Böörja" ❌
- "inför" → "inföör" ❌

### After Fix:
- "Börja" → "Börja" ✅
- "inför" → "inför" ✅

### Regression Tests:
- ✅ Swedish text (images/gang_p023.png) - No duplication
- ✅ Russian text (images/dvurog_p076.png) - No regression
- ✅ English text (images/jtg_p033.png, mh_p013.png) - No regression

## Files Modified

- `src/ocr_reflow/main.py` - Lines ~430-465 (find_rects function)

## Verification

Run verification script:
```bash
python verify_no_duplication.py
```

Then manually inspect `output_reflowed.png` to confirm Swedish characters appear correctly without duplication.

## Technical Details

### Why Diacritics Might Not Find Base Letters

Possible reasons a diacritic group might not find matching base components:
1. Very strict alignment thresholds
2. Unusual font rendering where dots are far from base
3. Noise or artifacts in the image
4. Legitimate standalone diacritics (rare but possible)

In all these cases, we still need to prevent duplicate placement, which is what the fix achieves.

## Related Issues

This fix complements the earlier Swedish diacritics fix (which grouped adjacent diacritics). Together they solve:
1. **First fix**: Group multiple dots (ä, ö) into single diacritic units
2. **This fix**: Prevent duplicates by marking all diacritics as merged

## Future Considerations

If you encounter issues with:
- Diacritics appearing without base letters
- Missing characters

Check the threshold values in the merging logic:
- `vertical_gap < median_height` - Controls how far above base letter diacritics can be
- `horizontal_overlap > 0 or horizontal_gap < median_height * 0.5` - Controls horizontal alignment

These may need adjustment for specific fonts or scanning quality.
