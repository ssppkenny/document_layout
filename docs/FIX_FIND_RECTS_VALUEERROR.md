# Fix for find_rects ValueError

## Problem

The `find_rects()` function in `main.py` was failing when called from Jupyter notebooks with the following error:

```
ValueError: too many values to unpack (expected 4)
```

## Root Cause

The function was expecting word arrays with 4 values:
```python
for xmin, ymin, xmax, ymax in line_words:
```

However, DocTR returns word arrays with 5 values (including confidence score):
```python
[xmin, ymin, xmax, ymax, confidence]
```

## Solution

Modified the `find_rects()` function to handle both formats:

```python
def find_rects(img, line_words):
    rects = []
    # Handle both formats: (xmin, ymin, xmax, ymax) or (xmin, ymin, xmax, ymax, confidence)
    for word in line_words:
        if len(word) == 5:
            xmin, ymin, xmax, ymax, _ = word  # Unpack 5 values, ignore confidence
        else:
            xmin, ymin, xmax, ymax = word  # Unpack 4 values
        # ... rest of function
```

## Testing

Verified the fix works with both formats:

```bash
✓ Testing with 5-element words (with confidence): Success!
✓ Testing with 4-element words (without confidence): Success!
```

## Impact

- ✅ Jupyter notebooks now work correctly
- ✅ Backward compatible with 4-element word arrays
- ✅ No changes needed to calling code
- ✅ All existing functionality preserved

## Files Modified

- `src/ocr_reflow/main.py` - Updated `find_rects()` function (lines 141-148)

## Status

✅ **FIXED** - The error is resolved and tested.
