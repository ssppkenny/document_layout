# Bug Fix: IndexError in margins() Function

## Issue

When running layout-based processing on certain images, the system encountered an IndexError:

```
IndexError: index 28 is out of bounds for axis 0 with size 28
```

This occurred in the `margins()` function at line:
```python
nbs = points[nbs_inds]
```

## Root Cause

The `margins()` function uses a KDTree to find the k-nearest neighbors (k=50) for margin detection. However, when a text box contains very few words (e.g., 28 words), requesting 50 neighbors results in:

1. The KDTree returns indices that may include the total number of points as an index
2. When there are only 28 points (0-27 valid indices), index 28 is out of bounds
3. This happens because KDTree pads the result with invalid indices when k > number of points

## Solution

Applied three fixes to `src/ocr_reflow/main.py`:

### Fix 1: Limit k to available points
```python
# Before
dists_left, inds_left = kdtree.query(left_points, k=50)
dists_right, inds_right = kdtree.query(right_points, k=50)

# After
k_neighbors = min(50, len(points))
dists_left, inds_left = kdtree.query(left_points, k=k_neighbors)
dists_right, inds_right = kdtree.query(right_points, k=k_neighbors)
```

### Fix 2: Filter invalid indices for left margins
```python
# After getting neighbor indices
nbs_inds = nbs_inds[1:]
# Add filtering
nbs_inds = nbs_inds[nbs_inds < len(points)]
nbs = points[nbs_inds]
```

### Fix 3: Filter invalid indices for right margins
```python
# Same filtering applied to right margin calculation
nbs_inds = nbs_inds[1:]
# Add filtering
nbs_inds = nbs_inds[nbs_inds < len(points)]
nbs = points[nbs_inds]
```

### Fix 4: Early return for edge cases
```python
def margins(words):
    """
    Detect left and right margins of text lines from word bounding boxes.
    Returns lists of (x, y) points representing the margin positions.
    """
    # Return empty margins if too few words
    if len(words) < 2:
        return [], []
    
    # ... rest of function
```

## Testing

The fix was applied to handle the specific case where:
- Image: `images/kf_p015.png`
- Layout detection found 18 boxes
- One text box at y=886 had very few words
- Previously caused IndexError, now handles gracefully

## Impact

- **Before**: System crashed with IndexError when processing text boxes with few words
- **After**: System handles small text boxes gracefully without errors
- **Performance**: No performance impact, only adds safety checks
- **Compatibility**: Fully backward compatible, improves robustness

## Files Modified

1. `src/ocr_reflow/main.py` - Fixed `margins()` function
2. `QUICK_REFERENCE.md` - Added troubleshooting entry

## Prevention

The fix prevents similar issues by:
1. Checking the actual number of points before querying neighbors
2. Filtering out any invalid indices returned by KDTree
3. Early return for edge cases with too few words
4. Using numpy boolean indexing for safe filtering

## Date

Fixed: January 31, 2026

## Status

✅ **RESOLVED** - System now handles all text box sizes correctly
