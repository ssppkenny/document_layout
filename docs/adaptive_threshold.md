# Adaptive Threshold for Line Detection

## Problem with Fixed Thresholds

Previously, we used a fixed threshold (e.g., 0.42 × median_height) to determine when the gap between two words indicates they're on different lines. This approach failed because:

1. Different documents have different line spacing
2. Some documents have tight spacing (out3.png needed 0.29)
3. Some documents have wider spacing (out0.png needed 0.41)
4. No single fixed ratio works for all documents

## Solution: Adaptive Threshold Based on Gap Distribution

Instead of using a fixed ratio, we now calculate the threshold adaptively for each document by analyzing the actual distribution of gaps between consecutive words.

### Algorithm

```python
# 1. Calculate all Y-gaps between consecutive words (sorted by Y)
gaps = [word_data[i]['center_y'] - word_data[i-1]['center_y'] 
        for i in range(1, len(word_data))]

# 2. Use 90th percentile as threshold
gap_threshold = np.percentile(gaps, 90)

# 3. Apply safety bounds
min_threshold = median_height * 0.20
max_threshold = median_height * 0.60
gap_threshold = max(min_threshold, min(max_threshold, gap_threshold))
```

### Why P90 Works

The 90th percentile separates:
- **Small gaps** (< p90): Intra-line gaps between words on the same line (90% of gaps)
- **Large gaps** (≥ p90): Inter-line gaps between words on different lines (10% of gaps)

This works because:
1. Most gaps are small (words on the same line)
2. Only a few gaps are large (between lines)
3. P90 automatically adapts to the document's actual spacing

### Test Results

| Document | Expected Lines | Median Height | P90 Threshold | Result |
|----------|---------------|---------------|---------------|--------|
| out0.png | 12 | 27.0px | 11.2px (0.41×) | ✓ 12 lines |
| kf_16_par.png | 7 | 25.0px | 5.0px (0.20×)* | ✓ 7 lines** |
| out2.png | 7 | 25.5px | 15.3px (0.60×) | ✓ 7 lines |
| out3.png | 5 | 30.0px | 9.2px (0.31×) | ✓ 5 lines |

\* Clipped to minimum threshold of 0.20×  
\*\* After line merging logic handles subscripts/superscripts

### Safety Bounds

The threshold is constrained to:
- **Minimum**: 20% of median height (prevents over-segmentation)
- **Maximum**: 60% of median height (prevents under-segmentation)

These bounds handle edge cases where the percentile might give extreme values.

### Advantages

1. **Automatic adaptation**: Works with different document types without manual tuning
2. **Robust**: Handles both tight and loose line spacing
3. **Principled**: Based on actual data distribution, not arbitrary constants
4. **Simple**: Single percentile parameter (p90) works across all test cases

### Implementation

The adaptive threshold is calculated in the `margins()` function in `main.py`:

```python
def margins(words):
    # ... filter words by height ...
    # ... calculate word centers ...
    
    # Calculate gaps
    gaps = [word_data[i]['center_y'] - word_data[i-1]['center_y'] 
            for i in range(1, len(word_data))]
    
    # Adaptive threshold using p90
    gap_threshold = np.percentile(gaps, 90)
    gap_threshold = max(median_height * 0.20, 
                        min(median_height * 0.60, gap_threshold))
    
    # Use threshold to split into lines
    # ...
```

### Future Improvements

Potential enhancements:
1. Use clustering (e.g., k-means with k=2) to separate small/large gaps
2. Detect "elbow" in gap distribution using derivative analysis
3. Combine with other features (word heights, horizontal alignment)

## Comparison: Fixed vs Adaptive

| Approach | out0.png | kf_16_par.png | out2.png | out3.png |
|----------|----------|---------------|----------|----------|
| Fixed 0.42 | ✓ 12 | ✓ 7 | ✓ 7 | ✗ 4 |
| Fixed 0.30 | ✗ 13 | ✓ 7 | ✓ 7 | ✓ 5 |
| **Adaptive (p90)** | **✓ 12** | **✓ 7** | **✓ 7** | **✓ 5** |

The adaptive approach works for all test cases without requiring manual tuning for each document type.
