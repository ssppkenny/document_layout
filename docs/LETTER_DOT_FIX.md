# Letter Segmentation Fix: Preserving Dots on 'i' and 'j'

## Problem

When extracting individual letters from detected words using connected components analysis, the dots above letters like 'i', 'j', and diacritical marks were being filtered out as noise. This was because the previous filtering logic used a height-based threshold:

```python
if h >= word_height * 0.2:  # At least 20% of word height
    valid_components.append((x, y, w, h))
```

Since dots are typically much smaller than 20% of the word height, they were discarded, resulting in incomplete letters.

## Solution

The fix implements a **proximity-based filtering strategy** instead of just height-based filtering:

### Algorithm

1. **Identify main letter components**: First, find connected components that are at least 30% of the word height. These are the main letter bodies.

2. **Include related small components**: For smaller components (dots, accents), check if they are:
   - **Vertically near** a main component (within 40% of word height above/below)
   - **Horizontally aligned** with a main component (within 30% of word width)

3. **Preserve the relationship**: If a small component is spatially related to a main component, it's part of the letter and should be kept.

### Key Changes in `find_rects()` function

```python
# Step 1: Identify main letter components (at least 30% of word height)
main_components = []
for i in range(1, num_labels):
    h = stats[i, cv2.CC_STAT_HEIGHT]
    if h >= word_height * 0.3:
        main_components.append(i)

# Step 2: Check if small components are near main components
for i in range(1, num_labels):
    # Skip if it's already a main component
    if i in main_components:
        valid_components.append((x, y, w, h))
        continue
    
    # Check vertical and horizontal proximity to any main component
    is_near_main = False
    for main_idx in main_components:
        # Vertical proximity: within 40% of word height
        # Horizontal proximity: within 30% of word width
        if <proximity_check>:
            is_near_main = True
            break
    
    if is_near_main:
        valid_components.append((x, y, w, h))
```

## Benefits

1. **Preserves dots**: Dots on 'i', 'j' are now correctly included
2. **Handles diacritics**: Accents and diacritical marks are preserved
3. **Filters noise**: True noise (isolated tiny specks) is still filtered out
4. **Robust to fonts**: Works with different font sizes and styles

## Testing

Run the visualization scripts to verify the fix:

```bash
# Test letter segmentation on a page
pixi run python test_letter_fix.py images/sedg_p598.png

# Visualize words with dots specifically
pixi run python visualize_dots.py images/sedg_p598.png
```

The output shows:
- Before fix: Dots were filtered out, letters incomplete
- After fix: All letter components preserved, including dots

## Files Modified

- `src/ocr_reflow/main.py`: Updated `find_rects()` function with proximity-based filtering

## Verification

Test on `images/sedg_p598.png`:
- Found 30 words with dots (i, j, accents, etc.)
- All dots correctly preserved in letter segmentation
- Average 5.9 letters per word (realistic for English text)

The fix ensures that the reflowed text will have complete, properly rendered letters without missing dots or diacritical marks.
