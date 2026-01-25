# Project Workflow

This document explains the complete workflow of the text segmentation and reflow process.

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                                  │
│                    (Scanned Document)                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TEXT DETECTION (doctr)                            │
│  - Detect text regions in the image                                 │
│  - Find word-level bounding boxes                                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CHARACTER SEGMENTATION (OpenCV)                         │
│  - Extract individual character bounding boxes                      │
│  - Calculate baseline for each character                            │
│  - Remove nested/enclosed rectangles                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LINE GROUPING (Spatial Analysis)                   │
│  - Group characters into lines based on y-coordinates               │
│  - Use KD-Tree for efficient spatial queries                        │
│  - Sort characters within each line                                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PARAGRAPH DETECTION                                 │
│  - Analyze horizontal indentation                                   │
│  - Detect short lines                                               │
│  - Mark paragraph boundaries                                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   WORD WRAPPING & REFLOW                             │
│  - Calculate available width                                        │
│  - Place characters with proper spacing                             │
│  - Prevent single-letter word splits                                │
│  - Handle line breaks intelligently                                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LINE SPACING CALCULATION                           │
│  - Calculate 95th percentile baseline values                        │
│  - Apply safety cap for outliers                                    │
│  - Ensure consistent vertical spacing                               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LETTER PLACEMENT                                   │
│  - Place each letter with baseline alignment                        │
│  - Apply paragraph indentation                                      │
│  - Add appropriate spacing between lines/paragraphs                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT IMAGE                                 │
│                    (Reflowed Document)                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Process Steps

### 1. Text Detection (main.py)
```python
detector = detection_predictor(arch='db_resnet50', pretrained=True)
result = detector(doc)
```
- Uses doctr's DB_ResNet50 model
- Detects word-level bounding boxes
- Returns coordinates for each detected text region

### 2. Character Segmentation (find_rects function)
```python
rects = find_rects(img, line_words)
```
- Extracts region for each word
- Applies Otsu thresholding
- Uses connected components analysis
- Finds individual character bounding boxes
- Removes enclosed rectangles (divide_conquer_4d)

### 3. Baseline Calculation (Letter dataclass)
```python
Letter(xmin, ymin, xmax, ymax, bl)
```
- `bl` = baseline offset from bottom
- Calculated as: `(ymax - ymin) - (centroid_y - ymin)`
- Critical for proper vertical alignment

### 4. Line Grouping (margins function + sorting)
```python
# Group by y-coordinate proximity
# Use KD-Tree for efficient neighbor finding
kdtree = KDTree(points)
```
- Groups characters with similar y-coordinates
- Handles multiple columns
- Sorts characters left-to-right within lines

### 5. Paragraph Detection (detect_paragraphs_and_spacing_from_lines)
```python
paragraph_starts, avg_first_xmin = detect_paragraphs_and_spacing_from_lines(lines, width)
```
- Analyzes first letter x-position in each line
- Detects significant right-shift (indentation)
- Marks short lines as paragraph endings

### 6. Word Wrapping (create_page_with_word_wrapping)
```python
# For each character:
# 1. Check if it fits on current line
# 2. Check if word split would be bad (1 letter)
# 3. Move to new line if needed
# 4. Add appropriate spacing
```

Key logic:
- Word boundary: space ≥ 0.5 × avg_char_width
- Prevent split: if either side has ≤ 1 letter
- Preserve original spacing between characters

### 7. Line Spacing (percentile-based calculation)
```python
# Collect all baseline values
all_above_baseline = [height - bl for all letters]
all_below_baseline = [bl for all letters]

# Use 95th percentile
max_above = percentile_95(all_above_baseline)
max_below = percentile_95(all_below_baseline)

# Apply safety cap
line_height = min(calculated, typical_height * 2.5)
```

Benefits:
- Ignores 5% worst outliers
- Prevents one bad letter from ruining spacing
- Maintains consistent appearance

### 8. Letter Placement (final rendering)
```python
# For each line:
# 1. Calculate baseline position
# 2. Place each letter aligned to baseline
# 3. Apply horizontal and vertical spacing
# 4. Apply paragraph indentation

y_offset = baseline_y - scaled_height + scaled_bl
```

## Data Flow Example

**Input:** Scanned page with text "The fact is..."

```
Step 1: Detection
  → Words: ["The", "fact", "is"]

Step 2: Segmentation
  → Characters: [T][h][e] [f][a][c][t] [i][s]

Step 3: Line Grouping
  → Line 1: [T][h][e] [f][a][c][t] [i][s]

Step 4: Paragraph Detection
  → Paragraph 1 starts at character 0

Step 5: Word Wrapping (narrow page)
  → Line 1: [T][h][e]
  → Line 2: [f][a][c][t] [i][s]
  
  (NOT split as [f] on line 1, [a][c][t] on line 2!)

Step 6: Line Spacing
  → Line height: 45 pixels (based on 95th percentile)

Step 7: Placement
  → Render at positions with baseline alignment

Output: Reflowed page with proper spacing
```

## Key Algorithms

### Word Split Prevention
```
IF letter would overflow:
    IF space < 0.5 × avg_char_width:  # In middle of word
        count_backward → letters_on_current
        count_forward → letters_on_next
        
        IF letters_on_current ≤ 1 OR letters_on_next ≤ 1:
            move_word_to_next_line()
```

### Baseline Alignment
```
For each letter:
    baseline_y = line_baseline_position
    y_top = baseline_y - (height - baseline_offset)
    
    place_letter(x, y_top)
```

### Line Height Calculation
```
1. Collect all baseline values
2. Sort values
3. Take 95th percentile
4. Calculate: height = above_95 + below_95 + spacing
5. Cap at: typical_height × 2.5 + spacing
```

## Performance Considerations

- **KD-Tree**: O(n log n) for spatial queries
- **Divide & Conquer**: O(n² log n) for rectangle enclosure
- **Percentile**: O(n log n) with sorting
- **Overall**: Linear in number of characters for main processing

## Output Files

1. **out.png** - Main reflowed output
2. **out1.png** - Debug: character bounding boxes
3. **out2.png** - Debug: detected lines

---

For more details, see the source code in `src/main.py` and `src/reflow.py`.
