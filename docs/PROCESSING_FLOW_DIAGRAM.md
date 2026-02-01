# Document Processing Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                                  │
│                    (Book page, paper, etc.)                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYOUT ANALYSIS (YOLO)                            │
│  Detects: text, titles, figures, tables, formulas, captions, etc.   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BOX PAIRING & GROUPING                          │
│  • Figures + Captions → "figure_and_caption"                        │
│  • Formulas + Captions → "isolate_formula_and_caption"             │
│  • Tables + Captions + Footnotes → "table_and_caption"             │
│  • Intersecting Plain Text → grouped "plain text"                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SORT BOXES (Y, then X)                           │
│              Ensures natural reading order                           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────┴───────┐
                    │               │
        ┌───────────▼───────────┐   │
        │   TEXT BOXES          │   │
        │  (plain text, title)  │   │
        └───────────┬───────────┘   │
                    │               │
                    ▼               │
        ┌───────────────────────┐   │
        │  CHARACTER DETECTION  │   │
        │     (doctr model)     │   │
        └───────────┬───────────┘   │
                    │               │
                    ▼               │
        ┌───────────────────────┐   │
        │  BASELINE ANALYSIS    │   │
        │  (polyfit per line)   │   │
        └───────────┬───────────┘   │
                    │               │
                    ▼               │
        ┌───────────────────────┐   │
        │  REFLOW WITH          │   │
        │  WORD WRAPPING        │   │
        └───────────┬───────────┘   │
                    │               │
                    │       ┌───────▼────────────┐
                    │       │   NON-TEXT BOXES   │
                    │       │ (figures, tables,  │
                    │       │   formulas, etc.)  │
                    │       └───────┬────────────┘
                    │               │
                    │               ▼
                    │       ┌───────────────────┐
                    │       │  EXTRACT BOX      │
                    │       │  AS IMAGE         │
                    │       └───────┬───────────┘
                    │               │
                    │               ▼
                    │       ┌───────────────────┐
                    │       │  APPLY ZOOM       │
                    │       │  (zoom_factor)    │
                    │       └───────┬───────────┘
                    │               │
                    │               ▼
                    │       ┌───────────────────┐
                    │       │  RESIZE IF        │
                    │       │  TOO WIDE         │
                    │       └───────┬───────────┘
                    │               │
                    │               ▼
                    │       ┌───────────────────┐
                    │       │  CENTER           │
                    │       │  HORIZONTALLY     │
                    │       └───────┬───────────┘
                    │               │
                    └───────────────┴───────────┐
                                                │
                                                ▼
                                ┌───────────────────────────┐
                                │   COMPOSE ON NEW PAGE     │
                                │  (dynamic page growth)    │
                                └───────────────┬───────────┘
                                                │
                                                ▼
                                ┌───────────────────────────┐
                                │     OUTPUT IMAGE          │
                                │  (reflowed document)      │
                                └───────────────────────────┘
```

## Detailed Processing Steps

### Step 1: Layout Analysis

```
Input Image (e.g., 2481 x 3508 pixels)
        │
        ▼
    ┌───────────────────────────────────┐
    │  YOLO Layout Detection Model      │
    │  (DocLayout-YOLO-DocStructBench)  │
    └───────────────┬───────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────┐
    │  Raw Boxes + Types                │
    │  Box 1: (50, 100, 500, 200) "title"     │
    │  Box 2: (50, 220, 500, 800) "plain text"│
    │  Box 3: (100, 850, 450, 1200) "figure"  │
    │  Box 4: (100, 1210, 450, 1250) "figure_caption" │
    │  ...                              │
    └───────────────────────────────────┘
```

### Step 2: Box Pairing

```
    ┌────────────────────────────────────────┐
    │  Figure Pairing                        │
    │  ─────────────────                     │
    │  For each figure_caption:              │
    │    1. Find nearest unpaired figure     │
    │    2. Calculate centroid distance      │
    │    3. Pair with nearest                │
    │    4. Create union bounding box        │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Result:                               │
    │  Box 3 + Box 4 → "figure_and_caption"  │
    │  Bounds: (100, 850, 450, 1250)         │
    └────────────────────────────────────────┘

    Similar process for:
    • formulas + formula_captions
    • tables + table_captions + table_footnotes
```

### Step 3: Text Box Processing

```
Text Box: (50, 220, 500, 800) "plain text"
        │
        ▼
┌────────────────────────────────────────┐
│  Extract Region from Image             │
│  region_img = img[220:800, 50:500]     │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Run Text Detection (doctr)            │
│  → Word-level bounding boxes           │
│  → Normalized coordinates              │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Find Line Margins                     │
│  → Left margin points                  │
│  → Right margin points                 │
│  → Group words into lines              │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Extract Characters per Line           │
│  → Connected components                │
│  → Filter enclosed characters          │
│  → Sort by X position                  │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Calculate Baseline per Line           │
│  → Find normal-height letters          │
│  → Polyfit baseline (y = mx + c)       │
│  → Store baseline offset per letter    │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Create Letter Objects                 │
│  Letter(xmin, ymin, xmax, ymax, bl)    │
│  where bl = baseline offset            │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Reflow with Word Wrapping             │
│  → Place letters left-to-right         │
│  → Wrap to new line when needed        │
│  → Preserve baseline alignment         │
│  → Detect & indent paragraphs          │
│  → Equal line spacing                  │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Reflowed Text Block                   │
│  (temporary page with text content)    │
└────────────────────────────────────────┘
```

### Step 4: Non-Text Box Processing

```
Non-Text Box: (100, 850, 450, 1250) "figure_and_caption"
        │
        ▼
┌────────────────────────────────────────┐
│  Extract Region from Image             │
│  region_img = img[850:1250, 100:450]   │
│  size: 350 x 400 pixels                │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Apply Zoom Factor (2.5)               │
│  new_size: 875 x 1000 pixels           │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Check if Fits Available Width         │
│  available_width = 1900 (2000 - margins)│
│  875 < 1900 ✓ OK                       │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Resize Image                          │
│  cv2.resize(..., INTER_CUBIC)          │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Calculate Center Position             │
│  x_offset = 50 + (1900 - 875) / 2      │
│  x_offset = 562                        │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Place on New Page                     │
│  new_page[y:y+1000, 562:562+875] = img │
└────────────────────────────────────────┘
```

### Step 5: Page Composition

```
┌────────────────────────────────────────┐
│  Initialize New Page                   │
│  size: 2000 x 3000 (expands as needed)│
│  fill: background_color                │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  current_y = 50 (top margin)           │
└────────────────┬───────────────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
┌─────────────┐      ┌─────────────┐
│ Text Block  │      │ Non-Text    │
│             │      │ Block       │
└──────┬──────┘      └──────┬──────┘
       │                    │
       ▼                    ▼
┌────────────────────────────────────────┐
│  Check if Fits                         │
│  if current_y + height > page_height:  │
│    expand page                         │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Copy Content to Page                  │
│  new_page[current_y:...] = content     │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  Update Position                       │
│  current_y += height + spacing         │
└────────────────┬───────────────────────┘
                 │
                 └──────────┬─────────────
                            │
                 (repeat for all boxes)
                            │
                            ▼
┌────────────────────────────────────────┐
│  Crop to Actual Content                │
│  final_page = page[:current_y+50, :]   │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│  OUTPUT: Reflowed Document             │
└────────────────────────────────────────┘
```

## Data Flow Summary

```
INPUT
  │
  ├─→ Layout Detection → Boxes + Types
  │
  ├─→ Pairing Logic → Grouped Boxes
  │
  ├─→ Sorting → Reading Order
  │
  ├─→ Type-Based Processing:
  │     │
  │     ├─→ Text → Character Detection → Baseline → Reflow
  │     │
  │     └─→ Non-Text → Extract → Zoom → Center
  │
  └─→ Composition → Dynamic Page Building → OUTPUT
```

## Key Algorithms

### 1. Pairing Algorithm (O(n²) per type)
```
For each caption:
    min_dist = infinity
    nearest = None
    For each unpaired element:
        dist = euclidean_distance(caption.centroid, element.centroid)
        if dist < min_dist:
            min_dist = dist
            nearest = element
    If nearest found:
        pair(caption, nearest)
```

### 2. Baseline Detection (O(n) per line)
```
For each line:
    normal_letters = filter by height (within 1 std dev)
    lower_points = [(center_x, bottom_y) for letter in normal_letters]
    m, c = polyfit(x_coords, y_coords, degree=1)
    For each letter:
        baseline_offset = letter.ymax - (m * letter.center_x + c)
```

### 3. Word Wrapping (O(n) for n letters)
```
current_x = left_margin
current_y = top_margin
For each letter:
    scaled_width = letter.width * zoom_factor
    if current_x + scaled_width > page_width - right_margin:
        current_x = left_margin  # New line
        current_y += line_height + spacing
    place_letter(current_x, current_y, letter)
    current_x += scaled_width + letter_spacing
```

## Performance Characteristics

| Operation | Complexity | Typical Time (GPU) |
|-----------|------------|-------------------|
| Layout Detection | O(image_size) | 1-2 seconds |
| Box Pairing | O(n²) per type | <0.1 seconds |
| Text Detection | O(box_size) | 0.5-1 second per box |
| Character Extraction | O(pixels) | 0.1-0.2 seconds per line |
| Reflow | O(n) letters | <0.1 seconds |
| Composition | O(pixels) | <0.1 seconds |
| **TOTAL** | **O(image_size)** | **2-5 seconds per page** |

---

*This diagram illustrates the complete document processing pipeline from input to output.*
