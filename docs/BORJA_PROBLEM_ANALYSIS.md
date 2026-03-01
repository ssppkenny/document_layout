# Börja/inför Segmentation Problem - Analysis Summary

## Visualization Created

File: `/tmp/borja_infor_segmentation_analysis.png`

This shows side-by-side for the word "Börja":
1. **ORIGINAL** - The word as it appears in the image
2. **RAW COMPONENTS (5)** - Individual connected components with colored boxes
3. **AFTER MERGING (2)** - Components after the merging algorithm

## The Problem

### Current State
- **Raw components detected: 5**
- **After merging: 2**
- **Expected: ~5 components after proper merging** (B, ö, r, j, a)

### Root Cause Analysis

The word "Börja" only shows 5 raw components, which means:
1. **Missing letters**: r, j, a are NOT in the detected word box
2. **The doctr word detector is only capturing "Bö"** (first 2 letters)
3. **The rest of the word ("rja") is in a separate word box**

This explains why you see "half of letter ö" - the algorithm is:
1. Detecting "Bö" as one word box
2. Extracting only those components (B parts + ö parts)
3. Merging them incorrectly into 2 components
4. When reflowed, the bounding box of these 2 components creates visual artifacts

## What to Check

### In `/tmp/borja_infor_segmentation_analysis.png`:

Look at the "RAW COMPONENTS" column:
- **Component 1 (Red)**: Should be part of B
- **Component 2 (Green)**: Should be small dot/accent
- **Component 3 (Blue)**: Should be ö dot 1
- **Component 4 (Yellow)**: Should be ö dot 2  
- **Component 5 (Magenta)**: Should be ö base

If you only see 5 components total, the letters r, j, a are missing!

### In `output_reflowed.png`:

Check the first few lines:
- Does "Börja" show as complete or partial?
- Is there visual garbage/artifacts between letters?
- Are letters cut off or duplicated?

## Likely Solutions

### Option 1: Fix doctr Word Detection
The word detector is splitting "Börja" incorrectly. We need to:
- Post-process doctr output to merge word boxes that are too close
- Or use a different word detection strategy

### Option 2: Expand Word Boxes (What We Tried Earlier)
- Expand word boxes horizontally to capture adjacent letters
- But this created other problems (too many small parts)

### Option 3: Use Line-Level Detection Instead
- Detect entire lines, not individual words
- Then segment words ourselves based on spacing
- This gives us more control over boundaries

## Next Steps

1. **First**: Look at the visualization to confirm the analysis
2. **Check**: How many word boxes does doctr create for "Börja"?
3. **Decide**: Which solution approach to take

## To Continue Investigation

Run this to see ALL word boxes in line 4:
```bash
cd /home/sergey/code/python/segmentation && pixi run python -c "
import cv2
from doctr.io import DocumentFile
from doctr.models import detection_predictor

img = cv2.imread('images/gang_p023_lines1.png')
img_h, img_w = img.shape[:2]

det_predictor = detection_predictor(arch='db_resnet50', pretrained=True)
docs = DocumentFile.from_images(['images/gang_p023_lines1.png'])
result = det_predictor(docs)
words_array = result[0]['words']

print('Words around line 4 (y=250-290):')
for i, word_box in enumerate(words_array):
    xmin = int(word_box[0] * img_w)
    ymin = int(word_box[1] * img_h)
    xmax = int(word_box[2] * img_w)
    ymax = int(word_box[3] * img_h)
    
    if 250 < ymin < 290:
        width = xmax - xmin
        print('  Box {}: x={}-{} ({}px)'.format(i, xmin, xmax, width))
"
```

This will show if "Börja" is split across multiple boxes.

---

**Generated**: 2026-02-28
