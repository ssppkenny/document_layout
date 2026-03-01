#!/usr/bin/env python3
"""
Comprehensive visualization of the "inför" word segmentation and reflow
Shows the original, raw components, merged components, and reflowed output
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def visualize_infor_word():
    print("="*80)
    print("DETAILED VISUALIZATION: inför word")
    print("="*80)
    print()

    img = cv2.imread('images/gang_p023_lines1.png')

    if img is None:
        print("✗ Could not load image")
        return

    # inför word location (Line 4, Word 5)
    xmin, ymin, xmax, ymax = 192, 63, 298, 111

    word_img = img[ymin:ymax, xmin:xmax]
    word_h, word_w = word_img.shape[:2]

    print(f"Word box: ({xmin},{ymin}) → ({xmax},{ymax})")
    print(f"Size: {word_w}x{word_h} pixels")
    print()

    # === STEP 1: Get raw connected components ===
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    raw_components = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w >= 2 and h >= 2:
            raw_components.append((x, y, w, h))

    raw_components.sort(key=lambda c: c[0])

    print(f"STEP 1: Raw components ({len(raw_components)} found)")
    print("-" * 60)

    heights = [h for x, y, w, h in raw_components]
    median_h = np.median(heights)
    print(f"Median height: {median_h:.1f}px")
    print()

    for i, (x, y, w, h) in enumerate(raw_components):
        y_pct = (y / word_h) * 100
        is_dot = (w < median_h * 0.8 and h < median_h * 0.4 and y < word_h * 0.4)
        label = "DOT" if is_dot else "LETTER"
        print(f"  {i+1:2d}. x={x:3d} y={y:3d} ({y_pct:4.1f}%) {w:2d}x{h:2d} [{label}]")

    print()

    # === STEP 2: Get merged components from find_rects ===
    words_list = [[xmin, ymin, xmax, ymax]]
    rectangles = find_rects(img, words_list, debug=False)

    print(f"STEP 2: After find_rects ({len(rectangles)} components)")
    print("-" * 60)

    for i, rect in enumerate(rectangles):
        rxmin, rymin, rxmax, rymax = rect
        w = rxmax - rxmin
        h = rymax - rymin
        # Convert to word-relative
        rel_x = rxmin - xmin
        rel_y = rymin - ymin
        print(f"  {i+1:2d}. word-rel: x={rel_x:3d} y={rel_y:3d} size {w:2d}x{h:2d}")

    print()

    # === STEP 3: Create visualizations ===
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
        (255, 128, 0), (128, 255, 0)
    ]

    scale = 8

    # 3a. Original word
    vis_original = cv2.resize(word_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 3b. Raw components
    vis_raw = word_img.copy()
    for i, (x, y, w, h) in enumerate(raw_components):
        color = colors[i % len(colors)]
        cv2.rectangle(vis_raw, (x, y), (x+w, y+h), color, 1)
        cv2.putText(vis_raw, str(i+1), (x, y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    vis_raw = cv2.resize(vis_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 3c. Merged components
    vis_merged = word_img.copy()
    for i, rect in enumerate(rectangles):
        rxmin, rymin, rxmax, rymax = rect
        # Convert to word-relative
        rel_x1 = rxmin - xmin
        rel_y1 = rymin - ymin
        rel_x2 = rxmax - xmin
        rel_y2 = rymax - ymin

        color = colors[i % len(colors)]
        cv2.rectangle(vis_merged, (rel_x1, rel_y1), (rel_x2, rel_y2), color, 2)
        cv2.putText(vis_merged, str(i+1), (rel_x1+2, rel_y1+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    vis_merged = cv2.resize(vis_merged, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # === STEP 4: Stack visualizations ===
    label_h = 50

    def add_label(img, text):
        label = np.ones((label_h, img.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(label, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        return np.vstack([label, img])

    vis_original = add_label(vis_original, f"ORIGINAL (infor)")
    vis_raw = add_label(vis_raw, f"RAW ({len(raw_components)} components)")
    vis_merged = add_label(vis_merged, f"MERGED ({len(rectangles)} components)")

    # Make same height
    max_h = max(vis_original.shape[0], vis_raw.shape[0], vis_merged.shape[0])

    def pad_height(img, target_h):
        if img.shape[0] < target_h:
            pad = np.ones((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img = np.vstack([img, pad])
        return img

    vis_original = pad_height(vis_original, max_h)
    vis_raw = pad_height(vis_raw, max_h)
    vis_merged = pad_height(vis_merged, max_h)

    # Stack horizontally
    spacer = np.ones((max_h, 30, 3), dtype=np.uint8) * 200
    combined = np.hstack([vis_original, spacer, vis_raw, spacer, vis_merged])

    # Add title
    title_h = 80
    title = np.ones((title_h, combined.shape[1], 3), dtype=np.uint8) * 230
    cv2.putText(title, "Word: infor (Swedish)", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    combined = np.vstack([title, combined])

    cv2.imwrite('/tmp/infor_detailed_visualization.png', combined)

    print("="*80)
    print("VISUALIZATION SAVED")
    print("="*80)
    print()
    print("File: /tmp/infor_detailed_visualization.png")
    print()
    print("This shows:")
    print("  1. ORIGINAL - The word as it appears in the image")
    print("  2. RAW - Connected components with colored boxes (numbered)")
    print("  3. MERGED - Components after merging diacritics (numbered)")
    print()
    print("Expected structure of 'inför':")
    print("  i = letter i + dot above")
    print("  n = letter n")
    print("  f = letter f")
    print("  ö = letter o + 2 dots above (should merge to 1 component)")
    print("  r = letter r")
    print("  Total: ~5 components expected after merging")
    print()
    print(f"Actual: {len(rectangles)} components")
    print()
    print("Please examine /tmp/infor_detailed_visualization.png and tell me:")
    print("  - Which components in MERGED look wrong?")
    print("  - Are the ö dots (2 small components near top) merged with 'o' base?")
    print("  - Are there extra components or missing components?")
    print("  - Do you see 'parts of letters' in the MERGED view?")
    print()
    print("="*80)

if __name__ == '__main__':
    visualize_infor_word()
