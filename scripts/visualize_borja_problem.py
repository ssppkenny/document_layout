#!/usr/bin/env python3
"""
Visualize the Börja segmentation problem
Shows exactly how the word is being split into components
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def visualize_borja_problem():
    print("="*80)
    print("BÖRJA SEGMENTATION PROBLEM VISUALIZATION")
    print("="*80)
    print()

    img = cv2.imread('images/gang_p023.png')

    # The Börja word box (from doctr detection)
    xmin, ymin, xmax, ymax = 93, 2150, 163, 2203

    # Extract the word
    word_img = img[ymin:ymax, xmin:xmax]
    word_h, word_w = word_img.shape[:2]

    print("STEP 1: Extract components using find_rects")
    print("-" * 80)

    words_list = [[xmin, ymin, xmax, ymax]]
    rectangles = find_rects(img, words_list, debug=False)

    print("Word box: ({},{}) → ({},{})".format(xmin, ymin, xmax, ymax))
    print("Components extracted by find_rects: {}".format(len(rectangles)))
    print()

    # Create visualization
    vis = word_img.copy()

    # Draw each component with different colors
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Orange
    ]

    print("Components (in word coordinates):")
    for i, rect in enumerate(rectangles):
        if hasattr(rect, 'x'):
            x, y, w, h = rect.x, rect.y, rect.w, rect.h
        else:
            x, y, w, h = rect

        color = colors[i % len(colors)]
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, str(i+1), (x+2, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        print("  Component {}: ({:2d},{:2d}) {}x{} pixels".format(i+1, x, y, w, h))

    print()

    # Save visualization
    vis_large = cv2.resize(vis, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('/tmp/borja_components_boxed.png', vis_large)

    print("STEP 2: Check raw connected components")
    print("-" * 80)

    # Get raw components before merging
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

    raw_components.sort(key=lambda c: c[0])  # Sort by x

    print("Raw connected components (before merging): {}".format(len(raw_components)))
    print()

    # Visualize raw components
    vis_raw = word_img.copy()
    for i, (x, y, w, h) in enumerate(raw_components):
        color = colors[i % len(colors)]
        cv2.rectangle(vis_raw, (x, y), (x+w, y+h), color, 1)
        cv2.putText(vis_raw, str(i+1), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        y_pct = (y / word_h) * 100
        print("  Raw {}: ({:2d},{:2d}) {:2d}x{:2d} y={:4.1f}%".format(i+1, x, y, w, h, y_pct))

    print()
    vis_raw_large = cv2.resize(vis_raw, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('/tmp/borja_raw_components.png', vis_raw_large)

    print("STEP 3: Analysis")
    print("-" * 80)
    print()

    print("Expected structure of 'Börja':")
    print("  - B (one component)")
    print("  - ö = 2 dots + o base (3 components, should merge to 1)")
    print("  - r (one component)")
    print("  - j = letter + dot (2 components, should merge to 1)")
    print("  - a (one component)")
    print("  Total: ~8 raw components → ~5 after merging")
    print()

    print("Actual results:")
    print("  Raw components: {}".format(len(raw_components)))
    print("  After merging:  {}".format(len(rectangles)))
    print()

    if len(rectangles) == 1:
        print("✗ PROBLEM: Everything merged into 1 component")
        print("  This means ALL components are being treated as diacritics")
        print("  or the merging is too aggressive")
    elif len(raw_components) < 7:
        print("✗ PROBLEM: Not enough raw components")
        print("  The word box may be cutting off the ö dots")
        print("  (Expected at least 7-8 raw components for 'Börja')")
    elif len(rectangles) > 5:
        print("✗ PROBLEM: Too many components after merging")
        print("  The ö dots are NOT being merged with the 'o'")
        print("  This causes 'Böörja' (extra ö dot shows separately)")
    else:
        print("✓ Component count looks reasonable")

    print()
    print("VISUALIZATION FILES CREATED:")
    print("-" * 80)
    print("  1. /tmp/borja_components_boxed.png")
    print("     Shows FINAL components (after merging) with colored boxes")
    print()
    print("  2. /tmp/borja_raw_components.png")
    print("     Shows RAW components (before merging) with colored boxes")
    print()
    print("  3. Compare these with output_reflowed.png to see the problem")
    print()
    print("="*80)

    # Also check the "inför" word
    print()
    print("CHECKING 'inför' WORD TOO:")
    print("-" * 80)

    # inför is at approximately (877, 2155, 965, 2193)
    xmin2, ymin2, xmax2, ymax2 = 877, 2155, 965, 2193
    words_list2 = [[xmin2, ymin2, xmax2, ymax2]]
    rectangles2 = find_rects(img, words_list2, debug=False)

    word_img2 = img[ymin2:ymax2, xmin2:xmax2]
    gray2 = cv2.cvtColor(word_img2, cv2.COLOR_BGR2GRAY)
    _, binary2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels2, _, stats2, _ = cv2.connectedComponentsWithStats(binary2, connectivity=8)

    raw_count2 = sum(1 for i in range(1, num_labels2)
                     if stats2[i, cv2.CC_STAT_WIDTH] >= 2 and stats2[i, cv2.CC_STAT_HEIGHT] >= 2)

    print("inför: {} raw components → {} after merging".format(raw_count2, len(rectangles2)))

    if len(rectangles2) > 5:
        print("  ✗ Similar problem: ö dots not merging correctly")

    print()
    print("="*80)

if __name__ == '__main__':
    visualize_borja_problem()
