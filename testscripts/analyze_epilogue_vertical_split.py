#!/usr/bin/env python3
"""
Analyze the SECOND title block (Epilogue) specifically to see vertical letter splitting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout
from ocr_reflow.main import get_doctr_model, find_rects
import tempfile

def analyze_second_title_block(image_path='images/jtg_p033.png'):
    """Analyze the Epilogue title block specifically"""

    print("=" * 70)
    print("ANALYZING SECOND TITLE BLOCK (Epilogue)")
    print("=" * 70)

    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)

    # Get ALL title boxes
    layout_boxes = layout(image_path)
    title_boxes = [(b, t) for b, t in layout_boxes if t == 'title']

    print(f"Found {len(title_boxes)} title boxes")

    # Sort by Y coordinate to get them in order
    title_boxes_sorted = sorted(title_boxes, key=lambda item: item[0].bounds[1])

    # Show all title boxes
    for i, (box_geom, box_type) in enumerate(title_boxes_sorted):
        bounds = box_geom.bounds
        print(f"  Title {i}: y={bounds[1]:.0f}, bounds=({bounds[0]:.0f}, {bounds[1]:.0f}, {bounds[2]:.0f}, {bounds[3]:.0f})")

    if len(title_boxes_sorted) < 2:
        print("ERROR: Expected 2 title boxes, found only", len(title_boxes_sorted))
        return

    # Process title with HIGHER Y coordinate (second/lower one on page = Epilogue)
    print("\n" + "=" * 70)
    print("PROCESSING TITLE WITH HIGHER Y (should be 'Epilogue')")
    print("=" * 70)

    box_geom, box_type = title_boxes_sorted[-1]  # Last = highest Y = lowest on page
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()
    region_h, region_w = region.shape[:2]

    print(f"Title region: {region_w}x{region_h} at position ({xmin}, {ymin})")

    # Get words
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        padding = 15  # Title padding
        words = []
        for i in range(len(words_array)):
            wx1 = int(words_array[i, 0] * region_w) - padding
            wy1 = int(words_array[i, 1] * region_h) - padding
            wx2 = int(words_array[i, 2] * region_w) + padding
            wy2 = int(words_array[i, 3] * region_h) + padding
            wx1 = max(0, wx1)
            wy1 = max(0, wy1)
            wx2 = min(region_w, wx2)
            wy2 = min(region_h, wy2)
            words.append((wx1, wy1, wx2, wy2))
    finally:
        import os
        os.unlink(tmp_path)

    print(f"Detected {len(words)} word(s) in Epilogue title")

    # Extract letters
    letters = find_rects(region, words)

    print(f"\nExtracted {len(letters)} letter components")
    print(f"Expected: 8 for 'Epilogue' (E-p-i-l-o-g-u-e)")

    if len(letters) > 12:
        print(f"\n⚠️  SEVERE OVER-SEGMENTATION!")
        print(f"   {len(letters)} components detected, expected 8")

    # Analyze in detail
    letter_details = []
    for lx1, ly1, lx2, ly2 in letters:
        w = lx2 - lx1
        h = ly2 - ly1
        letter_details.append({
            'x': lx1, 'y': ly1, 'w': w, 'h': h,
            'area': w * h,
            'aspect': h / w if w > 0 else 0
        })

    if letter_details:
        widths = [d['w'] for d in letter_details]
        heights = [d['h'] for d in letter_details]
        aspects = [d['aspect'] for d in letter_details]

        median_width = np.median(widths)
        median_height = np.median(heights)

        print(f"\nLetter statistics:")
        print(f"  Widths:  min={min(widths):3d}, max={max(widths):3d}, median={median_width:.1f}")
        print(f"  Heights: min={min(heights):3d}, max={max(heights):3d}, median={median_height:.1f}")
        print(f"  Aspect ratio (h/w): min={min(aspects):.2f}, max={max(aspects):.2f}, median={np.median(aspects):.2f}")

        # Check for vertically split letters
        # If a letter is split vertically, we'd see multiple narrow components with similar Y positions
        narrow_threshold = median_width * 0.5
        narrow_components = [i for i, d in enumerate(letter_details) if d['w'] < narrow_threshold]

        print(f"\nNarrow components (width < 50% median): {len(narrow_components)}")
        if narrow_components:
            print(f"  These could be vertically-split letter parts:")
            for i in narrow_components:
                d = letter_details[i]
                print(f"    Component {i}: x={d['x']:3d}, y={d['y']:3d}, w={d['w']:3d}, h={d['h']:3d}")

        # Check for vertically aligned narrow components (strong evidence of vertical splitting)
        vertical_splits = []
        for i in range(len(letter_details)):
            if letter_details[i]['w'] < narrow_threshold:
                # Look for other narrow components at similar Y position
                for j in range(i+1, len(letter_details)):
                    if letter_details[j]['w'] < narrow_threshold:
                        y_diff = abs(letter_details[i]['y'] - letter_details[j]['y'])
                        x_gap = abs(letter_details[i]['x'] - letter_details[j]['x'])
                        if y_diff < 10 and 5 < x_gap < median_width * 1.5:
                            vertical_splits.append((i, j))

        if vertical_splits:
            print(f"\n⚠️  VERTICAL SPLIT DETECTED!")
            print(f"   Found {len(vertical_splits)} pairs of narrow components at same Y position:")
            for i, j in vertical_splits:
                print(f"     Components {i} and {j} likely parts of same letter split vertically")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original region
    axes[0, 0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Epilogue Title Region')
    axes[0, 0].axis('off')

    # With word boxes
    vis1 = region.copy()
    for i, (wx1, wy1, wx2, wy2) in enumerate(words):
        cv2.rectangle(vis1, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
        cv2.putText(vis1, f"Word{i}", (wx1+5, wy1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    axes[0, 1].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Word Detection: {len(words)} word(s)')
    axes[0, 1].axis('off')

    # With letter boxes
    vis2 = region.copy()
    for i, (lx1, ly1, lx2, ly2) in enumerate(letters):
        w = lx2 - lx1
        # Color: green=normal width, red=narrow (likely split)
        color = (0, 255, 0) if w >= narrow_threshold else (0, 0, 255)
        cv2.rectangle(vis2, (lx1, ly1), (lx2, ly2), color, 2)
        cv2.putText(vis2, str(i), (lx1+2, ly1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    axes[1, 0].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Letters: {len(letters)} components (Red=narrow/split)')
    axes[1, 0].axis('off')

    # Width histogram
    if letter_details:
        axes[1, 1].bar(range(len(widths)), widths,
                      color=['green' if w >= narrow_threshold else 'red' for w in widths])
        axes[1, 1].axhline(y=median_width, color='blue', linestyle='--', label=f'Median: {median_width:.1f}')
        axes[1, 1].axhline(y=narrow_threshold, color='red', linestyle='--', label=f'Narrow threshold: {narrow_threshold:.1f}')
        axes[1, 1].set_xlabel('Component index')
        axes[1, 1].set_ylabel('Width (pixels)')
        axes[1, 1].set_title('Component Widths')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/epilogue_title_vertical_split_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to notebooks/epilogue_title_vertical_split_analysis.png")
    plt.close()

    print("\n" + "=" * 70)
    return letter_details

if __name__ == '__main__':
    analyze_second_title_block()
