#!/usr/bin/env python3
"""
Check if letters with descenders (g, p, y) have correct baseline values
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout
from ocr_reflow.main import get_doctr_model, find_rects
from doctr.io import DocumentFile
import tempfile

def check_baseline_values(image_path='images/jtg_p033.png'):
    """Check baseline values for letters in title"""

    print("Checking baseline values for title letters")
    print("=" * 70)

    img = cv2.imread(image_path)
    deskewed_img, _ = detect_and_correct_skew(img)

    layout_boxes = layout(image_path)
    title_boxes = [b for b, t in layout_boxes if t == 'title']

    if len(title_boxes) == 0:
        print("No title boxes found!")
        return

    print(f"Found {len(title_boxes)} title boxes")

    # Process first title box
    box_geom = title_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()

    # Get words
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        region_h, region_w = region.shape[:2]
        words = []
        for i in range(len(words_array)):
            words.append((
                int(words_array[i, 0] * region_w),
                int(words_array[i, 1] * region_h),
                int(words_array[i, 2] * region_w),
                int(words_array[i, 3] * region_h)
            ))
    finally:
        import os
        os.unlink(tmp_path)

    print(f"Words: {len(words)}")

    # The problem is that find_rects returns tuples, not Letter objects
    # We need to examine the actual Letter objects created in reflow

    # For now, just check the letter extraction
    letters = find_rects(region, words)
    print(f"Letters: {len(letters)}")

    heights = [(ly2 - ly1) for lx1, ly1, lx2, ly2 in letters]
    print(f"\nLetter heights:")
    print(f"  Min: {min(heights)}")
    print(f"  Max: {max(heights)}")
    print(f"  Mean: {np.mean(heights):.1f}")
    print(f"  Std: {np.std(heights):.1f}")

    # Show first 10 letters
    print(f"\nFirst 10 letters (x, y, w, h):")
    for i, (lx1, ly1, lx2, ly2) in enumerate(letters[:10]):
        w = lx2 - lx1
        h = ly2 - ly1
        print(f"  Letter {i}: x={lx1}, y={ly1}, w={w}, h={h}")

    print("\nNote: find_rects returns raw coordinates, not Letter objects")
    print("The baseline (bl) is calculated later in the reflow process")
    print("=" * 70)

if __name__ == '__main__':
    check_baseline_values()
