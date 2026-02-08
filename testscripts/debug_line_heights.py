#!/usr/bin/env python3
"""
Debug: Check if the varying line spacing correlates with merged i,j letters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
from doctr.io import DocumentFile
from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout
from ocr_reflow.main import get_doctr_model, find_rects
import tempfile

def debug_line_heights(image_path='images/sedg_p598.png'):
    """Debug why line heights vary"""

    print("Debugging line height variation")
    print("=" * 70)

    # Load and process
    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)
    layout_boxes = layout(image_path)
    plain_text_boxes = [b for b, t in layout_boxes if t == 'plain text']

    if len(plain_text_boxes) == 0:
        print("No plain text boxes")
        return

    # Process first box
    box_geom = plain_text_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()
    region_h, region_w = region.shape[:2]

    # Get words
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

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

    # Extract letters
    letters = find_rects(region, words)

    # Calculate letter heights
    letter_heights = [(ly2 - ly1) for lx1, ly1, lx2, ly2 in letters]
    median_height = np.median(letter_heights)

    # Classify letters
    normal = []
    merged = []  # Tall letters (i, j with dots)

    for i, (lx1, ly1, lx2, ly2) in enumerate(letters):
        h = ly2 - ly1
        if h > median_height * 1.2:
            merged.append((i, h))
        else:
            normal.append((i, h))

    print(f"Total letters: {len(letters)}")
    print(f"Normal letters: {len(normal)}")
    print(f"Merged letters (i, j): {len(merged)}")
    print(f"Median height: {median_height:.1f}")

    if len(merged) > 0:
        merged_heights = [h for _, h in merged]
        print(f"\nMerged letter heights:")
        print(f"  Min: {min(merged_heights)}")
        print(f"  Max: {max(merged_heights)}")
        print(f"  Mean: {np.mean(merged_heights):.1f}")
        print(f"  Increase over median: {(np.mean(merged_heights) / median_height - 1) * 100:.1f}%")

    if len(normal) > 0:
        normal_heights = [h for _, h in normal]
        print(f"\nNormal letter heights:")
        print(f"  Min: {min(normal_heights)}")
        print(f"  Max: {max(normal_heights)}")
        print(f"  Mean: {np.mean(normal_heights):.1f}")
        print(f"  Std: {np.std(normal_heights):.1f}")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("If merged letters are significantly taller (~30-50% taller),")
    print("lines with many merged letters will have more space above them.")
    print("This causes varying line spacing if we use per-line max height.")
    print("\nSOLUTION: Use constant line height per paragraph (already implemented)")
    print("=" * 70)

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'images/sedg_p598.png'
    debug_line_heights(image_path)
