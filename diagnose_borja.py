#!/usr/bin/env python3
"""
Diagnostic script to analyze the "Börja" → "Böörja" issue
"""

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import detection_predictor
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def main():
    print("="*80)
    print("DIAGNOSTIC REPORT: Börja Word Analysis")
    print("="*80)
    print()

    img = cv2.imread('images/gang_p023.png')
    img_h, img_w = img.shape[:2]

    # Get all word detections
    det_predictor = detection_predictor(arch='db_resnet50', pretrained=True)
    docs = DocumentFile.from_images(['images/gang_p023.png'])
    result = det_predictor(docs)
    words_array = result[0]['words']

    # Find the line with "Börja med att sova"
    target_y = 2150
    words_in_line = []

    for i, word_box in enumerate(words_array):
        xmin = int(word_box[0] * img_w)
        ymin = int(word_box[1] * img_h)
        xmax = int(word_box[2] * img_w)
        ymax = int(word_box[3] * img_h)

        if 2100 < ymin < 2250:
            words_in_line.append((i, xmin, ymin, xmax, ymax))

    words_in_line.sort(key=lambda w: w[1])

    print("Step 1: Word Detection by doctr")
    print("-" * 80)
    print("Expected line text: 'Börja med att sova.'")
    print()
    print("Words detected (left to right):")
    for idx, (wid, xmin, ymin, xmax, ymax) in enumerate(words_in_line[:5]):
        width = xmax - xmin
        print("  Word {}: box=({:3d},{:4d})→({:3d},{:4d}) width={}px".format(
            idx+1, xmin, ymin, xmax, ymax, width))

    print()
    print("Step 2: Component Extraction for Word 1 (should be 'Börja')")
    print("-" * 80)

    # Test first word
    if words_in_line:
        wid, xmin, ymin, xmax, ymax = words_in_line[0]

        print("Word box: ({},{}) → ({},{})".format(xmin, ymin, xmax, ymax))
        print()

        # Extract components using find_rects
        words_list = [[xmin, ymin, xmax, ymax]]
        rectangles = find_rects(img, words_list, debug=False)

        print("Components extracted: {}".format(len(rectangles)))
        print()

        # Visualize the word
        word_img = img[ymin:ymax, xmin:xmax]
        cv2.imwrite('/tmp/borja_word_box.png', word_img)

        # Check if box contains full word
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        raw_components = 0
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w >= 2 and h >= 2:
                raw_components += 1

        print("Raw connected components in word box: {}".format(raw_components))
        print()

        print("Step 3: Analysis")
        print("-" * 80)
        print()

        expected_components = 5  # B, ö (with dots merged), r, j (with dot), a

        if len(rectangles) == expected_components:
            print("✓ CORRECT: Got {} components as expected".format(expected_components))
            print("  This suggests 'Börja' should display correctly")
        elif len(rectangles) > expected_components:
            print("✗ TOO MANY: Got {} components (expected ~{})".format(
                len(rectangles), expected_components))
            print("  This suggests ö dots are NOT being merged → 'Böörja'")
        else:
            print("✗ TOO FEW: Got {} components (expected ~{})".format(
                len(rectangles), expected_components))
            print("  This suggests over-merging or missing components")

        print()
        print("Raw components: {}".format(raw_components))
        if raw_components < 7:
            print("  → WARNING: Word box may be cutting off diacritics")
            print("  → ö dots might be outside the doctr word bounding box")

        print()
        print("Saved word visualization to: /tmp/borja_word_box.png")

    print()
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    print()
    print("1. Check output_reflowed.png to see if 'Börja' displays as 'Böörja'")
    print("2. Check /tmp/borja_word_box.png to see if ö dots are in the box")
    print("3. Report back: Does the reflowed output show 'Böörja' or 'Börja'?")
    print()

if __name__ == '__main__':
    main()
