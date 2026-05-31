#!/usr/bin/env python3
"""
Diagnose specific Swedish words that are getting extra characters
"""

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import detection_predictor
import sys
import os

sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def analyze_specific_words(image_path, target_words=['Börja', 'inför']):
    """Find and analyze specific Swedish words"""

    print(f"\n{'='*80}")
    print(f"ANALYZING SPECIFIC WORDS: {target_words}")
    print(f"{'='*80}\n")

    # Load image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Get words
    model = detection_predictor(pretrained=True)
    docs = DocumentFile.from_images([image_path])
    result = model(docs)
    words_raw = result[0]["words"]

    # Convert to pixel coordinates with padding
    words = []
    for word in words_raw:
        xmin = int(word[0] * w) - 5
        ymin = int(word[1] * h) - 5
        xmax = int(word[2] * w) + 5
        ymax = int(word[3] * h) + 5

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        words.append([xmin, ymin, xmax, ymax])

    print(f"Total words detected: {len(words)}\n")

    # Analyze all words to find patterns
    print("Analyzing all words for diacritic patterns:")
    print(f"{'='*80}\n")

    problem_words = []

    for idx, (xmin, ymin, xmax, ymax) in enumerate(words[:50]):  # Check first 50 words
        word_height = ymax - ymin
        word_width = xmax - xmin

        # Extract word
        box_img = img[ymin:ymax, xmin:xmax].copy()

        # Get character rectangles
        try:
            rectangles = find_rects(box_img, [[xmin, ymin, xmax, ymax]])

            if len(rectangles) == 0:
                continue

            # Analyze character segmentation
            word_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
            _, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_binary, 8, cv2.CV_32S)

            # Get components
            components = []
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w_comp = stats[i, cv2.CC_STAT_WIDTH]
                h_comp = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                if w_comp >= 2 and h_comp >= 2 and area >= 4:
                    components.append((x, y, w_comp, h_comp))

            # Check for diacritics
            if len(components) > 1:
                component_heights = [h for x, y, w, h in components]
                median_height = np.median(component_heights)

                # Count diacritics
                diacritics = []
                main_letters = []

                for x, y, w_comp, h_comp in components:
                    is_diacritic = (h_comp < median_height * 0.4 and
                                   w_comp < median_height * 0.8 and
                                   w_comp * h_comp < (median_height ** 2) * 0.3 and
                                   h_comp < word_height * 0.25 and
                                   w_comp < word_width * 0.5)

                    if is_diacritic:
                        diacritics.append((x, y, w_comp, h_comp))
                    else:
                        main_letters.append((x, y, w_comp, h_comp))

                # If we have diacritics, analyze
                if len(diacritics) >= 2:
                    print(f"Word {idx+1} at ({xmin}, {ymin}): {word_width}x{word_height}")
                    print(f"  Components: {num_labels-1} total, {len(main_letters)} main, {len(diacritics)} diacritics")
                    print(f"  Segmented into {len(rectangles)} character(s)")
                    print(f"  Median height: {median_height:.1f}px")

                    # Show diacritic positions
                    print(f"  Diacritics:")
                    for dx, dy, dw, dh in diacritics:
                        print(f"    - ({dx}, {dy}) size={dw}x{dh}")

                    # Show main letter positions
                    print(f"  Main letters:")
                    for mx, my, mw, mh in main_letters:
                        print(f"    - ({mx}, {my}) size={mw}x{mh}")

                    # Check if diacritics are being duplicated
                    # Look for overlapping character rectangles
                    rect_overlaps = []
                    for i, (xmin1, xmax1, ymin1, ymax1) in enumerate(rectangles):
                        for j, (xmin2, xmax2, ymin2, ymax2) in enumerate(rectangles):
                            if i >= j:
                                continue

                            # Check overlap
                            overlap_x = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
                            overlap_y = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))

                            if overlap_x > 0 and overlap_y > 0:
                                overlap_area = overlap_x * overlap_y
                                area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                                area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

                                overlap_ratio = overlap_area / min(area1, area2)
                                if overlap_ratio > 0.3:  # Significant overlap
                                    rect_overlaps.append((i, j, overlap_ratio))

                    if rect_overlaps:
                        print(f"  ⚠ WARNING: Found {len(rect_overlaps)} overlapping character rectangles!")
                        for i, j, ratio in rect_overlaps:
                            print(f"    Chars {i} and {j} overlap by {ratio*100:.1f}%")
                        problem_words.append((idx+1, xmin, ymin, xmax, ymax, len(diacritics), len(rectangles)))

                    print()

        except Exception as e:
            pass

    print(f"\n{'='*80}")
    print(f"SUMMARY: Found {len(problem_words)} words with potential issues")
    print(f"{'='*80}\n")

    for word_idx, xmin, ymin, xmax, ymax, num_diac, num_chars in problem_words:
        print(f"Word {word_idx}: ({xmin}, {ymin}) → ({xmax}, {ymax}), {num_diac} diacritics, {num_chars} chars")

if __name__ == '__main__':
    analyze_specific_words('images/gang_p023.png')
