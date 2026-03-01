#!/usr/bin/env python3
"""
Diagnose Swedish diacritics (ä, ö, å) segmentation issues
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
import sys
import os

def analyze_diacritics(image_path):
    """Analyze how Swedish diacritics are being segmented"""

    print(f"\n{'='*80}")
    print(f"ANALYZING SWEDISH DIACRITICS IN: {image_path}")
    print(f"{'='*80}\n")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image {image_path}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Run doctr word detection using DocumentFile (same as main.py)
    print("\nLoading doctr model...")
    from doctr.io import DocumentFile
    model = detection_predictor(pretrained=True)

    print("Running word detection...")
    docs = DocumentFile.from_images([image_path])
    result = model(docs)

    # Extract words (same format as main.py)
    h, w = img.shape[:2]
    words_raw = result[0]["words"]

    # Convert to pixel coordinates with padding (same as main.py)
    words = []
    for word in words_raw:
        xmin = int(word[0] * w) - 5
        ymin = int(word[1] * h) - 5
        xmax = int(word[2] * w) + 5
        ymax = int(word[3] * h) + 5

        # Clamp to image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        words.append([xmin, ymin, xmax, ymax])

    print(f"✓ Detected {len(words)} words")

    # Analyze a few words that likely contain Swedish diacritics
    # Look for words in the middle of the page
    img_vis = img.copy()

    # Sort words by position
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))

    # Analyze first 20 words to find diacritics
    print(f"\n{'='*80}")
    print("ANALYZING WORD SEGMENTATION (first 20 words):")
    print(f"{'='*80}\n")

    for idx, (xmin, ymin, xmax, ymax) in enumerate(words_sorted[:20]):
        word_height = ymax - ymin
        word_width = xmax - xmin

        print(f"\n--- Word {idx+1}: ({xmin}, {ymin}) → ({xmax}, {ymax}), size: {word_width}x{word_height} ---")

        # Extract word region
        word_img = img[ymin:ymax, xmin:xmax].copy()
        word_gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_binary, 8, cv2.CV_32S)

        print(f"  Connected components: {num_labels-1}")

        # Analyze components
        components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if w >= 2 and h >= 2 and area >= 4:
                components.append((x, y, w, h, area))
                print(f"    Component {i}: ({x}, {y}) size={w}x{h}, area={area}, height_ratio={h/word_height:.2f}")

        # Check for potential diacritics (small components above larger ones)
        if len(components) > 1:
            component_heights = [h for x, y, w, h, a in components]
            median_height = np.median(component_heights)

            print(f"  Median component height: {median_height:.1f}px")

            # Find small components (potential diacritics)
            small_comps = [(i, x, y, w, h) for i, (x, y, w, h, a) in enumerate(components)
                          if h < median_height * 0.4]

            large_comps = [(i, x, y, w, h) for i, (x, y, w, h, a) in enumerate(components)
                          if h >= median_height * 0.4]

            if small_comps and large_comps:
                print(f"  → POTENTIAL DIACRITIC PATTERN: {len(small_comps)} small component(s), {len(large_comps)} large component(s)")

                for si, sx, sy, sw, sh in small_comps:
                    print(f"    Small comp: ({sx}, {sy}) size={sw}x{sh}, y_pos={sy/word_height:.2f}")

                    # Check which large components are below
                    for li, lx, ly, lw, lh in large_comps:
                        vertical_gap = ly - (sy + sh)
                        horizontal_overlap = min(sx + sw, lx + lw) - max(sx, lx)

                        if sy < ly and horizontal_overlap > 0:
                            print(f"      Below this: comp at ({lx}, {ly}) size={lw}x{lh}, gap={vertical_gap}px, overlap={horizontal_overlap}px")

        # Draw word box on visualization
        cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img_vis, str(idx+1), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save visualization
    output_path = 'output_swedish_diacritics_analysis.png'
    cv2.imwrite(output_path, img_vis)
    print(f"\n✓ Saved word visualization to: {output_path}")

    # Now check the current diacritic detection logic
    print(f"\n{'='*80}")
    print("CURRENT DIACRITIC DETECTION THRESHOLDS:")
    print(f"{'='*80}")
    print("Checking a sample word with median_height simulation...")

    # Simulate on word 5 if it exists
    if len(words_sorted) > 5:
        xmin, ymin, xmax, ymax = words_sorted[5]
        word_height = ymax - ymin
        word_width = xmax - xmin

        word_img = img[ymin:ymax, xmin:xmax].copy()
        word_gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_binary, 8, cv2.CV_32S)

        components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if w >= 2 and h >= 2:
                components.append((x, y, w, h))

        if components:
            component_heights = [h for x, y, w, h in components]
            median_height = np.median(component_heights)

            print(f"\nWord 6: word_height={word_height}px, word_width={word_width}px, median_comp_height={median_height:.1f}px")
            print(f"\nDiacritic detection criteria (current code):")
            print(f"  - h < median_height * 0.4 = {median_height * 0.4:.1f}px")
            print(f"  - w < median_height * 0.8 = {median_height * 0.8:.1f}px")
            print(f"  - area < median_height² * 0.3 = {median_height**2 * 0.3:.1f}px²")
            print(f"  - h < word_height * 0.25 = {word_height * 0.25:.1f}px")
            print(f"  - w < word_width * 0.5 = {word_width * 0.5:.1f}px")

            print(f"\nComponents classification:")
            for i, (x, y, w, h) in enumerate(components):
                is_diacritic = (h < median_height * 0.4 and
                               w < median_height * 0.8 and
                               w * h < (median_height ** 2) * 0.3 and
                               h < word_height * 0.25 and
                               w < word_width * 0.5)

                print(f"  Component {i}: size={w}x{h}, area={w*h}, → {'DIACRITIC' if is_diacritic else 'MAIN'}")
                print(f"    h={h:.1f} < {median_height*0.4:.1f}? {h < median_height * 0.4}")
                print(f"    w={w:.1f} < {median_height*0.8:.1f}? {w < median_height * 0.8}")
                print(f"    area={w*h:.1f} < {median_height**2*0.3:.1f}? {w*h < (median_height ** 2) * 0.3}")

if __name__ == '__main__':
    image_path = 'images/gang_p023.png'
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    analyze_diacritics(image_path)
