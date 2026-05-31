#!/usr/bin/env python3
"""
Find and analyze the word Börja.
"""

import cv2
import numpy as np
import sys

sys.path.insert(0, 'src/ocr_reflow')
from doctr.models import detection_predictor
import torch

def analyze_borja():
    img = cv2.imread('images/gang_p023_lines1.png')

    # Get word detection
    model = detection_predictor(arch='db_resnet50', pretrained=True, assume_straight_pages=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    result = model([img])

    words = result[0]['words']
    h, w = img.shape[:2]

    print(f"Total words: {len(words)}")
    print("\nSearching for Börja (should be 5-6 letters, around 50-70 pixels wide)...")

    # Börja is likely 50-80 pixels wide and around 20-30 pixels tall
    # Look for words with these characteristics
    candidates = []

    for i, word in enumerate(words):
        xmin = int(word[0] * w)
        ymin = int(word[1] * h)
        xmax = int(word[2] * w)
        ymax = int(word[3] * h)

        word_w = xmax - xmin
        word_h = ymax - ymin

        # Börja is roughly 50-80 pixels wide, 25-35 pixels tall
        if 45 < word_w < 85 and 20 < word_h < 40:
            candidates.append((i, xmin, ymin, xmax, ymax, word_w, word_h))

    print(f"\nFound {len(candidates)} candidate words:")
    for i, xmin, ymin, xmax, ymax, word_w, word_h in candidates:
        print(f"  W{i+1}: ({xmin},{ymin})-({xmax},{ymax}) size={word_w}x{word_h}")

    # Analyze each candidate
    for idx, xmin, ymin, xmax, ymax, word_w, word_h in candidates[:5]:  # Check first 5 candidates
        print(f"\n{'='*80}")
        print(f"Analyzing W{idx+1}")
        print(f"{'='*80}")

        # Extract word region
        word_img = img[ymin:ymax, xmin:xmax].copy()

        # Apply binarization
        word_gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_binary, 8, cv2.CV_32S)

        print(f"Connected components: {num_labels-1}")

        # Look for pattern: should have ~5-6 letter components + 2 diacritics
        if num_labels-1 >= 6 and num_labels-1 <= 10:
            print(f"  → This might be Börja (has {num_labels-1} components)")

            # Analyze components
            components_info = []
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w_comp = stats[i, cv2.CC_STAT_WIDTH]
                h_comp = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                components_info.append({
                    'id': i,
                    'x': x,
                    'y': y,
                    'w': w_comp,
                    'h': h_comp,
                    'area': area
                })

            # Sort by x position
            components_info.sort(key=lambda c: c['x'])

            median_height = np.median([c['h'] for c in components_info])

            print(f"\n  Components (sorted left to right):")
            for comp in components_info:
                is_diacritic = (comp['h'] < median_height * 0.5 and
                               comp['w'] < median_height * 1.0 and
                               comp['w'] * comp['h'] < (median_height ** 2) * 0.4)

                comp_type = "DIAC" if is_diacritic else "LETT"
                print(f"    #{comp['id']:2d} [{comp_type}]: x={comp['x']:3d} y={comp['y']:3d} w={comp['w']:2d} h={comp['h']:2d}")

if __name__ == '__main__':
    analyze_borja()
