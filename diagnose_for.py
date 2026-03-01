#!/usr/bin/env python3
"""
Diagnose why 'för' still has split ö when other words are fixed.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')

from binarization import binarize_document, normalize_image
from doctr.models import detection_predictor
import torch

def analyze_for():
    # Read and binarize the image
    img = cv2.imread('images/gang_p023_lines1.png')
    normalized = normalize_image(img)
    binarized = binarize_document(normalized, method='otsu')
    binarized_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    # Get word detection
    model = detection_predictor(arch='db_resnet50', pretrained=True, assume_straight_pages=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    result = model([binarized_bgr])

    words = result[0]['words']
    h, w = binarized_bgr.shape[:2]

    print("Analyzing W48 (Börja) and W19 (rök)...")
    print(f"Total words detected: {len(words)}")

    # Analyze specific words
    words_to_check = [18, 47]  # W19 and W48 (0-indexed)
    word_names = ["W19 (rök)", "W48 (Börja)"]

    for word_idx, word_name in zip(words_to_check, word_names):
        if word_idx >= len(words):
            print(f"\n{word_name}: Index out of range")
            continue
        word = words[word_idx]
        xmin = int(word[0] * w)
        ymin = int(word[1] * h)
        xmax = int(word[2] * w)
        ymax = int(word[3] * h)

        word_w = xmax - xmin
        word_h = ymax - ymin

        print(f"\n{'='*80}")
        print(f"{word_name}: ({xmin},{ymin})-({xmax},{ymax}) size={word_w}x{word_h}")
        print(f"{'='*80}")

        # Extract word region from binarized image
        word_img = binarized[ymin:ymax, xmin:xmax].copy()

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(word_img), 8, cv2.CV_32S)

        print(f"  Connected components: {num_labels-1}")

        # Analyze components
        component_info = []
            for j in range(1, num_labels):
                x = stats[j, cv2.CC_STAT_LEFT]
                y = stats[j, cv2.CC_STAT_TOP]
                w_comp = stats[j, cv2.CC_STAT_WIDTH]
                h_comp = stats[j, cv2.CC_STAT_HEIGHT]
                component_info.append({'id': j, 'x': x, 'y': y, 'w': w_comp, 'h': h_comp})

            component_info.sort(key=lambda c: c['x'])

            heights = [c['h'] for c in component_info]
            median_height = np.median(heights)

            print(f"\n  Components (sorted left to right), median_height={median_height:.1f}:")
            for comp in component_info:
                # Classify as diacritic or letter
                is_diacritic = (comp['h'] < median_height * 0.5 and
                               comp['w'] < median_height * 1.0 and
                               comp['w'] * comp['h'] < (median_height ** 2) * 0.4)

                comp_type = "DIAC" if is_diacritic else "LETT"
                print(f"    #{comp['id']} [{comp_type}]: x={comp['x']:3d} y={comp['y']:3d} "
                      f"w={comp['w']:2d} h={comp['h']:2d}")

            # Check if would merge with current thresholds
            letter_comps = [c for c in component_info if not (
                c['h'] < median_height * 0.5 and
                c['w'] < median_height * 1.0 and
                c['w'] * c['h'] < (median_height ** 2) * 0.4)]

            if len(letter_comps) >= 2:
                print(f"\n  Checking merge conditions for letter components:")
                for idx_i in range(len(letter_comps)):
                    for idx_j in range(idx_i + 1, len(letter_comps)):
                        c_i = letter_comps[idx_i]
                        c_j = letter_comps[idx_j]

                        h_gap = max(0, max(c_i['x'] - (c_j['x'] + c_j['w']),
                                          c_j['x'] - (c_i['x'] + c_i['w'])))
                        v_overlap = min(c_i['y'] + c_i['h'], c_j['y'] + c_j['h']) - max(c_i['y'], c_j['y'])
                        y_center_i = c_i['y'] + c_i['h'] / 2
                        y_center_j = c_j['y'] + c_j['h'] / 2
                        v_center_dist = abs(y_center_i - y_center_j)

                        min_h = min(c_i['h'], c_j['h'])
                        height_ratio = max(c_i['h'], c_j['h']) / max(min_h, 1)

                        # Current thresholds
                        gap_threshold = median_height * 0.4
                        center_threshold = median_height * 0.5
                        overlap_threshold = min_h * 0.3
                        height_ratio_threshold = 1.4

                        should_merge = (h_gap < gap_threshold and
                                       v_center_dist < center_threshold and
                                       v_overlap > overlap_threshold and
                                       height_ratio < height_ratio_threshold)

                        print(f"    Comp #{c_i['id']} vs #{c_j['id']}:")
                        print(f"      h_gap={h_gap:.1f} < {gap_threshold:.1f}? {h_gap < gap_threshold}")
                        print(f"      v_center_dist={v_center_dist:.1f} < {center_threshold:.1f}? {v_center_dist < center_threshold}")
                        print(f"      v_overlap={v_overlap} > {overlap_threshold:.1f}? {v_overlap > overlap_threshold}")
                        print(f"      height_ratio={height_ratio:.2f} < {height_ratio_threshold}? {height_ratio < height_ratio_threshold}")
                        print(f"      → MERGE: {should_merge}")

if __name__ == '__main__':
    analyze_for()
