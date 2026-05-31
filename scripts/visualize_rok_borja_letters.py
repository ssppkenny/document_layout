#!/usr/bin/env python3
"""
Visualize letter segmentation for W19 (rök) and W48 (Börja) to debug split ö issue.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')

from binarization import binarize_document, normalize_image
from doctr.models import detection_predictor
import torch

def visualize_letter_segmentation():
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

    # Process W19 (rök) and W48 (Börja)
    for word_idx, word_name in [(18, "W19 (rök)"), (47, "W48 (Börja)")]:
        word = words[word_idx]
        xmin = int(word[0] * w)
        ymin = int(word[1] * h)
        xmax = int(word[2] * w)
        ymax = int(word[3] * h)

        print(f"\n{'='*80}")
        print(f"Processing {word_name}")
        print(f"{'='*80}")

        # Extract word region
        word_img = binarized[ymin:ymax, xmin:xmax].copy()
        word_h, word_w = word_img.shape

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(word_img), 8, cv2.CV_32S)

        print(f"Total connected components: {num_labels - 1}")

        # Create visualization
        vis_img = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR)

        # Simulate the find_rects logic
        component_info = []
        for j in range(1, num_labels):
            x = stats[j, cv2.CC_STAT_LEFT]
            y = stats[j, cv2.CC_STAT_TOP]
            w_comp = stats[j, cv2.CC_STAT_WIDTH]
            h_comp = stats[j, cv2.CC_STAT_HEIGHT]
            area = stats[j, cv2.CC_STAT_AREA]

            # Filter noise
            if w_comp < 2 or h_comp < 2 or area < 4:
                continue

            component_info.append({
                'id': j, 'x': x, 'y': y, 'w': w_comp, 'h': h_comp
            })

        # Calculate median height
        heights = [c['h'] for c in component_info]
        median_height = np.median(heights) if heights else word_h * 0.5

        # Classify components
        print(f"\nMedian height: {median_height:.1f}")
        print("\nComponents classification:")

        diacritics = []
        letters = []

        for comp in component_info:
            # Use binarization thresholds
            is_diacritic = (comp['h'] < median_height * 0.5 and
                           comp['w'] < median_height * 1.0 and
                           comp['w'] * comp['h'] < (median_height ** 2) * 0.4)

            if is_diacritic:
                diacritics.append(comp)
                color = (255, 0, 0)  # Blue for diacritics
                label = f"D{comp['id']}"
            else:
                letters.append(comp)
                color = (0, 0, 255)  # Red for letters
                label = f"L{comp['id']}"

            # Draw rectangle
            cv2.rectangle(vis_img,
                         (comp['x'], comp['y']),
                         (comp['x'] + comp['w'], comp['y'] + comp['h']),
                         color, 2)

            # Add label
            cv2.putText(vis_img, label,
                       (comp['x'], comp['y'] - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            comp_type = "DIAC" if is_diacritic else "LETT"
            print(f"  #{comp['id']:2d} [{comp_type}]: x={comp['x']:3d} y={comp['y']:3d} "
                  f"w={comp['w']:2d} h={comp['h']:2d}")

        # Check horizontal merging conditions for letters
        print(f"\nLetter components: {len(letters)}")
        if len(letters) >= 2:
            print("Checking horizontal merge conditions:")
            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    c_i = letters[i]
                    c_j = letters[j]

                    h_gap = max(0, max(c_i['x'] - (c_j['x'] + c_j['w']),
                                      c_j['x'] - (c_i['x'] + c_i['w'])))
                    v_overlap = min(c_i['y'] + c_i['h'], c_j['y'] + c_j['h']) - max(c_i['y'], c_j['y'])
                    y_center_i = c_i['y'] + c_i['h'] / 2
                    y_center_j = c_j['y'] + c_j['h'] / 2
                    v_center_dist = abs(y_center_i - y_center_j)

                    min_h = min(c_i['h'], c_j['h'])
                    height_ratio = max(c_i['h'], c_j['h']) / max(min_h, 1)

                    gap_threshold = median_height * 0.4
                    center_threshold = median_height * 0.5
                    overlap_threshold = min_h * 0.3
                    height_ratio_threshold = 1.4

                    should_merge = (h_gap < gap_threshold and
                                   v_center_dist < center_threshold and
                                   v_overlap > overlap_threshold and
                                   height_ratio < height_ratio_threshold)

                    status = "✓ MERGE" if should_merge else "✗ NO"
                    print(f"\n  L{c_i['id']} vs L{c_j['id']}: {status}")
                    print(f"    h_gap={h_gap:.1f} < {gap_threshold:.1f}? {h_gap < gap_threshold}")
                    print(f"    v_center={v_center_dist:.1f} < {center_threshold:.1f}? {v_center_dist < center_threshold}")
                    print(f"    v_overlap={v_overlap} > {overlap_threshold:.1f}? {v_overlap > overlap_threshold}")
                    print(f"    h_ratio={height_ratio:.2f} < {height_ratio_threshold}? {height_ratio < height_ratio_threshold}")

                    # Draw line between components being checked
                    if should_merge:
                        cx_i = c_i['x'] + c_i['w'] // 2
                        cy_i = c_i['y'] + c_i['h'] // 2
                        cx_j = c_j['x'] + c_j['w'] // 2
                        cy_j = c_j['y'] + c_j['h'] // 2
                        cv2.line(vis_img, (cx_i, cy_i), (cx_j, cy_j), (0, 255, 0), 2)

        # Save visualization
        output_path = f'word_segmentation_{word_name.split()[0]}.png'
        cv2.imwrite(output_path, vis_img)
        print(f"\nVisualization saved to: {output_path}")

if __name__ == '__main__':
    visualize_letter_segmentation()
