#!/usr/bin/env python3
"""
Visualize the ACTUAL letter rectangles after all merging (horizontal + diacritics)
by extracting them from the binarized image, exactly as reflow will see them.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')

from binarization import binarize_document, normalize_image
from doctr.models import detection_predictor
from main import find_rects
import torch

def visualize_after_all_merging():
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
        word_img = binarized_bgr[ymin:ymax, xmin:xmax].copy()

        # Call find_rects to get the ACTUAL letter rectangles after all merging
        # This simulates what main.py does
        line_words = [(xmin, ymin, xmax, ymax)]
        letter_rects = find_rects(binarized_bgr, line_words, use_binarization=True)

        print(f"find_rects returned {len(letter_rects)} letter rectangles")

        # Create visualization showing what reflow will extract
        vis_img = word_img.copy()

        for idx, (lx1, ly1, lx2, ly2) in enumerate(letter_rects):
            # Convert to word-relative coordinates
            rel_x1 = lx1 - xmin
            rel_y1 = ly1 - ymin
            rel_x2 = lx2 - xmin
            rel_y2 = ly2 - ymin

            # Draw rectangle
            cv2.rectangle(vis_img, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 0, 255), 2)

            # Add label
            label = f"L{idx}"
            cv2.putText(vis_img, label, (rel_x1, rel_y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Extract what reflow will see
            letter_img = binarized[ly1:ly2, lx1:lx2]
            letter_filename = f'letter_extract_{word_name.split()[0]}_L{idx}.png'
            cv2.imwrite(letter_filename, letter_img)

            print(f"  Letter {idx}: ({lx1},{ly1})-({lx2},{ly2}) size={lx2-lx1}x{ly2-ly1}")
            print(f"    Extracted to: {letter_filename}")

        # Save visualization
        output_path = f'after_merge_{word_name.split()[0]}.png'
        cv2.imwrite(output_path, vis_img)
        print(f"\nVisualization saved to: {output_path}")

if __name__ == '__main__':
    visualize_after_all_merging()
