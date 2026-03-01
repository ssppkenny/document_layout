#!/usr/bin/env python3
"""
Create a visual map of all words so user can identify which have Swedish diacritics.
"""

import cv2
import sys
import numpy as np

sys.path.insert(0, 'src/ocr_reflow')
from doctr.models import detection_predictor
import torch

def main():
    img_path = 'images/gang_p023_lines1.png'
    img = cv2.imread(img_path)

    if img is None:
        print(f"ERROR: Could not read {img_path}")
        return

    # Get word detection
    model = detection_predictor(arch='db_resnet50', pretrained=True, assume_straight_pages=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    result = model([img])

    words = result[0]['words']
    h, w = img.shape[:2]

    # Create visualization
    vis = img.copy()

    print(f"Creating word map for {len(words)} words...")
    print()

    for i, word in enumerate(words):
        xmin = int(word[0] * w)
        ymin = int(word[1] * h)
        xmax = int(word[2] * w)
        ymax = int(word[3] * h)

        # Draw bounding box
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw label with better visibility
        label = f"W{i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label above the box with padding
        label_y = max(ymin - 10, text_h + 5)  # Place above box, but not off-screen
        label_x = xmin

        # Draw white background for better contrast
        padding = 3
        cv2.rectangle(vis,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + padding),
                     (255, 255, 255), -1)

        # Draw black border around label background
        cv2.rectangle(vis,
                     (label_x - padding, label_y - text_h - padding),
                     (label_x + text_w + padding, label_y + padding),
                     (0, 0, 0), 1)

        # Draw text in blue for visibility
        cv2.putText(vis, label, (label_x, label_y), font, font_scale, (255, 0, 0), thickness)

    # Save visualization
    output_path = '/tmp/gang_word_map.png'
    cv2.imwrite(output_path, vis)
    print(f"✓ Saved word map to {output_path}")
    print()
    print("Please open the image and tell me the word numbers (W#) for words containing:")
    print("  - ö (o with two dots)")
    print("  - å (a with ring)")
    print("  - ä (a with two dots)")
    print()
    print("For example: 'W5, W12, W23 have ö'")

if __name__ == '__main__':
    main()
