#!/usr/bin/env python3
"""
Visualize all detected words with labels for manual identification.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')

from binarization import binarize_document, normalize_image
from doctr.models import detection_predictor
import torch

def visualize_words():
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

    # Create visualization
    vis_img = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    for i, word in enumerate(words):
        xmin = int(word[0] * w)
        ymin = int(word[1] * h)
        xmax = int(word[2] * w)
        ymax = int(word[3] * h)

        # Draw rectangle around word
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Add word number label INSIDE the rectangle
        label = f"W{i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Calculate position inside rectangle (top-left corner with small padding)
        label_x = xmin + 3
        label_y = ymin + label_size[1] + 3

        # Draw background rectangle for label inside the word box
        cv2.rectangle(vis_img, (label_x - 2, label_y - label_size[1] - 2),
                     (label_x + label_size[0] + 2, label_y + 2), (0, 0, 255), -1)
        cv2.putText(vis_img, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save visualization
    output_path = 'word_labels_visualization.png'
    success = cv2.imwrite(output_path, vis_img)
    if success:
        print(f"Visualization saved to: {output_path}")
        print(f"Total words detected: {len(words)}")
        print("\nPlease check the image and tell me the word numbers for:")
        print("  - rök (word with split ö)")
        print("  - Börja (word with split ö)")
    else:
        print(f"ERROR: Failed to save visualization to {output_path}")

if __name__ == '__main__':
    try:
        visualize_words()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
