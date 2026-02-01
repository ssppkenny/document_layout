#!/usr/bin/env python3
"""
Test different threshold values across all test images
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import sys
sys.path.insert(0, 'src')
from ocr_reflow.main import margins

# Test images with expected line counts
test_cases = [
    ('notebooks/out0.png', 12),
    ('images/kf_16_par.png', 7),
    ('images/out2.png', 7),
    ('notebooks/out3.png', 5)
]

# Test different threshold values
test_thresholds = [0.30, 0.35, 0.38, 0.40, 0.42, 0.45]

print("=" * 80)
print("TESTING DIFFERENT THRESHOLDS")
print("=" * 80)

for threshold_factor in test_thresholds:
    print(f"\n{'='*80}")
    print(f"THRESHOLD FACTOR: {threshold_factor}")
    print(f"{'='*80}")

    all_pass = True

    for img_path, expected_lines in test_cases:
        # Load and process image
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        model = detection_predictor(pretrained=True)
        docs = DocumentFile.from_images([img_path])
        result = model(docs)
        words = result[0]['words']

        # Convert coordinates
        words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
        words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
        words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
        words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
        words = words.astype(np.int32)

        word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
        median_height = np.median(word_heights)

        # Modified margins function to use custom threshold
        height_threshold = median_height * 0.70
        filtered_words = []
        for xmin, ymin, xmax, ymax, conf in words:
            word_height = ymax - ymin
            if word_height >= height_threshold:
                filtered_words.append((xmin, ymin, xmax, ymax, conf))

        if len(filtered_words) < 2:
            continue

        word_data = []
        for xmin, ymin, xmax, ymax, conf in filtered_words:
            center_y = (ymin + ymax) / 2
            word_data.append({
                'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                'center_y': center_y, 'height': ymax - ymin
            })

        word_data.sort(key=lambda w: w['center_y'])

        lines = []
        current_line = [word_data[0]]
        gap_threshold = median_height * threshold_factor

        for i in range(1, len(word_data)):
            prev_word = word_data[i-1]
            curr_word = word_data[i]
            y_gap = curr_word['center_y'] - prev_word['center_y']

            if y_gap > gap_threshold:
                lines.append(current_line)
                current_line = [curr_word]
            else:
                current_line.append(curr_word)

        if current_line:
            lines.append(current_line)

        detected_lines = len(lines)
        status = "✓" if detected_lines == expected_lines else "✗"
        if detected_lines != expected_lines:
            all_pass = False

        print(f"  {status} {img_path}: detected {detected_lines} lines (expected {expected_lines})")

    if all_pass:
        print(f"\n  ✓✓✓ THRESHOLD {threshold_factor} WORKS FOR ALL TEST CASES! ✓✓✓")
