#!/usr/bin/env python3
"""
Find optimal thresholds for all test images
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile

# Test images with expected line counts
test_cases = [
    ('notebooks/out0.png', 12),
    ('images/kf_16_par.png', 7),
    ('images/out2.png', 7),
    ('notebooks/out3.png', 5)
]

print("=" * 80)
print("FINDING OPTIMAL THRESHOLDS FOR EACH IMAGE")
print("=" * 80)

for img_path, expected_lines in test_cases:
    print(f"\n{img_path} (expected: {expected_lines} lines)")
    print("-" * 80)

    # Load and process image
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ✗ Could not load image")
        continue

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

    word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
    median_height = np.median(word_heights)

    print(f"  Median word height: {median_height:.1f}px")

    # Get center Y for each word
    word_centers = []
    for i, (xmin, ymin, xmax, ymax, conf) in enumerate(words):
        center_y = (ymin + ymax) / 2
        word_centers.append(center_y)

    word_centers.sort()

    # Find threshold that gives expected lines
    found = False
    for threshold_factor in np.arange(0.2, 0.6, 0.005):
        gap_threshold = median_height * threshold_factor

        # Cluster into lines
        lines_count = 1
        for i in range(1, len(word_centers)):
            if word_centers[i] - word_centers[i-1] > gap_threshold:
                lines_count += 1

        if lines_count == expected_lines:
            print(f"  ✓ Threshold factor: {threshold_factor:.3f} ({gap_threshold:.1f}px)")
            found = True
            break

    if not found:
        print(f"  ✗ Could not find exact threshold for {expected_lines} lines")

        # Show what current 0.42 gives
        gap_threshold = median_height * 0.42
        lines_count = 1
        for i in range(1, len(word_centers)):
            if word_centers[i] - word_centers[i-1] > gap_threshold:
                lines_count += 1
        print(f"  Current threshold (0.42 = {gap_threshold:.1f}px): {lines_count} lines")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("Different images require different thresholds:")
print("  - out0.png: 0.41 (12 lines)")
print("  - kf_16_par.png: needs checking")
print("  - out2.png: needs checking")
print("  - out3.png: 0.29 (5 lines)")
print("\nSolution: Need a more adaptive approach or accept a compromise threshold")
