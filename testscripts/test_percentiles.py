#!/usr/bin/env python3
"""
Test different percentiles for adaptive threshold
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile

test_cases = [
    ('notebooks/out0.png', 12),
    ('images/kf_16_par.png', 7),
    ('images/out2.png', 7),
    ('notebooks/out3.png', 5)
]

print("=" * 80)
print("TESTING DIFFERENT PERCENTILES FOR ADAPTIVE THRESHOLD")
print("=" * 80)

for percentile in [75, 78, 80, 82, 85, 87, 90]:
    print(f"\n{'='*80}")
    print(f"USING PERCENTILE: p{percentile}")
    print(f"{'='*80}")

    all_pass = True

    for img_path, expected_lines in test_cases:
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        model = detection_predictor(pretrained=True)
        docs = DocumentFile.from_images([img_path])
        result = model(docs)
        words = result[0]['words']

        words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
        words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
        words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
        words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

        word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
        median_height = np.median(word_heights)
        height_threshold = median_height * 0.70

        filtered_words = []
        for xmin, ymin, xmax, ymax, conf in words:
            if (ymax - ymin) >= height_threshold:
                filtered_words.append((xmin, ymin, xmax, ymax, conf))

        if len(filtered_words) < 2:
            continue

        word_data = []
        for xmin, ymin, xmax, ymax, conf in filtered_words:
            center_y = (ymin + ymax) / 2
            word_data.append({'center_y': center_y})

        word_data.sort(key=lambda w: w['center_y'])

        # Calculate gaps
        gaps = [word_data[i]['center_y'] - word_data[i-1]['center_y']
                for i in range(1, len(word_data))]

        # Calculate threshold
        gap_threshold = np.percentile(gaps, percentile)
        min_threshold = median_height * 0.20
        max_threshold = median_height * 0.60
        gap_threshold = max(min_threshold, min(max_threshold, gap_threshold))

        # Count lines
        lines_count = 1
        for gap in gaps:
            if gap > gap_threshold:
                lines_count += 1

        status = "✓" if lines_count == expected_lines else "✗"
        if lines_count != expected_lines:
            all_pass = False

        print(f"  {status} {img_path}: {lines_count} lines (expected {expected_lines}) [threshold: {gap_threshold:.1f}px]")

    if all_pass:
        print(f"\n  ✓✓✓ PERCENTILE p{percentile} WORKS FOR ALL TEST CASES! ✓✓✓")
