#!/usr/bin/env python3
"""
Analyze characteristics of test images to find a general threshold calculation rule
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
print("ANALYZING IMAGE CHARACTERISTICS")
print("=" * 80)

all_data = []

for img_path, expected_lines in test_cases:
    print(f"\n{img_path} (expected: {expected_lines} lines)")
    print("-" * 80)

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

    # Calculate statistics
    word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
    median_height = np.median(word_heights)
    mean_height = np.mean(word_heights)
    std_height = np.std(word_heights)

    # Get Y-center gaps between consecutive words
    word_centers = []
    for xmin, ymin, xmax, ymax, conf in words:
        center_y = (ymin + ymax) / 2
        word_centers.append(center_y)

    word_centers.sort()

    gaps = []
    for i in range(1, len(word_centers)):
        gap = word_centers[i] - word_centers[i-1]
        gaps.append(gap)

    gaps.sort()

    # Find the gap that separates lines (look for a jump in gap sizes)
    # This is the minimum gap between lines
    min_interline_gap = None
    max_intraline_gap = None

    # Cluster gaps into small (within line) and large (between lines)
    if len(gaps) > expected_lines:
        # The first (n_words - n_lines) gaps are within lines
        # The last (n_lines - 1) gaps are between lines
        n_words = len(word_centers)
        n_intraline_gaps = n_words - expected_lines
        n_interline_gaps = expected_lines - 1

        if n_intraline_gaps > 0:
            max_intraline_gap = gaps[n_intraline_gaps - 1]
        if n_interline_gaps > 0:
            min_interline_gap = gaps[n_intraline_gaps]

    # Find optimal threshold
    optimal_threshold = None
    for threshold_factor in np.arange(0.2, 0.6, 0.005):
        gap_threshold = median_height * threshold_factor
        lines_count = 1
        for gap in gaps:
            if gap > gap_threshold:
                lines_count += 1
        if lines_count == expected_lines:
            optimal_threshold = threshold_factor
            break

    print(f"  Words: {len(words)}")
    print(f"  Median height: {median_height:.1f}px")
    print(f"  Mean height: {mean_height:.1f}px")
    print(f"  Std height: {std_height:.1f}px")
    print(f"  Total gaps: {len(gaps)}")

    if max_intraline_gap is not None:
        print(f"  Max intra-line gap: {max_intraline_gap:.1f}px ({max_intraline_gap/median_height:.3f} × median)")
    if min_interline_gap is not None:
        print(f"  Min inter-line gap: {min_interline_gap:.1f}px ({min_interline_gap/median_height:.3f} × median)")

    if optimal_threshold:
        print(f"  Optimal threshold factor: {optimal_threshold:.3f}")
        print(f"  Optimal threshold: {median_height * optimal_threshold:.1f}px")

    # Calculate ratio of heights to gaps
    if min_interline_gap and max_intraline_gap:
        separation_ratio = min_interline_gap / max_intraline_gap
        print(f"  Separation ratio (inter/intra): {separation_ratio:.2f}")

    all_data.append({
        'path': img_path,
        'expected_lines': expected_lines,
        'median_height': median_height,
        'mean_height': mean_height,
        'std_height': std_height,
        'max_intraline_gap': max_intraline_gap,
        'min_interline_gap': min_interline_gap,
        'optimal_threshold': optimal_threshold,
        'gaps': gaps
    })

print("\n" + "=" * 80)
print("LOOKING FOR PATTERNS")
print("=" * 80)

# Analyze if there's a pattern
print("\nOptimal thresholds as ratio of median height:")
for data in all_data:
    if data['optimal_threshold']:
        print(f"  {data['path']}: {data['optimal_threshold']:.3f}")

# Try to find a relationship based on gap analysis
print("\nLooking for adaptive threshold based on gap distribution...")

for data in all_data:
    gaps = data['gaps']
    median_height = data['median_height']

    # Calculate percentiles of gaps
    if len(gaps) > 10:
        p25 = np.percentile(gaps, 25)
        p50 = np.percentile(gaps, 50)
        p75 = np.percentile(gaps, 75)
        p90 = np.percentile(gaps, 90)

        print(f"\n{data['path']}:")
        print(f"  Gap p25: {p25:.1f}px ({p25/median_height:.3f} × median)")
        print(f"  Gap p50: {p50:.1f}px ({p50/median_height:.3f} × median)")
        print(f"  Gap p75: {p75:.1f}px ({p75/median_height:.3f} × median)")
        print(f"  Gap p90: {p90:.1f}px ({p90/median_height:.3f} × median)")

        # The threshold should be between p75 and p90 for most cases
        suggested_threshold = (p75 + p90) / 2
        print(f"  Suggested threshold: {suggested_threshold:.1f}px ({suggested_threshold/median_height:.3f} × median)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("We need an adaptive threshold based on actual gap distribution!")
print("Options:")
print("  1. Use percentile of gaps (e.g., p75 or between p75-p90)")
print("  2. Use clustering to separate small vs large gaps")
print("  3. Use the 'elbow' in gap distribution")
