#!/usr/bin/env python3
"""
Test if the horizontal merging is working for split ö letters.
"""

import cv2
import numpy as np

def test_merging():
    """Test the merging logic with actual values from W18 (rök)"""

    # From diagnostic: W18 has 5 components
    # #4 [LETTER]: x=3 y=17 w=16 h=22 (left part of ö)
    # #5 [LETTER]: x=21 y=18 w=21 h=21 (right part of ö)

    components = [
        (3, 17, 16, 22),   # Left part of ö
        (21, 18, 21, 21),  # Right part of ö
    ]

    median_height = 21.0
    word_height = 42

    x_i, y_i, w_i, h_i = components[0]
    x_j, y_j, w_j, h_j = components[1]

    horizontal_gap = max(0, max(x_i - (x_j + w_j), x_j - (x_i + w_i)))
    vertical_overlap = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)

    # NEW: vertical center distance
    y_center_i = y_i + h_i / 2
    y_center_j = y_j + h_j / 2
    vertical_center_distance = abs(y_center_i - y_center_j)

    print("="*80)
    print("Testing horizontal merging for split ö in W18 (rök)")
    print("="*80)
    print(f"\nComponent 1 (left part): x={x_i} y={y_i} w={w_i} h={h_i} center_y={y_center_i:.1f}")
    print(f"Component 2 (right part): x={x_j} y={y_j} w={w_j} h={h_j} center_y={y_center_j:.1f}")
    print(f"\nMedian height: {median_height}")
    print(f"Word height: {word_height}")
    print(f"\nHorizontal gap: {horizontal_gap}")
    print(f"Vertical overlap: {vertical_overlap}")
    print(f"Vertical center distance: {vertical_center_distance:.1f}")

    # Test with NEW binarization-enabled thresholds
    min_height = min(h_i, h_j)
    merge_threshold_gap = median_height * 0.3
    merge_threshold_center = median_height * 0.3
    merge_threshold_overlap = min_height * 0.2

    print(f"\n--- NEW LOGIC WITH BINARIZATION ---")
    print(f"Merge threshold gap: {merge_threshold_gap:.1f} (30% of median_height)")
    print(f"Merge threshold center: {merge_threshold_center:.1f} (30% of median_height)")
    print(f"Merge threshold overlap: {merge_threshold_overlap:.1f} (20% of min_height={min_height})")
    print(f"\nCondition checks:")
    print(f"  horizontal_gap ({horizontal_gap}) < merge_threshold_gap ({merge_threshold_gap:.1f})? {horizontal_gap < merge_threshold_gap}")
    print(f"  vertical_center_distance ({vertical_center_distance:.1f}) < merge_threshold_center ({merge_threshold_center:.1f})? {vertical_center_distance < merge_threshold_center}")
    print(f"  vertical_overlap ({vertical_overlap}) > merge_threshold_overlap ({merge_threshold_overlap:.1f})? {vertical_overlap > merge_threshold_overlap}")

    should_merge_new = (horizontal_gap < merge_threshold_gap and
                       vertical_center_distance < merge_threshold_center and
                       vertical_overlap > merge_threshold_overlap)
    print(f"\n>>> SHOULD MERGE: {should_merge_new}")

    if not should_merge_new:
        print(f"\n{'='*80}")
        print("❌ PROBLEM: Components would NOT be merged even with new logic!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("✓ Components SHOULD be merged with new logic")
        print(f"{'='*80}")

if __name__ == '__main__':
    test_merging()
