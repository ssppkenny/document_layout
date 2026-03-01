#!/usr/bin/env python3
"""
Simulate the exact merging logic for W51 för to find the bug.
"""

import numpy as np

# W51 "för" - ALL components from diagnostic
all_components = [
    (1, 5, 6, 17, 35, 180),   # f
    (4, 20, 20, 22, 22, 200), # left part of ö
    (2, 23, 10, 5, 6, 22),    # diacritic dot
    (3, 32, 10, 5, 6, 22),    # diacritic dot
    (5, 44, 20, 16, 21, 129), # r or right part of ö
]

word_height = 48
component_heights = [h for _, x, y, w, h, area in all_components]
median_height = np.median(component_heights)

print("="*80)
print("Simulating W51 (för) classification and merging")
print("="*80)
print(f"\nWord height: {word_height}")
print(f"Median height: {median_height}")

# Step 1: Classify components as diacritic or letter
print("\n--- STEP 1: Classification ---")
dots_to_merge = []
main_letters_to_merge = []

for comp_idx, (id, x, y, w, h, area) in enumerate(all_components):
    # Using binarization thresholds (more lenient)
    is_diacritic = (h < median_height * 0.5 and
                   w < median_height * 1.0 and
                   w * h < (median_height ** 2) * 0.4 and
                   h < word_height * 0.3 and
                   w < word_height * 0.6)

    comp_type = "DIACRITIC" if is_diacritic else "LETTER"
    print(f"Component #{id}: x={x:3d} y={y:3d} w={w:2d} h={h:2d} → {comp_type}")

    if is_diacritic:
        dots_to_merge.append((comp_idx, x, y, w, h))
    else:
        main_letters_to_merge.append((comp_idx, x, y, w, h))

print(f"\nResult: {len(main_letters_to_merge)} letters, {len(dots_to_merge)} diacritics")

# Step 2: Try to merge main letters
print("\n--- STEP 2: Horizontal Letter Merging ---")
if len(main_letters_to_merge) > 1:
    print(f"Attempting to merge {len(main_letters_to_merge)} main letter components...")

    merged_count = 0
    for i, (idx_i, x_i, y_i, w_i, h_i) in enumerate(main_letters_to_merge):
        for j, (idx_j, x_j, y_j, w_j, h_j) in enumerate(main_letters_to_merge):
            if i >= j:
                continue

            horizontal_gap = max(0, max(x_i - (x_j + w_j), x_j - (x_i + w_i)))
            vertical_overlap = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)
            y_center_i = y_i + h_i / 2
            y_center_j = y_j + h_j / 2
            vertical_center_distance = abs(y_center_i - y_center_j)

            min_height = min(h_i, h_j)
            merge_threshold_gap = median_height * 0.3
            merge_threshold_center = median_height * 0.4
            merge_threshold_overlap = min_height * 0.2

            should_merge = (horizontal_gap < merge_threshold_gap and
                           vertical_center_distance < merge_threshold_center and
                           vertical_overlap > merge_threshold_overlap)

            if should_merge:
                print(f"\n  ✓ MERGE comp#{i} and comp#{j}:")
                print(f"    comp#{i}: ({x_i},{y_i}) {w_i}x{h_i}")
                print(f"    comp#{j}: ({x_j},{y_j}) {w_j}x{h_j}")
                print(f"    gap={horizontal_gap:.1f} < {merge_threshold_gap:.1f}")
                print(f"    center_dist={vertical_center_distance:.1f} < {merge_threshold_center:.1f}")
                print(f"    overlap={vertical_overlap} > {merge_threshold_overlap:.1f}")
                merged_count += 1

    if merged_count == 0:
        print("  ✗ NO components were merged!")
else:
    print("Only 1 main letter component - nothing to merge")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if len(main_letters_to_merge) >= 2 and merged_count > 0:
    print("✓ Split ö parts SHOULD be merging in the real code")
    print("  If they're not, there's a bug in the actual merging implementation")
elif len(main_letters_to_merge) < 2:
    print("❌ PROBLEM: Not enough LETTER components!")
    print("  One or both parts of ö might be misclassified as DIACRITIC")
else:
    print("❌ PROBLEM: Components don't meet merge criteria")
    print("  Thresholds may need adjustment")
