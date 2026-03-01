#!/usr/bin/env python3
"""
Test why W51 (för) components are not merging.
"""

# W51 "för" components:
# #4 [LETTER]: x=20 y=20 w=22 h=22 (left part of ö)
# #5 [LETTER]: x=44 y=20 w=16 h=21 (right part of ö OR letter r?)

components = {
    '#4': (20, 20, 22, 22),
    '#5': (44, 20, 16, 21),
}

median_height = 21.0
word_height = 48

print("="*80)
print("Testing W51 (för) component merging")
print("="*80)

x_i, y_i, w_i, h_i = components['#4']
x_j, y_j, w_j, h_j = components['#5']

horizontal_gap = max(0, max(x_i - (x_j + w_j), x_j - (x_i + w_i)))
vertical_overlap = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)
y_center_i = y_i + h_i / 2
y_center_j = y_j + h_j / 2
vertical_center_distance = abs(y_center_i - y_center_j)

print(f"\nComponent #4: x={x_i} y={y_i} w={w_i} h={h_i} center_y={y_center_i:.1f}")
print(f"Component #5: x={x_j} y={y_j} w={w_j} h={h_j} center_y={y_center_j:.1f}")
print(f"\nHorizontal gap: {horizontal_gap}")
print(f"Vertical overlap: {vertical_overlap}")
print(f"Vertical center distance: {vertical_center_distance:.1f}")

min_height = min(h_i, h_j)
merge_threshold_gap = median_height * 0.3
merge_threshold_center = median_height * 0.4
merge_threshold_overlap = min_height * 0.2

print(f"\n--- CURRENT THRESHOLDS ---")
print(f"Merge threshold gap: {merge_threshold_gap:.1f} (30% of median_height={median_height})")
print(f"Merge threshold center: {merge_threshold_center:.1f} (40% of median_height)")
print(f"Merge threshold overlap: {merge_threshold_overlap:.1f} (20% of min_height={min_height})")

print(f"\nCondition checks:")
print(f"  horizontal_gap ({horizontal_gap}) < merge_threshold_gap ({merge_threshold_gap:.1f})? {horizontal_gap < merge_threshold_gap}")
print(f"  vertical_center_distance ({vertical_center_distance:.1f}) < merge_threshold_center ({merge_threshold_center:.1f})? {vertical_center_distance < merge_threshold_center}")
print(f"  vertical_overlap ({vertical_overlap}) > merge_threshold_overlap ({merge_threshold_overlap:.1f})? {vertical_overlap > merge_threshold_overlap}")

should_merge = (horizontal_gap < merge_threshold_gap and
               vertical_center_distance < merge_threshold_center and
               vertical_overlap > merge_threshold_overlap)

print(f"\n>>> SHOULD MERGE: {should_merge}")

if should_merge:
    print(f"\n{'='*80}")
    print("✓ Components SHOULD merge - but user says they're still split!")
    print(f"{'='*80}")
    print("\nPossible reasons:")
    print("1. The merging code is not being executed")
    print("2. Components are being processed in wrong order")
    print("3. One component is being classified as DIACRITIC instead of LETTER")
    print("\nNeed to check component classification!")
