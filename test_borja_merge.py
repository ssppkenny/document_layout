#!/usr/bin/env python3
"""
Test if W36 (Börja) ö components would merge.
"""

# W36 components from diagnostic:
# #4 [LETTER]: x=26 y=15 w=10 h=22 (left part of ö)
# #1 [LETTER]: x=40 y=1 w=9 h=35 (could be right part of ö OR letter r/j)

# Let's check if #4 and something merges
components = {
    '#4': (26, 15, 10, 22),  # Possible left part of ö
    '#1': (40, 1, 9, 35),    # Tall letter - could be r or j, OR right part of ö?
}

median_height = 21.5
word_height = 39

print("="*80)
print("Testing W36 (possible Börja) component merging")
print("="*80)

# Check if #4 and #1 would merge
x_i, y_i, w_i, h_i = components['#4']
x_j, y_j, w_j, h_j = components['#1']

horizontal_gap = max(0, max(x_i - (x_j + w_j), x_j - (x_i + w_i)))
vertical_overlap = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)
y_center_i = y_i + h_i / 2
y_center_j = y_j + h_j / 2
vertical_center_distance = abs(y_center_i - y_center_j)

print(f"\nComponent #4 (left of ö?): x={x_i} y={y_i} w={w_i} h={h_i} center_y={y_center_i:.1f}")
print(f"Component #1 (right of ö?): x={x_j} y={y_j} w={w_j} h={h_j} center_y={y_center_j:.1f}")
print(f"\nHorizontal gap: {horizontal_gap}")
print(f"Vertical overlap: {vertical_overlap}")
print(f"Vertical center distance: {vertical_center_distance:.1f}")

# Test with current thresholds
min_height = min(h_i, h_j)
merge_threshold_gap = median_height * 0.3
merge_threshold_center = median_height * 0.4  # Increased from 0.3 to 0.4
merge_threshold_overlap = min_height * 0.2

print(f"\n--- NEW THRESHOLDS (40% center) ---")
print(f"Merge threshold gap: {merge_threshold_gap:.1f} (30% of median_height={median_height})")
print(f"Merge threshold center: {merge_threshold_center:.1f} (40% of median_height) ← INCREASED")
print(f"Merge threshold overlap: {merge_threshold_overlap:.1f} (20% of min_height={min_height})")

print(f"\nCondition checks:")
print(f"  horizontal_gap ({horizontal_gap}) < merge_threshold_gap ({merge_threshold_gap:.1f})? {horizontal_gap < merge_threshold_gap}")
print(f"  vertical_center_distance ({vertical_center_distance:.1f}) < merge_threshold_center ({merge_threshold_center:.1f})? {vertical_center_distance < merge_threshold_center}")
print(f"  vertical_overlap ({vertical_overlap}) > merge_threshold_overlap ({merge_threshold_overlap:.1f})? {vertical_overlap > merge_threshold_overlap}")

should_merge = (horizontal_gap < merge_threshold_gap and
               vertical_center_distance < merge_threshold_center and
               vertical_overlap > merge_threshold_overlap)

print(f"\n>>> SHOULD MERGE: {should_merge}")

if not should_merge:
    print(f"\n{'='*80}")
    print("❌ PROBLEM FOUND!")
    print(f"{'='*80}")

    if vertical_center_distance >= merge_threshold_center:
        print(f"\n  Issue: Vertical center distance too large!")
        print(f"  Component #4 center: {y_center_i:.1f}")
        print(f"  Component #1 center: {y_center_j:.1f}")
        print(f"  Distance: {vertical_center_distance:.1f}")
        print(f"  Threshold: {merge_threshold_center:.1f}")
        print(f"\n  Component #1 (h={h_j}) is MUCH TALLER than #4 (h={h_i})")
        print(f"  This suggests #1 is NOT part of ö, but a tall letter like 'r' or 'j'")
        print(f"\n  Real ö split might be between different components!")
