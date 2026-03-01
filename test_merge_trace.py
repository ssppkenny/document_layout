#!/usr/bin/env python3
"""
Test a single word with detailed tracing to find duplication issue
"""

import cv2
import numpy as np

# Load image
img = cv2.imread('images/gang_p023.png')

# Pick a word that we know has "ö" with 2 dots - word 7 from earlier diagnostic
# Word 7 from diagnostic: (546, 200) → (627, 253)
word_box = (546, 200, 627, 253)
xmin, ymin, xmax, ymax = word_box

# Extract word
word_img = img[ymin:ymax, xmin:xmax].copy()
word_gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
_, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_binary, 8, cv2.CV_32S)

word_height = ymax - ymin
word_width = xmax - xmin

print(f"Word at {word_box}")
print(f"Found {num_labels-1} connected components\n")

# Build valid_components list (same as in find_rects)
main_components = []
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    if w < 2 or h < 2 or area < 4:
        continue

    if h >= word_height * 0.3:
        main_components.append(i)

valid_components = []
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    if w < 2 or h < 2 or area < 4:
        continue

    if i in main_components:
        valid_components.append((x, y, w, h))
        continue

    # Check if near main component
    is_near_main = False
    for main_idx in main_components:
        main_x = stats[main_idx, cv2.CC_STAT_LEFT]
        main_y = stats[main_idx, cv2.CC_STAT_TOP]
        main_w = stats[main_idx, cv2.CC_STAT_WIDTH]
        main_h = stats[main_idx, cv2.CC_STAT_HEIGHT]
        main_bottom = main_y + main_h

        max_distance_above = word_height * 0.4

        if y < main_bottom + max_distance_above and y + h > main_y - max_distance_above:
            horizontal_gap = max(0, max(x - (main_x + main_w), main_x - (x + w)))
            if horizontal_gap < word_width * 0.3:
                is_near_main = True
                break

    if is_near_main:
        valid_components.append((x, y, w, h))

print(f"valid_components ({len(valid_components)} total):")
for idx, (x, y, w, h) in enumerate(valid_components):
    print(f"  [{idx}] ({x}, {y}) size={w}x{h}")

# Calculate median height
component_heights = [h for x, y, w, h in valid_components]
median_height = np.median(component_heights) if len(component_heights) > 0 else word_height * 0.5

print(f"\nMedian height: {median_height:.1f}px\n")

# Classify
dots_to_merge = []
main_letters_to_merge = []

for comp_idx, (x, y, w, h) in enumerate(valid_components):
    is_diacritic = (h < median_height * 0.4 and
                   w < median_height * 0.8 and
                   w * h < (median_height ** 2) * 0.3 and
                   h < word_height * 0.25 and
                   w < word_width * 0.5)

    if is_diacritic:
        dots_to_merge.append((comp_idx, x, y, w, h))
        print(f"  [{comp_idx}] DIACRITIC: ({x}, {y}) size={w}x{h}")
    else:
        main_letters_to_merge.append((comp_idx, x, y, w, h))
        print(f"  [{comp_idx}] MAIN: ({x}, {y}) size={w}x{h}")

# Group diacritics
print(f"\nGrouping diacritics...")
diacritic_groups = []
used_diacritics = set()

for i, (dot_idx_i, dx_i, dy_i, dw_i, dh_i) in enumerate(dots_to_merge):
    if i in used_diacritics:
        continue

    group = [(dot_idx_i, dx_i, dy_i, dw_i, dh_i)]
    used_diacritics.add(i)

    for j, (dot_idx_j, dx_j, dy_j, dw_j, dh_j) in enumerate(dots_to_merge):
        if j in used_diacritics or i == j:
            continue

        vertical_diff = abs((dy_i + dh_i/2) - (dy_j + dh_j/2))
        max_diacritic_h = max(dh_i, dh_j)
        horizontal_gap = max(0, max(dx_i - (dx_j + dw_j), dx_j - (dx_i + dw_i)))

        if (vertical_diff < max_diacritic_h * 0.5 and
            horizontal_gap < median_height * 0.6):
            group.append((dot_idx_j, dx_j, dy_j, dw_j, dh_j))
            used_diacritics.add(j)
            print(f"  → Grouped diacritics [{dot_idx_i}] and [{dot_idx_j}]")

    diacritic_groups.append(group)

print(f"\nFound {len(diacritic_groups)} diacritic group(s)")

# Merge with main letters
print(f"\nMerging diacritic groups with main letters...")
merged_indices = set()
merged_components = []

for group_idx, diacritic_group in enumerate(diacritic_groups):
    print(f"\nGroup {group_idx+1}:")
    print(f"  Contains {len(diacritic_group)} diacritic(s):")
    for dot_idx, dx, dy, dw, dh in diacritic_group:
        print(f"    [{dot_idx}] at ({dx}, {dy})")

    # Calculate bounding box
    group_left = min(dx for _, dx, dy, dw, dh in diacritic_group)
    group_top = min(dy for _, dx, dy, dw, dh in diacritic_group)
    group_right = max(dx + dw for _, dx, dy, dw, dh in diacritic_group)
    group_bottom = max(dy + dh for _, dx, dy, dw, dh in diacritic_group)

    print(f"  Group bbox: ({group_left}, {group_top}) → ({group_right}, {group_bottom})")

    # Find matching main letters
    matching_components = []

    for main_idx, mx, my, mw, mh in main_letters_to_merge:
        if main_idx in merged_indices:
            continue

        main_top = my
        main_left = mx
        main_right = mx + mw
        main_bottom = my + mh

        vertical_gap = main_top - group_bottom

        if vertical_gap < median_height:
            horizontal_overlap = min(group_right, main_right) - max(group_left, main_left)
            horizontal_gap = max(0, max(main_left - group_right, group_left - main_right))

            is_horizontally_aligned = (horizontal_overlap > 0 or horizontal_gap < median_height * 0.5)
            is_vertically_ok = (group_bottom <= main_bottom)

            if is_horizontally_aligned and is_vertically_ok:
                matching_components.append((main_idx, mx, my, mw, mh))
                print(f"  → Matches main letter [{main_idx}] at ({mx}, {my})")

    if matching_components:
        all_components = [(dx, dy, dw, dh) for _, dx, dy, dw, dh in diacritic_group] + \
                        [(mx, my, mw, mh) for _, mx, my, mw, mh in matching_components]

        merged_x = min(x for x, y, w, h in all_components)
        merged_y = min(y for x, y, w, h in all_components)
        merged_right = max(x + w for x, y, w, h in all_components)
        merged_bottom = max(y + h for x, y, w, h in all_components)
        merged_w = merged_right - merged_x
        merged_h = merged_bottom - merged_y

        merged_components.append((merged_x, merged_y, merged_w, merged_h))
        print(f"  → Created merged component: ({merged_x}, {merged_y}) size={merged_w}x{merged_h}")

        # Mark as used
        for dot_idx, _, _, _, _ in diacritic_group:
            merged_indices.add(dot_idx)
            print(f"    Marked [{dot_idx}] as merged")
        for main_idx, _, _, _, _ in matching_components:
            merged_indices.add(main_idx)
            print(f"    Marked [{main_idx}] as merged")

# Add non-merged components
print(f"\nAdding non-merged components...")
print(f"merged_indices = {merged_indices}")

for comp_idx, (x, y, w, h) in enumerate(valid_components):
    if comp_idx not in merged_indices:
        merged_components.append((x, y, w, h))
        print(f"  Added non-merged [{comp_idx}]: ({x}, {y}) size={w}x{h}")
    else:
        print(f"  Skipped merged [{comp_idx}]")

print(f"\n{'='*80}")
print(f"FINAL RESULT: {len(merged_components)} components")
print(f"{'='*80}")
for idx, (x, y, w, h) in enumerate(merged_components):
    print(f"  Component {idx}: ({x}, {y}) size={w}x{h}")
