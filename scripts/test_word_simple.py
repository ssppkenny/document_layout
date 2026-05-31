#!/usr/bin/env python3
"""
Simple test: Extract one word with Swedish diacritics and show segmentation
"""

import cv2
import numpy as np

# Test on a crop from the image that contains Swedish diacritics
img = cv2.imread('images/gang_p023.png')

# Word 7 from diagnostic (has 2 diacritic components): (546, 200) → (627, 253)
word_box = (546, 200, 627, 253)
xmin, ymin, xmax, ymax = word_box

# Extract word
word_img = img[ymin:ymax, xmin:xmax].copy()
word_gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
_, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find connected components (before any merging)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_binary, 8, cv2.CV_32S)

print(f"Word at {word_box}: Found {num_labels-1} connected components")

# Show component details
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    print(f"  Component {i}: ({x}, {y}) size={w}x{h}")

# Visualize
word_vis = word_img.copy()
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    cv2.rectangle(word_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.putText(word_vis, str(i), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

# Save
cv2.imwrite('test_word_components.png', word_vis)
cv2.imwrite('test_word_binary.png', word_binary)
print("\n✓ Saved test_word_components.png and test_word_binary.png")

# Now test with our merge logic
print("\n" + "="*60)
print("Testing with diacritic merging logic:")
print("="*60)

word_height = ymax - ymin
word_width = xmax - xmin

# Calculate median component height
components = []
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    if w >= 2 and h >= 2 and area >= 4:
        components.append((x, y, w, h))

if components:
    component_heights = [h for x, y, w, h in components]
    median_height = np.median(component_heights)
    print(f"\nMedian component height: {median_height:.1f}px")

    # Classify as diacritics or main letters
    diacritics = []
    main_letters = []

    for x, y, w, h in components:
        is_diacritic = (h < median_height * 0.4 and
                       w < median_height * 0.8 and
                       w * h < (median_height ** 2) * 0.3 and
                       h < word_height * 0.25 and
                       w < word_width * 0.5)

        if is_diacritic:
            diacritics.append((x, y, w, h))
            print(f"  DIACRITIC: ({x}, {y}) size={w}x{h}")
        else:
            main_letters.append((x, y, w, h))
            print(f"  MAIN: ({x}, {y}) size={w}x{h}")

    # Test diacritic grouping (new logic)
    print(f"\n{'='*60}")
    print("Testing diacritic grouping (for Swedish ä, ö):")
    print(f"{'='*60}")

    diacritic_groups = []
    used = set()

    for i, (dx_i, dy_i, dw_i, dh_i) in enumerate(diacritics):
        if i in used:
            continue

        group = [(dx_i, dy_i, dw_i, dh_i)]
        used.add(i)

        # Find adjacent diacritics
        for j, (dx_j, dy_j, dw_j, dh_j) in enumerate(diacritics):
            if j in used or i == j:
                continue

            vertical_diff = abs((dy_i + dh_i/2) - (dy_j + dh_j/2))
            max_h = max(dh_i, dh_j)
            horizontal_gap = max(0, max(dx_i - (dx_j + dw_j), dx_j - (dx_i + dw_i)))

            if vertical_diff < max_h * 0.5 and horizontal_gap < median_height * 0.6:
                group.append((dx_j, dy_j, dw_j, dh_j))
                used.add(j)
                print(f"  → Grouped diacritics at ({dx_i}, {dy_i}) and ({dx_j}, {dy_j})")

        diacritic_groups.append(group)

    print(f"\nFound {len(diacritic_groups)} diacritic group(s)")
    for idx, group in enumerate(diacritic_groups):
        print(f"  Group {idx+1}: {len(group)} component(s)")
        for dx, dy, dw, dh in group:
            print(f"    - ({dx}, {dy}) size={dw}x{dh}")
