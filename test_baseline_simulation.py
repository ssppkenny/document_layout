#!/usr/bin/env python3
"""
Detailed test to trace baseline calculations for W32-W37
"""

import cv2
import numpy as np
from math import ceil

img = cv2.imread('images/gang_p023_lines1.png')

# Exact letter-level components for the line with W32-W37
# These would be extracted by find_rects in the actual code
# For now, simulate with word-level bboxes

print("="*70)
print("SIMULATING BASELINE CALCULATION FOR W32-W37 LINE")
print("="*70)
print()

# Simulate line_letters (would come from find_rects)
# Using word bboxes as approximation
line_letters = [
    (95, 188, 112, 203),    # W32 "
    (104, 187, 218, 235),   # W33 Börja
    (231, 187, 315, 225),   # W34 med
    (326, 194, 382, 225),   # W35 att
    (393, 198, 485, 222),   # W36 sova
    (487, 185, 507, 202),   # W37 .
]

# Step 1: Calculate median height
heights = [ymax - ymin for xmin, ymin, xmax, ymax in line_letters]
m_height = np.median(heights)
print("Step 1: Median height = {:.1f}px".format(m_height))
print()

# Step 2: Fit baseline through normal letters
normal_letters = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in line_letters
                  if (ymax - ymin) >= m_height * 0.75]

print("Step 2: Normal letters (height >= 75% of median):")
for i, (xmin, ymin, xmax, ymax) in enumerate(normal_letters):
    print("  Letter {}: height={}px".format(i+1, ymax-ymin))
print()

if len(normal_letters) > 1:
    lower_points = [((xmin + xmax) / 2, ymax) for xmin, ymin, xmax, ymax in normal_letters]
    x_coords = [x for x, y in lower_points]
    y_coords = [y for x, y in lower_points]
    m, c = np.polyfit(x_coords, y_coords, 1)
    print("Step 3: Fitted baseline: y = {:.6f}x + {:.2f}".format(m, c))
else:
    m, c = 0, 0
    print("Step 3: Not enough normal letters, using horizontal baseline")
print()

# Step 4: Calculate reference baseline (NEW METHOD)
normal_ymaxs = [ymax for xmin, ymin, xmax, ymax in line_letters
                if (ymax - ymin) >= m_height * 0.75]

if normal_ymaxs:
    reference_baseline_y = np.median(normal_ymaxs)
    print("Step 4: Reference baseline y (median of normal ymaxs) = {:.1f}".format(reference_baseline_y))
else:
    reference_baseline_y = np.median([ymax for _, _, _, ymax in line_letters])
    print("Step 4: Reference baseline y (median of all ymaxs) = {:.1f}".format(reference_baseline_y))
print()

# Step 5: Calculate baseline shifts
print("Step 5: Baseline shifts for each letter:")
print()

for i, (xmin, ymin, xmax, ymax) in enumerate(line_letters):
    letter_height = ymax - ymin
    letter_width = xmax - xmin
    center_x = (xmin + xmax) / 2
    baseline_y_at_x = ceil(m * center_x + c)

    # Determine shift
    min_y_in_line = min(ly for _, ly, _, _ in line_letters)
    is_opening_quote = (
        letter_height < m_height * 0.75 and
        letter_width < 20 and
        ymin <= min_y_in_line + 5
    )

    if is_opening_quote:
        baseline_shift = -int(letter_height * 0.5)
        method = "opening quote (negative)"
    elif letter_height < m_height * 0.75:
        baseline_shift = int(letter_height * 0.8)
        method = "punctuation (own height)"
    else:
        # NEW: Use reference baseline instead of individual ymax
        baseline_shift = int(reference_baseline_y - baseline_y_at_x)
        method = "normal (REFERENCE)"

    print("  Letter {}: ymax={:3d}, shift={:4.0f}px ({})".format(
        i+1, ymax, baseline_shift, method))

print()
print("="*70)
print("EXPECTED RESULT:")
print("  All NORMAL letters (W33-W36) should have SIMILAR shifts")
print("  This means they will align horizontally on reflowed page")
print("  Opening quote (W32) will be negative (appears at top)")
print("="*70)
