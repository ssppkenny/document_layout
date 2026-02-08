#!/usr/bin/env python3
"""
Visual verification: Draw lines showing 145px spacing
"""

import cv2
import numpy as np

img = cv2.imread('../output_reflowed.png')
if img is None:
    print("Error: Can't load output_reflowed.png")
    exit(1)

# Create visualization
vis = img.copy()

# Draw horizontal lines every 145px starting from top margin (50px)
top_margin = 50
line_height = 145
color = (0, 255, 0)  # Green

y = top_margin
line_num = 0
while y < img.shape[0]:
    cv2.line(vis, (0, y), (img.shape[1], y), color, 2)
    cv2.putText(vis, f"{line_num}", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += line_height
    line_num += 1

# Save
cv2.imwrite('../notebooks/spacing_verification.png', vis)
print(f"✓ Saved visualization to notebooks/spacing_verification.png")
print(f"Green lines are drawn every {line_height}px starting from top margin {top_margin}px")
print("If the fix is working, text baselines should align with green lines")
