#!/usr/bin/env python3
"""
Extract and display specific words from reflowed output to verify Swedish characters
"""

import cv2
import numpy as np

# Load reflowed image
img = cv2.imread('output_reflowed.png')
if img is None:
    print("ERROR: Could not load output_reflowed.png")
    exit(1)

print(f"Reflowed image size: {img.shape[1]}x{img.shape[0]}")

# Extract first few lines to examine words
# Typically text starts around y=100-200
top_section = img[100:500, :600]

cv2.imwrite('/tmp/reflowed_top_section.png', top_section)
print("Saved top section to /tmp/reflowed_top_section.png")

# Look for words that should contain ö, ä, å
# Sample various vertical positions
samples = [
    (100, 300, 0, 600, "line_1"),
    (300, 500, 0, 600, "line_2"),
    (500, 700, 0, 600, "line_3"),
    (700, 900, 0, 600, "line_4"),
    (900, 1100, 0, 600, "line_5"),
]

for y1, y2, x1, x2, label in samples:
    if y2 <= img.shape[0] and x2 <= img.shape[1]:
        section = img[y1:y2, x1:x2]
        cv2.imwrite(f'/tmp/reflowed_{label}.png', section)
        print(f"Saved {label} to /tmp/reflowed_{label}.png")

print("\nTo visually inspect:")
print("  eog /tmp/reflowed_*.png")
print("\nLook for words with ö, ä, å to see if they're properly formed")
