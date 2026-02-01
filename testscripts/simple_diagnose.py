#!/usr/bin/env python3
"""
Simple diagnostic - just show what find_rects returns for out13.png
This uses inline code to avoid import issues
"""
import cv2
import numpy as np

print("Loading image...")
img = cv2.imread('notebooks/out13.png')
if img is None:
    print("ERROR: Could not load image")
    exit(1)

print(f"Image loaded: {img.shape[1]}x{img.shape[0]}")

# Manually extract one word as a test
# Based on previous output, first word is around [73, 2, 140, 27]
word_box = (73, 2, 140, 27)
print(f"\nTesting on first word: {word_box}")

xmin, ymin, xmax, ymax = word_box
word_img = img[ymin:ymax, xmin:xmax, :].copy()

# Convert to grayscale and threshold
gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

print(f"Word image size: {word_img.shape[1]}x{word_img.shape[0]}")

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

print(f"Found {num_labels - 1} components")

# Show each component
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    # Convert to absolute coordinates
    abs_x = x + xmin
    abs_y = y + ymin

    print(f"  Component {i}: pos=({abs_x}, {abs_y}) size={w}x{h} area={area}")

    if w < 3 or h < 3:
        print(f"    -> TOO SMALL (< 3x3), would be filtered")
    elif h < 8:  # Likely a fragment
        print(f"    -> FRAGMENT? (height < 8)")

print("\n✓ Basic analysis complete")
print("This shows what connected components finds in the first word")
