#!/usr/bin/env python3
"""
Verify that lines are actually 145px apart by checking a small sample
"""

import cv2
import numpy as np

img = cv2.imread('../output_reflowed.png')
if img is None:
    print("Error: output_reflowed.png not found")
    exit(1)

h, w = img.shape[:2]
print(f"Image: {w}x{h}")

# Sample first 500 pixels
sample = img[:500, :, :]

# Convert to grayscale
gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

# Find text rows
projection = np.sum(255 - gray, axis=1)

# Simple peak finding
threshold = np.mean(projection) + 0.5 * np.std(projection)

peaks = []
for i in range(5, len(projection) - 5):
    if projection[i] > threshold:
        is_peak = all(projection[i] >= projection[j] for j in range(i-5, i+6) if j != i)
        if is_peak and (not peaks or i - peaks[-1] > 10):
            peaks.append(i)

print(f"\nFound {len(peaks)} lines in first 500px")
print("\nLine positions and spacings:")
for i, peak in enumerate(peaks):
    if i > 0:
        spacing = peak - peaks[i-1]
        print(f"Line {i}: y={peak}, spacing from prev={spacing}px")
    else:
        print(f"Line {i}: y={peak}")

if len(peaks) >= 2:
    spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
    print(f"\nSpacing statistics:")
    print(f"  Min: {min(spacings)}")
    print(f"  Max: {max(spacings)}")
    print(f"  Mean: {np.mean(spacings):.1f}")
    print(f"  Expected: 145px")

    if abs(np.mean(spacings) - 145) < 10:
        print("\n✅ SUCCESS! Lines are ~145px apart as expected")
    else:
        print(f"\n❌ PROBLEM! Lines are {np.mean(spacings):.1f}px apart, not 145px")
