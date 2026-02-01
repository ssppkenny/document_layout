#!/usr/bin/env python3
"""
Analyze the last word in the reflowed line to check for clipping
Extract individual letters and check their integrity
"""
import cv2
import numpy as np

# Load the reflowed second line
line2 = cv2.imread('out13_reflowed_line2.png')

if line2 is None:
    print("ERROR: Could not load out13_reflowed_line2.png")
    exit(1)

print(f'Line 2 size: {line2.shape[1]}x{line2.shape[0]}')

# Convert to grayscale
gray = cv2.cvtColor(line2, cv2.COLOR_BGR2GRAY)

# Find all text regions (dark pixels)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours to locate letters
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f'Found {len(contours)} contours')

# Filter to get reasonable letter-sized contours
letter_contours = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)

    # Filter by size (letters should be 10-200 pixels wide and 10-100 pixels high)
    if w >= 10 and w <= 200 and h >= 10 and h <= 100 and area > 100:
        letter_contours.append((x, y, w, h, area))

# Sort by x position
letter_contours.sort(key=lambda c: c[0])

print(f'Found {len(letter_contours)} letter-sized contours')

# Show the last 5 letters
print('\nLast 5 letters:')
for i in range(max(0, len(letter_contours) - 5), len(letter_contours)):
    x, y, w, h, area = letter_contours[i]
    print(f'  Letter {i}: x={x} y={y} w={w} h={h} area={area}')

    # Check if letter is near the right edge
    distance_from_right = line2.shape[1] - (x + w)
    if distance_from_right < 50:
        print(f'    ⚠ WARNING: Letter is near right edge (distance: {distance_from_right} pixels)')

    # Extract and save letter for visual inspection
    letter_img = line2[y:y+h, x:x+w].copy()
    letter_filename = f'letter_{i}_x{x}.png'
    cv2.imwrite(letter_filename, letter_img)
    print(f'    Saved to: {letter_filename}')

# Create visualization with bounding boxes
vis = line2.copy()
for i, (x, y, w, h, _) in enumerate(letter_contours):
    # Draw green box for most letters, red for last 3
    color = (0, 0, 255) if i >= len(letter_contours) - 3 else (0, 255, 0)
    cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)

    # Add letter index
    cv2.putText(vis, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite('out13_line2_annotated.png', vis)
print(f'\n✓ Annotated line saved to: out13_line2_annotated.png')
print('  Green boxes = all letters')
print('  Red boxes = last 3 letters')
