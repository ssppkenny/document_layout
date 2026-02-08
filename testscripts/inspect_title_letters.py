#!/usr/bin/env python3
"""
Visual inspection: Extract and display a few title letters including those with descenders
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

reflowed = cv2.imread('../output_reflowed.png')
if reflowed is None:
    print("No output_reflowed.png found")
    exit(1)

# Extract first 400 pixels (title region)
title_region = reflowed[:400, :, :]

# Convert to grayscale and find text
gray = cv2.cvtColor(title_region, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding boxes
boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 20:  # Filter small noise
        boxes.append((x, y, w, h))

# Sort by position (left to right)
boxes.sort(key=lambda b: (b[1], b[0]))

print(f"Found {len(boxes)} letter candidates in title region")

if len(boxes) > 0:
    # Group into lines
    lines = []
    current_line = [boxes[0]]

    for box in boxes[1:]:
        x, y, w, h = box
        last_y = current_line[-1][1]

        if abs(y - last_y) < 30:  # Same line
            current_line.append(box)
        else:  # New line
            lines.append(current_line)
            current_line = [box]

    if current_line:
        lines.append(current_line)

    print(f"Grouped into {len(lines)} lines")

    # Show first line (title)
    if lines:
        first_line = lines[0]
        print(f"\nFirst line: {len(first_line)} letters")

        # Create visualization
        fig, axes = plt.subplots(2, min(len(first_line), 10), figsize=(20, 4))

        for i, (x, y, w, h) in enumerate(first_line[:min(len(first_line), 10)]):
            letter_img = title_region[y:y+h, x:x+w]

            if len(first_line) == 1:
                ax = axes
            elif len(first_line) <= 10:
                ax = axes[0, i] if len(first_line) > 1 else axes[i]
            else:
                ax = axes[0, i]

            if letter_img.shape[0] > 0 and letter_img.shape[1] > 0:
                ax.imshow(cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB))
                ax.set_title(f'L{i}\n{w}x{h}', fontsize=8)
                ax.axis('off')

            print(f"  Letter {i}: x={x}, y={y}, w={w}, h={h}")

        # Hide unused subplots
        if len(first_line) < 10:
            for i in range(len(first_line), 10):
                if len(first_line) > 1:
                    axes[0, i].axis('off')
                    axes[1, i].axis('off')

        # Show baseline alignment
        if len(axes.shape) > 1:
            for i, (x, y, w, h) in enumerate(first_line[:10]):
                axes[1, i].barh([0], [h], height=0.5, label='Height')
                axes[1, i].axvline(x=0, color='r', linestyle='--', label='Top')
                axes[1, i].axvline(x=h, color='b', linestyle='--', label='Bottom')
                axes[1, i].set_xlim(-5, max(80, h+5))
                axes[1, i].set_title(f'y={y}', fontsize=8)
                axes[1, i].legend(fontsize=6)

        plt.tight_layout()
        plt.savefig('notebooks/title_letters_detail.png', dpi=150)
        print("\n✓ Saved to notebooks/title_letters_detail.png")
        plt.close()

        # Check for height variation that suggests clipping
        heights = [h for _, _, _, h in first_line]
        print(f"\nHeight statistics:")
        print(f"  Min: {min(heights)}")
        print(f"  Max: {max(heights)}")
        print(f"  Range: {max(heights) - min(heights)}")

        if max(heights) - min(heights) > 30:
            print("\n⚠️  HIGH VARIATION: Likely some letters are clipped")
            print("  Letters with descenders (g, p, y, q, j) should be taller")
