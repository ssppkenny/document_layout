#!/usr/bin/env python3
"""
Check the actual reflowed output to count letters in the title
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load reflowed output
img = cv2.imread('../output_reflowed.png')
if img is None:
    print("No output_reflowed.png found!")
    exit(1)

# Look at first 400 pixels (title area)
title_area = img[:400, :, :]

# Convert to grayscale and find components
gray = cv2.cvtColor(title_area, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find all text components
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and get bounding boxes
boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 5 and h > 10:  # Filter noise
        boxes.append((x, y, w, h, w*h))

# Sort by position
boxes.sort(key=lambda b: (b[1], b[0]))

print(f"Found {len(boxes)} text components in title area (first 400px)")

# Group into lines
lines = []
current_line = []
for box in boxes:
    x, y, w, h, area = box
    if not current_line or abs(y - current_line[0][1]) < 30:
        current_line.append(box)
    else:
        if current_line:
            lines.append(current_line)
        current_line = [box]
if current_line:
    lines.append(current_line)

print(f"\nGrouped into {len(lines)} lines")

# Analyze first line (likely the title)
if lines:
    first_line = lines[0]
    print(f"\nFirst line (title): {len(first_line)} components")

    # Check for narrow components (likely over-split)
    widths = [w for x, y, w, h, area in first_line]
    median_width = np.median(widths)
    narrow_threshold = median_width * 0.4

    narrow_count = sum(1 for w in widths if w < narrow_threshold)

    print(f"  Width stats: min={min(widths)}, max={max(widths)}, median={median_width:.1f}")
    print(f"  Narrow components (< 40% median): {narrow_count}")

    if len(first_line) > 12:
        print(f"\n⚠️  SEVERE OVER-SPLITTING! {len(first_line)} components in first line")
        print(f"  Expected ~8-10 for 'Epilogue' + potential extra letters")

    # Show details
    print(f"\nFirst line component details:")
    for i, (x, y, w, h, area) in enumerate(first_line[:20]):  # Show first 20
        status = "NARROW" if w < narrow_threshold else "OK"
        print(f"  {i:2d}: x={x:4d}, y={y:3d}, w={w:3d}, h={h:3d}, area={area:5d}  {status}")

    if len(first_line) > 20:
        print(f"  ... and {len(first_line) - 20} more")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # Title area with boxes
    vis = title_area.copy()
    for i, (x, y, w, h, area) in enumerate(first_line):
        color = (0, 255, 0) if w >= narrow_threshold else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, str(i), (x+2, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Title Line: {len(first_line)} components (Green=OK, Red=too narrow)')
    axes[0].axis('off')

    # Width distribution
    axes[1].bar(range(len(first_line)), widths,
               color=['green' if w >= narrow_threshold else 'red' for w in widths])
    axes[1].axhline(y=median_width, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_width:.1f}')
    axes[1].axhline(y=narrow_threshold, color='red', linestyle='--', linewidth=1, label=f'Narrow threshold: {narrow_threshold:.1f}')
    axes[1].set_xlabel('Component index')
    axes[1].set_ylabel('Width (pixels)')
    axes[1].set_title(f'Component Widths - {narrow_count} narrow components suggest over-splitting')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/reflowed_title_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to notebooks/reflowed_title_analysis.png")
    plt.close()
