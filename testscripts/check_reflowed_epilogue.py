#!/usr/bin/env python3
"""
Check the actual reflowed Epilogue letters to see vertical splitting
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load reflowed output
img = cv2.imread('../output_reflowed.png')
if img is None:
    print("No output_reflowed.png found!")
    exit(1)

print("Checking reflowed output for Epilogue title")

# Find the second title line (after FIGURE 1.16)
# Look in the area after the first ~200px
search_area = img[200:600, :, :]

gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get boxes
boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 5 and h > 15:
        boxes.append((x, y + 200, w, h, w*h))  # Adjust y back to full image coords

# Group into lines
boxes.sort(key=lambda b: (b[1], b[0]))
lines = []
current_line = []

for box in boxes:
    x, y, w, h, area = box
    if not current_line or abs(y - current_line[0][1]) < 40:
        current_line.append(box)
    else:
        if current_line:
            lines.append(current_line)
        current_line = [box]
if current_line:
    lines.append(current_line)

print(f"\nGrouped into {len(lines)} lines")

# Show ALL lines
for i, line in enumerate(lines):
    print(f"Line {i}: {len(line)} components")

# Find the line that's likely "Epilogue" (should have ~8-10 components)
epilogue_line = None
epilogue_idx = -1
for i, line in enumerate(lines):
    if 6 <= len(line) <= 12:
        print(f"  → Line {i} is a candidate: {len(line)} components (possible Epilogue)")
        if epilogue_line is None:  # Take first candidate
            epilogue_line = line
            epilogue_idx = i

if epilogue_line:
    print(f"\nAnalyzing suspected Epilogue line: {len(epilogue_line)} components")

    widths = [w for x, y, w, h, a in epilogue_line]
    heights = [h for x, y, w, h, a in epilogue_line]

    median_width = np.median(widths)
    median_height = np.median(heights)

    print(f"  Widths: min={min(widths)}, max={max(widths)}, median={median_width:.1f}")
    print(f"  Heights: min={min(heights)}, max={max(heights)}, median={median_height:.1f}")

    # Check for very narrow components
    narrow_threshold = median_width * 0.4
    narrow_components = [(i, x, y, w, h) for i, (x, y, w, h, a) in enumerate(epilogue_line) if w < narrow_threshold]

    if narrow_components:
        print(f"\n⚠️  {len(narrow_components)} VERY NARROW components detected:")
        for i, x, y, w, h in narrow_components:
            print(f"    Component {i}: x={x}, w={w} (only {w/median_width:.1%} of median width)")
        print("\n  These narrow components indicate VERTICAL SPLITTING of letters!")

    # Check for components that should be merged (same Y, close X, similar height)
    merge_candidates = []
    for i in range(len(epilogue_line)):
        x1, y1, w1, h1, _ = epilogue_line[i]
        for j in range(i+1, len(epilogue_line)):
            x2, y2, w2, h2, _ = epilogue_line[j]

            y_diff = abs(y1 - y2)
            x_gap = x2 - (x1 + w1)
            height_diff = abs(h1 - h2)

            # If at same Y, small gap, similar height => likely same letter split
            if y_diff < 10 and 0 < x_gap < median_width * 0.5 and height_diff < 10:
                merge_candidates.append((i, j, x_gap))

    if merge_candidates:
        print(f"\n⚠️  {len(merge_candidates)} pairs should probably be MERGED:")
        for i, j, gap in merge_candidates:
            x1, y1, w1, h1, _ = epilogue_line[i]
            x2, y2, w2, h2, _ = epilogue_line[j]
            print(f"    Components {i} and {j}: gap={gap}px, likely split parts of ONE letter")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    # Show the line with boxes
    y_min = min(y for x, y, w, h, a in epilogue_line) - 10
    y_max = max(y + h for x, y, w, h, a in epilogue_line) + 10
    x_min = min(x for x, y, w, h, a in epilogue_line) - 10
    x_max = max(x + w for x, y, w, h, a in epilogue_line) + 10

    line_img = img[y_min:y_max, x_min:x_max, :].copy()

    for i, (x, y, w, h, a) in enumerate(epilogue_line):
        # Adjust coordinates to line_img
        lx = x - x_min
        ly = y - y_min

        # Color: red if narrow, green otherwise
        color = (0, 0, 255) if w < narrow_threshold else (0, 255, 0)
        cv2.rectangle(line_img, (lx, ly), (lx+w, ly+h), color, 2)
        cv2.putText(line_img, str(i), (lx+2, ly+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    axes[0].imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Epilogue Line: {len(epilogue_line)} components (Red=narrow, may indicate vertical split)')
    axes[0].axis('off')

    # Width distribution
    axes[1].bar(range(len(widths)), widths,
               color=['red' if w < narrow_threshold else 'green' for w in widths])
    axes[1].axhline(y=median_width, color='blue', linestyle='--', label=f'Median: {median_width:.1f}')
    axes[1].axhline(y=narrow_threshold, color='red', linestyle='--', label=f'Narrow: {narrow_threshold:.1f}')
    axes[1].set_xlabel('Component index')
    axes[1].set_ylabel('Width (pixels)')
    axes[1].set_title('Component Widths in Reflowed Epilogue')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/reflowed_epilogue_split_check.png', dpi=150)
    print("\n✓ Saved to notebooks/reflowed_epilogue_split_check.png")
    plt.close()

else:
    print("\nCould not identify Epilogue line automatically")
