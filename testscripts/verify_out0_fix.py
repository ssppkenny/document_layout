#!/usr/bin/env python3
"""
Comprehensive verification that out0.png is now processed correctly with 12 lines
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from docs.main import margins

print("=" * 80)
print("VERIFICATION: out0.png Line Detection")
print("=" * 80)

# Load image
image_path = "notebooks/out0.png"
img = cv2.imread(image_path)
if img is None:
    print(f"✗ ERROR: Could not load {image_path}")
    sys.exit(1)

img_h, img_w, _ = img.shape
print(f"\n✓ Image loaded: {img_w} x {img_h}")

# Run text detection
print("✓ Running text detection...")
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images([image_path])
result = model(docs)
words = result[0]["words"]

# Convert normalized coordinates to absolute
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
words = words.astype(np.int32)

print(f"✓ Detected {len(words)} words")

# Detect margins (lines)
left_margins, right_margins = margins(words)

print(f"\n✓ Detected {len(left_margins)} lines")

# Create visualization
vis_img = img.copy()

# Draw all detected words in light gray
for xmin, ymin, xmax, ymax, _ in words:
    cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (200, 200, 200), 1)

# Draw detected lines with different colors
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

for i, (l, r) in enumerate(zip(left_margins, right_margins)):
    color = colors[i % len(colors)]

    # Draw line connecting left and right margins
    cv2.line(vis_img, l, r, color, 2)

    # Draw circles at leftmost point (blue-ish)
    cv2.circle(vis_img, l, 8, (255, 0, 0), -1)  # Blue filled circle
    cv2.circle(vis_img, l, 8, (255, 255, 255), 2)  # White border

    # Draw circles at rightmost point (yellow-ish)
    cv2.circle(vis_img, r, 8, (0, 255, 255), -1)  # Yellow filled circle
    cv2.circle(vis_img, r, 8, (255, 255, 255), 2)  # White border

    # Add line number label at the center
    mid_x = (l[0] + r[0]) // 2
    mid_y = (l[1] + r[1]) // 2
    cv2.putText(vis_img, f"L{i+1}", (mid_x - 20, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Save visualization
output_path = "out0_line_detection_visualization.png"
cv2.imwrite(output_path, vis_img)
print(f"✓ Visualization saved to: {output_path}")

if len(left_margins) == 12:
    print("\n" + "=" * 80)
    print("✓✓✓ SUCCESS! Correctly detected 12 lines ✓✓✓")
    print("=" * 80)
    print("\nLine details:")
    for i, (l, r) in enumerate(zip(left_margins, right_margins)):
        print(f"  Line {i+1:2d}: left=({l[0]:4d}, {l[1]:3d})  right=({r[0]:4d}, {r[1]:3d})  width={r[0]-l[0]:4d}px")
    print(f"\n📊 Visual representation saved to: {output_path}")
    print("   - Blue circles: leftmost points")
    print("   - Yellow circles: rightmost points")
    print("   - Colored lines: detected text lines")
    print("   - Gray rectangles: detected words")
    sys.exit(0)
else:
    print(f"\n✗ FAILURE: Expected 12 lines but got {len(left_margins)}")
    print("\nDetected lines:")
    for i, (l, r) in enumerate(zip(left_margins, right_margins)):
        print(f"  Line {i+1}: left=({l[0]}, {l[1]})  right=({r[0]}, {r[1]})")
    print(f"\n📊 Visual representation saved to: {output_path}")
    sys.exit(1)
