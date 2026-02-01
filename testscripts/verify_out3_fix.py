#!/usr/bin/env python3
"""
Verification that out3.png is now processed correctly with 5 lines using adaptive threshold
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from ocr_reflow.main import margins, visualize_detected_lines

print("=" * 80)
print("VERIFICATION: out3.png Line Detection with Adaptive Threshold")
print("=" * 80)

# Load image
image_path = "notebooks/out3.png"
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

# Detect margins (lines) using adaptive threshold
left_margins, right_margins = margins(words)

# Create visualization
vis_img = visualize_detected_lines(img, words, left_margins, right_margins,
                                   output_path="out3_line_detection_visualization.png")

print(f"\n✓ Detected {len(left_margins)} lines")

if len(left_margins) == 5:
    print("\n" + "=" * 80)
    print("✓✓✓ SUCCESS! Correctly detected 5 lines using ADAPTIVE THRESHOLD ✓✓✓")
    print("=" * 80)
    print("\nLine details:")
    for i, (l, r) in enumerate(zip(left_margins, right_margins)):
        print(f"  Line {i+1:2d}: left=({l[0]:4d}, {l[1]:3d})  right=({r[0]:4d}, {r[1]:3d})  width={r[0]-l[0]:4d}px")
    print(f"\n📊 Visual representation saved to: out3_line_detection_visualization.png")
    print("   - Blue circles: leftmost points")
    print("   - Yellow circles: rightmost points")
    print("   - Colored lines: detected text lines")
    print("   - Gray rectangles: detected words")
    print("\n✨ The adaptive threshold automatically adjusted to this document's spacing!")
    sys.exit(0)
else:
    print(f"\n✗ FAILURE: Expected 5 lines but got {len(left_margins)}")
    print("\nDetected lines:")
    for i, (l, r) in enumerate(zip(left_margins, right_margins)):
        print(f"  Line {i+1}: left=({l[0]}, {l[1]})  right=({r[0]}, {r[1]})")
    print(f"\n📊 Visual representation saved to: out3_line_detection_visualization.png")
    sys.exit(1)
