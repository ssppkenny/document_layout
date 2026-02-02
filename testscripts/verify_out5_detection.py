#!/usr/bin/env python3
"""
Create visualization for out5.png showing current detection
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import sys
sys.path.insert(0, 'src')
from docs.main import margins, merge_close_lines, visualize_detected_lines

image_path = "images/out5.png"
img = cv2.imread(image_path)
img_h, img_w, _ = img.shape

print(f"Analyzing {image_path}")
print("=" * 80)

# Run text detection
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images([image_path])
result = model(docs)
words = result[0]["words"]

# Convert coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
words = words.astype(np.int32)

print(f"Words detected: {len(words)}")

# Detect margins
left_margins, right_margins = margins(words)
print(f"Lines before merging: {len(left_margins)}")

for i, (l, r) in enumerate(zip(left_margins, right_margins)):
    print(f"  Line {i+1}: y={l[1]}")

# Merge
left_margins_m, right_margins_m = merge_close_lines(left_margins, right_margins, words, y_threshold=20)
print(f"\nLines after merging (y_threshold=20): {len(left_margins_m)}")

for i, (l, r) in enumerate(zip(left_margins_m, right_margins_m)):
    print(f"  Line {i+1}: y={l[1]}")

# Create visualization
vis_img = visualize_detected_lines(img, words, left_margins_m, right_margins_m,
                                   output_path="out5_current_detection.png")

print(f"\n✓ Visualization saved to: out5_current_detection.png")
print(f"Currently detecting {len(left_margins_m)} lines (expected 6)")

if len(left_margins_m) == 6:
    print("✓✓✓ SUCCESS!")
else:
    print(f"✗ Need to adjust from {len(left_margins_m)} to 6 lines")
