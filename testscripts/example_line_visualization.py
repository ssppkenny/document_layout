#!/usr/bin/env python3
"""
Example script showing how to use the visualize_detected_lines function
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from ocr_reflow import margins, visualize_detected_lines

# Example usage
image_path = "notebooks/out0.png"

print("=" * 80)
print("LINE DETECTION AND VISUALIZATION EXAMPLE")
print("=" * 80)

# Load image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load {image_path}")
    exit(1)

img_h, img_w = img.shape[:2]
print(f"\n1. Loaded image: {img_w} x {img_h}")

# Run text detection
print("2. Running text detection...")
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

print(f"   Detected {len(words)} words")

# Detect line margins
print("3. Detecting text line margins...")
left_margins, right_margins = margins(words)
print(f"   Detected {len(left_margins)} lines")

# Visualize detected lines
print("4. Creating visualization...")
vis_img = visualize_detected_lines(
    img, words, left_margins, right_margins,
    output_path="line_detection_example.png"
)

print("\n" + "=" * 80)
print("✓ VISUALIZATION COMPLETE")
print("=" * 80)
print(f"\nOutput: line_detection_example.png")
print("\nVisualization legend:")
print("  - Blue circles: Leftmost points of each line")
print("  - Yellow circles: Rightmost points of each line")
print("  - Colored lines: Detected text baselines")
print("  - Gray rectangles: Individual detected words")
print(f"\nDetected lines:")
for i, (l, r) in enumerate(zip(left_margins, right_margins)):
    width = r[0] - l[0]
    print(f"  Line {i+1:2d}: ({l[0]:4d}, {l[1]:3d}) → ({r[0]:4d}, {r[1]:3d})  width={width:4d}px")

print("\n" + "=" * 80)
print("You can also use this function in a Jupyter notebook:")
print("=" * 80)
print("""
from ocr_reflow import margins, visualize_detected_lines
import cv2
import matplotlib.pyplot as plt
from doctr.models import detection_predictor
from doctr.io import DocumentFile

# Load your image
img = cv2.imread('your_image.png')
img_h, img_w = img.shape[:2]

# Detect words
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images(['your_image.png'])
result = model(docs)
words = result[0]["words"]

# Convert coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

# Detect lines and visualize
left_margins, right_margins = margins(words)
vis_img = visualize_detected_lines(img, words, left_margins, right_margins)

# Display in notebook
plt.figure(figsize=(15, 20))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.title(f'Detected Lines: {len(left_margins)}')
plt.axis('off')
plt.show()
""")
