#!/usr/bin/env python3
"""
Debug script to show individual word images and connected components
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from doctr.models import detection_predictor
from doctr.io import DocumentFile

# Load image
img = cv2.imread('notebooks/out13.png')
img_h, img_w, _ = img.shape
print(f'Image size: {img_w}x{img_h}')

# Get word detections
print('Running word detection...')
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images(['notebooks/out13.png'])
result = model(docs)
words = result[0]['words']

# Convert to absolute coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

line_words = [(int(xmin), int(ymin), int(xmax), int(ymax)) for xmin, ymin, xmax, ymax, _ in words]

print(f'Detected {len(line_words)} words\n')

# Focus on the last 3 words
for i in range(max(0, len(line_words) - 3), len(line_words)):
    word_box = line_words[i]
    xmin, ymin, xmax, ymax = word_box
    word_img = img[ymin:ymax, xmin:xmax, :].copy()

    print(f'Word {i}: box=[{xmin}, {ymin}, {xmax}, {ymax}], size={xmax-xmin}x{ymax-ymin}')

    # Save word image
    word_filename = f'word_{i}_original.png'
    cv2.imwrite(word_filename, word_img)
    print(f'  Saved to: {word_filename}')

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    binary_filename = f'word_{i}_binary.png'
    cv2.imwrite(binary_filename, binary)
    print(f'  Binary saved to: {binary_filename}')

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    print(f'  Found {num_labels - 1} connected components:')

    word_height = ymax - ymin
    for j in range(1, num_labels):
        x = stats[j, cv2.CC_STAT_LEFT]
        y = stats[j, cv2.CC_STAT_TOP]
        w = stats[j, cv2.CC_STAT_WIDTH]
        h = stats[j, cv2.CC_STAT_HEIGHT]
        area = stats[j, cv2.CC_STAT_AREA]

        # Check if it would pass our filters
        passes = w >= 3 and h >= 3 and area >= 9 and h >= word_height * 0.2
        status = "✓ KEPT" if passes else "✗ FILTERED"

        print(f'    Component {j}: pos=({x},{y}) size={w}x{h} area={area} height_ratio={h/word_height:.2f} {status}')

    print()

print('✓ Debug complete!')
