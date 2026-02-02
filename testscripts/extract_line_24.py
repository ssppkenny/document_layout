#!/usr/bin/env python3
"""
Extract line 24 from dvurog_p021.png for analysis
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from doctr.models import detection_predictor
from doctr.io import DocumentFile
from docs.main import margins, merge_close_lines

# Load image
img = cv2.imread('images/dvurog_p021.png')
img_h, img_w, _ = img.shape
print(f'Image size: {img_w}x{img_h}')

# Get word detections
print('Running word detection...')
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images(['images/dvurog_p021.png'])
result = model(docs)
words = result[0]['words']

# Convert to absolute coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

print(f'Detected {len(words)} words')

# Find margins
left_margins, right_margins = margins(words)
left_margins, right_margins = merge_close_lines(left_margins, right_margins, words, y_threshold=20)

print(f'\nFound {len(left_margins)} lines')

# Extract line 24 (index 23, counting from 0)
line_idx = 23
if line_idx < len(left_margins):
    left_pt = left_margins[line_idx]
    right_pt = right_margins[line_idx]

    print(f'\nLine {line_idx + 1} (24):')
    print(f'  Left margin: {left_pt}')
    print(f'  Right margin: {right_pt}')

    # Find the bounding box for this line
    # Get all words that intersect this line
    line_y_min = int(min(left_pt[1], right_pt[1]) - 20)
    line_y_max = int(max(left_pt[1], right_pt[1]) + 20)

    line_words = []
    for xmin, ymin, xmax, ymax, _ in words:
        word_center_y = (ymin + ymax) / 2
        if line_y_min <= word_center_y <= line_y_max:
            line_words.append((xmin, ymin, xmax, ymax))

    print(f'  Found {len(line_words)} words in this line')

    if line_words:
        # Calculate bounding box
        min_x = int(min(w[0] for w in line_words))
        min_y = int(min(w[1] for w in line_words))
        max_x = int(max(w[2] for w in line_words))
        max_y = int(max(w[3] for w in line_words))

        # Add some padding
        padding = 5
        min_x = int(max(0, min_x - padding))
        min_y = int(max(0, min_y - padding))
        max_x = int(min(img_w, max_x + padding))
        max_y = int(min(img_h, max_y + padding))

        # Extract line
        line_img = img[min_y:max_y, min_x:max_x].copy()

        # Save
        output_path = 'line_24_extracted.png'
        cv2.imwrite(output_path, line_img)
        print(f'\n✓ Line extracted and saved to: {output_path}')
        print(f'  Bounding box: ({min_x}, {min_y}, {max_x}, {max_y})')
        print(f'  Line size: {max_x - min_x}x{max_y - min_y}')
else:
    print(f'\nERROR: Only {len(left_margins)} lines found, cannot extract line 24')
