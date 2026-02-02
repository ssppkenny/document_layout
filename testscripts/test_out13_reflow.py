#!/usr/bin/env python3
"""
Test reflow of notebooks/out13.png to check if letters are clipped
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from doctr.models import detection_predictor
from doctr.io import DocumentFile
from docs.main import find_rects, margins
from ocr_reflow.reflow import create_page_with_word_wrapping
from docs.main import Letter
from operator import itemgetter
from math import ceil
from shapely.geometry import LineString, box

# Load image
img = cv2.imread('notebooks/out13.png')
img_h, img_w, _ = img.shape
print(f'Image size: {img_w}x{img_h}')

# Detect background
flat_img = img.reshape(-1, 3)
background_color = tuple(np.median(flat_img, axis=0).astype(np.uint8).tolist())
print(f'Background color: {background_color}')

# Get word detections
print('Running word detection...')
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images(['notebooks/out13.png'])
result = model(docs)
words = result[0]['words']

# Convert to absolute coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32) - 5  # left: expand left
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) - 5  # top: expand up
words[:, 2] = (words[:, 2] * img_w).astype(np.int32) + 5  # right: expand right
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) + 5  # bottom: expand down
# Clamp to image bounds
words[:, 0] = np.maximum(words[:, 0], 0)
words[:, 1] = np.maximum(words[:, 1], 0)
words[:, 2] = np.minimum(words[:, 2], img_w)
words[:, 3] = np.minimum(words[:, 3], img_h)

line_words = [(int(xmin), int(ymin), int(xmax), int(ymax)) for xmin, ymin, xmax, ymax, _ in words]

print(f'Detected {len(line_words)} words')

# Find margins (should detect 1 line)
left_margins, right_margins = margins(words)
print(f'Found {len(left_margins)} lines')

if len(left_margins) == 0:
    print('ERROR: No lines detected')
    exit(1)

# Create word rectangles
rectangles = dict([
    (box(w_xmin, w_ymin, w_xmax, w_ymax),
     (int(w_xmin), int(w_ymin), int(w_xmax), int(w_ymax)))
    for (w_xmin, w_ymin, w_xmax, w_ymax, _) in words
])

# Group words into lines
lines = []
for l, r in zip(left_margins, right_margins):
    line = LineString([(l[0], l[1]), (r[0], r[1])])
    line_words_in_line = []
    for b in rectangles:
        if line.intersects(b):
            line_words_in_line.append(rectangles[b])
    if line_words_in_line:
        lines.append(sorted(line_words_in_line))

print(f'Grouped into {len(lines)} lines')

# Extract letters from lines
all_lines = []
for line_idx, line in enumerate(lines):
    line_letters = find_rects(img, line)
    line_letters = sorted(line_letters, key=itemgetter(0))

    if len(line_letters) == 0:
        continue

    print(f'\nLine {line_idx}: {len(line_letters)} letters')

    # Calculate baseline
    heights = [l_ymax - l_ymin for l_xmin, l_ymin, l_xmax, l_ymax in line_letters]
    m_height = np.median(heights)
    sd = np.std(heights) if len(heights) > 1 else 0

    normal_letters = [
        (l_xmin, l_ymin, l_xmax, l_ymax)
        for l_xmin, l_ymin, l_xmax, l_ymax in line_letters
        if abs((l_ymax - l_ymin) - m_height) < sd
    ]

    if len(normal_letters) > 1:
        lower_points = [((l_xmin + l_xmax) / 2, l_ymax) for l_xmin, l_ymin, l_xmax, l_ymax in normal_letters]
        try:
            x_coords = [x for x, y in lower_points]
            y_coords = [y for x, y in lower_points]
            m, c = np.polyfit(x_coords, y_coords, 1)
            print(f'  Baseline slope: {m:.6f}')
        except:
            m, c = 0, 0
    else:
        m, c = 0, 0

    letters = [
        Letter(l_xmin, l_ymin, l_xmax, l_ymax, l_ymax - ceil(m * ((l_xmin + l_xmax) / 2) + c))
        for l_xmin, l_ymin, l_xmax, l_ymax in line_letters
    ]

    all_lines.append(letters)

    # Show first and last few letters
    print(f'  First 3 letters:')
    for i in range(min(3, len(letters))):
        l = letters[i]
        print(f'    Letter {i}: x=[{l.xmin},{l.xmax}] y=[{l.ymin},{l.ymax}] bl={l.bl}')

    print(f'  Last 3 letters:')
    for i in range(max(0, len(letters) - 3), len(letters)):
        l = letters[i]
        print(f'    Letter {i}: x=[{l.xmin},{l.xmax}] y=[{l.ymin},{l.ymax}] bl={l.bl}')

# Create reflowed page
print('\nCreating reflowed page...')
zoom_factor = 2.5
new_page_width = 2000

reflowed = create_page_with_word_wrapping(
    all_lines, img, zoom_factor, new_page_width,
    background_color=background_color
)

output_path = 'out13_reflowed_test.png'
cv2.imwrite(output_path, reflowed)
print(f'\n✓ Reflowed page saved to: {output_path}')
print(f'  Size: {reflowed.shape[1]}x{reflowed.shape[0]}')
