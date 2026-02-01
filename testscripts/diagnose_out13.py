#!/usr/bin/env python3
"""
Diagnostic script to visualize detected words and letters in out13.png
"""
import cv2
import numpy as np
import sys

print("Starting diagnostic script...", flush=True)

sys.path.insert(0, 'src')

print("Importing modules...", flush=True)
from doctr.models import detection_predictor
from doctr.io import DocumentFile
print("Imports successful", flush=True)

from ocr_reflow.main import find_rects
print("find_rects imported", flush=True)

# Load image
print("Loading image...", flush=True)
img = cv2.imread('notebooks/out13.png')
if img is None:
    print("ERROR: Could not load notebooks/out13.png")
    sys.exit(1)

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

print(f'Detected {len(line_words)} words')

# Extract letters
print('Extracting letters...')
rects = find_rects(img, line_words)
print(f'Extracted {len(rects)} letter rectangles')

# Create visualization
vis = img.copy()

# Draw word boxes in blue
for xmin, ymin, xmax, ymax in line_words:
    cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

# Draw letter boxes in red
for xmin, ymin, xmax, ymax in rects:
    cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

# Save visualization
output_file = 'out13_word_letter_detection.png'
cv2.imwrite(output_file, vis)
print(f'\n✓ Visualization saved to: {output_file}')
print('  Blue boxes = detected words')
print('  Red boxes = extracted letters')

# Print statistics
print(f'\nWord statistics:')
word_widths = [xmax - xmin for xmin, ymin, xmax, ymax in line_words]
word_heights = [ymax - ymin for xmin, ymin, xmax, ymax in line_words]
print(f'  Width: min={min(word_widths)}, max={max(word_widths)}, avg={sum(word_widths)/len(word_widths):.1f}')
print(f'  Height: min={min(word_heights)}, max={max(word_heights)}, avg={sum(word_heights)/len(word_heights):.1f}')

print(f'\nLetter statistics:')
letter_widths = [xmax - xmin for xmin, ymin, xmax, ymax in rects]
letter_heights = [ymax - ymin for xmin, ymin, xmax, ymax in rects]
print(f'  Width: min={min(letter_widths)}, max={max(letter_widths)}, avg={sum(letter_widths)/len(letter_widths):.1f}')
print(f'  Height: min={min(letter_heights)}, max={max(letter_heights)}, avg={sum(letter_heights)/len(letter_heights):.1f}')

# Show last 5 words with their letters
print(f'\nLast 3 words and their letters:')
for i in range(max(0, len(line_words) - 3), len(line_words)):
    word_box = line_words[i]
    print(f'\n  Word {i}: [{word_box[0]}, {word_box[1]}, {word_box[2]}, {word_box[3]}]')

    # Find letters in this word
    word_letters = [r for r in rects if r[0] >= word_box[0] and r[2] <= word_box[2] + 5]
    print(f'    Contains {len(word_letters)} letters:')
    for letter in word_letters[:10]:  # Show up to 10 letters
        w = letter[2] - letter[0]
        h = letter[3] - letter[1]
        print(f'      [{letter[0]}, {letter[1]}, {letter[2]}, {letter[3]}] size={w}x{h}')

print(f'\n✓ Analysis complete!')
