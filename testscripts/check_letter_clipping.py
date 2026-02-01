#!/usr/bin/env python3
"""
Final diagnostic: Check if letter regions from original are properly placed in reflowed output
Compare letter boxes between original and reflowed
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from doctr.models import detection_predictor
from doctr.io import DocumentFile
from ocr_reflow.main import find_rects

# Load images
original = cv2.imread('notebooks/out13.png')
reflowed = cv2.imread('out13_reflowed_test.png')

orig_h, orig_w = original.shape[:2]

# Get word detections for line 2 (the longer line)
print('Detecting words in original...')
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images(['notebooks/out13.png'])
result = model(docs)
words = result[0]['words']

# Convert to absolute coordinates
words[:, 0] = (words[:, 0] * orig_w).astype(np.int32)
words[:, 1] = (words[:, 1] * orig_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * orig_w).astype(np.int32)
words[:, 3] = (words[:, 3] * orig_h).astype(np.int32) - 2

line_words = [(int(xmin), int(ymin), int(xmax), int(ymax)) for xmin, ymin, xmax, ymax, _ in words]

# Filter to get only line 2 words (y > 10)
line2_words = [w for w in line_words if w[1] > 10]

print(f'Line 2 has {len(line2_words)} words')

# Show last 3 words
print('\nLast 3 words in original:')
for i in range(max(0, len(line2_words) - 3), len(line2_words)):
    w = line2_words[i]
    print(f'  Word {i}: x=[{w[0]},{w[2]}] y=[{w[1]},{w[3]}] size={w[2]-w[0]}x{w[3]-w[1]}')

# Extract letters from last 3 words
print('\nLast 3 words - letter detection:')
for i in range(max(0, len(line2_words) - 3), len(line2_words)):
    word = line2_words[i]
    letters = find_rects(original, [word])
    print(f'  Word {i} has {len(letters)} letters:')
    for j, (lx, ly, lx2, ly2) in enumerate(letters):
        print(f'    Letter {j}: x=[{lx},{lx2}] y=[{ly},{ly2}] size={lx2-lx}x{ly2-ly}')

        # Extract letter region
        letter_img = original[ly:ly2, lx:lx2].copy()

        # Check if letter has any clipping (touches edges of word box)
        if ly <= word[1] + 1:
            print(f'      ⚠ Letter touches TOP edge of word box')
        if ly2 >= word[3] - 1:
            print(f'      ⚠ Letter touches BOTTOM edge of word box')
        if lx <= word[0] + 1:
            print(f'      ⚠ Letter touches LEFT edge of word box')
        if lx2 >= word[2] - 1:
            print(f'      ⚠ Letter touches RIGHT edge of word box')

print('\n✓ Analysis complete')
print('\nSUMMARY:')
print('- If letters touch the edges of word boxes in the original, they may be clipped')
print('- The padding we added should help, but if word boxes themselves are too small,')
print('  the letters will still be incomplete')
