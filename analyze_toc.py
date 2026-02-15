"""
Analyze TOC page structure for mh_p005.png
"""
import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from ocr_reflow.toc_detection import detect_toc_page, extract_page_number_candidates
import re

# Load image
img = cv2.imread('images/mh_p005.png')
h, w = img.shape[:2]
print(f'Image size: {w}x{h}')

# Run OCR
print('\nRunning OCR...')
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
docs = DocumentFile.from_images(['images/mh_p005.png'])
result = model(docs)

# Get words and their text
words_data = result.export()['pages'][0]['blocks'][0]['lines']
print(f'\nTotal lines detected: {len(words_data)}')

# Extract all words with their text and coordinates
all_words = []
all_texts = []

for line in words_data:
    for word_data in line['words']:
        # Get geometry
        geom = word_data['geometry']
        xmin = int(geom[0][0] * w)
        ymin = int(geom[0][1] * h)
        xmax = int(geom[1][0] * w)
        ymax = int(geom[1][1] * h)

        text = word_data['value']
        all_words.append((xmin, ymin, xmax, ymax))
        all_texts.append(text)

print(f'Total words: {len(all_words)}')

# Group words by lines (vertical position)
line_groups = []
current_line = []
current_y = all_words[0][1] if all_words else 0
line_threshold = 20

for i, (word, text) in enumerate(zip(all_words, all_texts)):
    xmin, ymin, xmax, ymax = word

    if abs(ymin - current_y) > line_threshold and current_line:
        line_groups.append(current_line)
        current_line = []
        current_y = ymin

    current_line.append({
        'text': text,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax
    })

if current_line:
    line_groups.append(current_line)

print(f'\nGrouped into {len(line_groups)} lines:')
print('='*80)

# Analyze each line
for i, line in enumerate(line_groups):
    line_text = ' '.join([w['text'] for w in line])
    last_word = line[-1]['text']
    first_word = line[0]['text']

    # Check if line ends with a number
    is_roman = bool(re.match(r'^[ivxlcdm]+$', last_word.lower()))
    is_arabic = last_word.isdigit()

    # Calculate alignment
    last_xmin = line[-1]['xmin']
    first_xmin = line[0]['xmin']

    marker = ''
    if is_roman:
        marker = ' [ROMAN]'
    elif is_arabic:
        marker = ' [ARABIC]'

    print(f'Line {i:2d}: first_x={first_xmin:4d}, last_x={last_xmin:4d}, last_word="{last_word:6s}"{marker}')
    print(f'        {line_text[:80]}')

# Test TOC detection
print('\n' + '='*80)
print('TOC Detection Test:')
is_toc, confidence, page_numbers = detect_toc_page(all_words, all_texts, w, h, min_page_numbers=3)
print(f'Is TOC: {is_toc}')
print(f'Confidence: {confidence:.2f}')
print(f'Page numbers found: {len(page_numbers)}')
for pn in page_numbers:
    print(f'  - "{pn.text}" (value={pn.value}) at line {pn.line_index}, x={pn.xmin}')

# Analyze alignment of last words
print('\n' + '='*80)
print('Alignment Analysis:')
last_word_positions = [line[-1]['xmin'] for line in line_groups]
median_x = np.median(last_word_positions)
std_x = np.std(last_word_positions)
print(f'Last word X positions: median={median_x:.1f}, std={std_x:.1f}')

# Count aligned words
aligned_count = sum(1 for x in last_word_positions if abs(x - median_x) < 30)
print(f'Words within 30px of median: {aligned_count}/{len(last_word_positions)}')
print(f'Alignment percentage: {100*aligned_count/len(last_word_positions):.1f}%')
