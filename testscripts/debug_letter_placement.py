"""
Debug script to trace letter placement for out13.png
"""
import cv2
import numpy as np
from math import ceil
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from operator import itemgetter

filename = 'notebooks/out13.png'
img = cv2.imread(filename)
img_h, img_w, _ = img.shape

model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images([filename])
result = model(docs)
words = result[0]["words"]

words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

line_words = [(int(xmin), int(ymin), int(xmax), int(ymax)) for xmin, ymin, xmax, ymax, _ in words]

# Find letters
all_letters = []
for xmin, ymin, xmax, ymax in line_words:
    r = img[ymin:ymax, xmin:xmax, :].copy()
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r, 8, cv2.CV_32S)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        all_letters.append((x+xmin, y+ymin, x+w+xmin, y+h+ymin))

line_letters = sorted(all_letters, key=itemgetter(0))

# Calculate baseline
heights = [ymax - ymin for xmin, ymin, xmax, ymax in line_letters]
m_height = np.median(heights)
sd = np.std(heights)
normal_letters = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in line_letters if abs((ymax-ymin)-m_height) < sd]
lower_points = [((xmin+xmax)/2, ymax) for xmin, ymin, xmax, ymax in normal_letters]

x_coords = [x for x, y in lower_points]
y_coords = [y for x, y in lower_points]
m, c = np.polyfit(x_coords, y_coords, 1)

print(f'Baseline: y = {m:.6f}*x + {c:.2f}')
print(f'Slope: {m}')

# Simulate reflow with zoom_factor = 2.5
zoom_factor = 2.5
left_margin = 50
available_width = 2000 - 50 - 50  # new_page_width - left - right margins

# Calculate baseline shift for each letter
letters_with_bl = []
for xmin, ymin, xmax, ymax in line_letters:
    x_center = (xmin + xmax) / 2
    expected_baseline_y = m * x_center + c
    bl = ymax - ceil(expected_baseline_y)

    scaled_width = int((xmax - xmin) * zoom_factor)
    scaled_height = int((ymax - ymin) * zoom_factor)
    scaled_bl = int(bl * zoom_factor)

    letters_with_bl.append({
        'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
        'bl': bl, 'baseline_slope': m,
        'scaled_width': scaled_width,
        'scaled_height': scaled_height,
        'scaled_bl': scaled_bl
    })

print(f'\nTotal letters: {len(letters_with_bl)}')

# Check for abnormal dimensions
print(f'\nChecking for abnormal letter dimensions:')
for i, item in enumerate(letters_with_bl):
    if item['scaled_width'] < 5 or item['scaled_height'] < 5:
        print(f'  Letter {i}: width={item["scaled_width"]}, height={item["scaled_height"]} - TOO SMALL!')
    if item['scaled_width'] > 100:
        print(f'  Letter {i}: width={item["scaled_width"]} - TOO WIDE!')

# Calculate line height as the reflow code does
max_above = max(item['scaled_height'] - item['scaled_bl'] for item in letters_with_bl)
max_below = max(item['scaled_bl'] for item in letters_with_bl)
scaled_baseline_slope = m * zoom_factor
max_baseline_shift = abs(scaled_baseline_slope * available_width)

print(f'\\nLine height calculation:')
print(f'  Max above baseline: {max_above}')
print(f'  Max below baseline: {max_below}')
print(f'  Scaled baseline slope: {scaled_baseline_slope}')
print(f'  Max baseline shift over {available_width}px width: {max_baseline_shift:.2f}')

line_height = int((max_above + max_below + max_baseline_shift) * 1.2 + 10)
print(f'  Calculated line height: {line_height}')

# Simulate placement
current_y = 50  # top_margin
max_above_baseline_for_line = max(item['scaled_height'] - item['scaled_bl'] for item in letters_with_bl)
baseline_y_at_start = current_y + max_above_baseline_for_line

print(f'\\nPlacement simulation:')
print(f'  current_y (top of line area): {current_y}')
print(f'  max_above_baseline_for_line: {max_above_baseline_for_line}')
print(f'  baseline_y_at_start: {baseline_y_at_start}')

# Check first, middle, and last letters
test_indices = [0, len(letters_with_bl)//2, -1]
current_x = left_margin

for idx in test_indices:
    item = letters_with_bl[idx]

    # Skip to approximate position
    if idx == len(letters_with_bl)//2:
        current_x = left_margin + 800
    elif idx == -1:
        current_x = left_margin + 1500

    baseline_y_here = int(baseline_y_at_start + scaled_baseline_slope * (current_x - left_margin))
    y_offset = int(baseline_y_here - item['scaled_height'] + item['scaled_bl'])

    y_start = max(0, y_offset)
    y_end = y_offset + item['scaled_height']

    clipped = (y_end - y_start) != item['scaled_height']

    print(f'\n  Letter {idx} (original x={item["xmin"]}):')
    print(f'    Placed at X={current_x}')
    print(f'    baseline_y_here: {baseline_y_here}')
    print(f'    y_offset (top): {y_offset}')
    print(f'    y_start: {y_start}, y_end: {y_end}')
    print(f'    Height: {item["scaled_height"]}, actual: {y_end - y_start}')
    print(f'    CLIPPED: {clipped}' if clipped else f'    OK')
