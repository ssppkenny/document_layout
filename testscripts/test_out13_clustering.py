"""
Debug script to understand why out13.png is being split into 2 lines
"""
import numpy as np
import cv2
from doctr.models import detection_predictor
from doctr.io import DocumentFile

# Load image
filename = 'notebooks/out13.png'
img = cv2.imread(filename)
img_h, img_w, _ = img.shape

# Run text detection
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images([filename])
result = model(docs)
words = result[0]["words"]

# Convert to absolute coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

# Calculate median word height
word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
median_height = np.median(word_heights)
height_threshold = median_height * 0.70

print(f"Median height: {median_height}, threshold: {height_threshold}")

# Filter words
filtered_words = []
for xmin, ymin, xmax, ymax, conf in words:
    word_height = ymax - ymin
    if word_height >= height_threshold:
        filtered_words.append((xmin, ymin, xmax, ymax, conf))

print(f"\nFiltered to {len(filtered_words)} words")

# Prepare word data
word_data = []
for i, (xmin, ymin, xmax, ymax, conf) in enumerate(filtered_words):
    center_y = (ymin + ymax) / 2
    word_data.append({
        'id': i,
        'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
        'center_y': center_y, 'height': ymax - ymin
    })

# Sort by X position
word_data.sort(key=lambda w: w['xmin'])

print("\nWords sorted by X:")
for w in word_data:
    print(f"  Word {w['id']}: x=[{w['xmin']}, {w['xmax']}], y=[{w['ymin']}, {w['ymax']}], center_y={w['center_y']:.1f}, h={w['height']}")

# Try clustering
lines = []
used = [False] * len(word_data)

for i, word in enumerate(word_data):
    if used[i]:
        continue

    print(f"\n--- Starting new line with word {word['id']} ---")
    current_line = [word]
    used[i] = True

    # Find all other words that belong to this line
    changed = True
    iteration = 0
    while changed:
        iteration += 1
        print(f"  Iteration {iteration}, current line has {len(current_line)} words")
        changed = False
        for j, other_word in enumerate(word_data):
            if used[j]:
                continue

            # Check if this word overlaps with any word in the current line
            overlaps_any = False
            for line_word in current_line:
                # Calculate vertical overlap
                overlap_top = max(other_word['ymin'], line_word['ymin'])
                overlap_bottom = min(other_word['ymax'], line_word['ymax'])
                overlap = max(0, overlap_bottom - overlap_top)

                min_height = min(other_word['height'], line_word['height'])
                overlap_ratio = overlap / min_height if min_height > 0 else 0

                if overlap >= 0.4 * min_height:
                    print(f"    Word {other_word['id']} overlaps with word {line_word['id']}: overlap={overlap}, min_height={min_height}, ratio={overlap_ratio:.2f}")
                    overlaps_any = True
                    break

            if overlaps_any:
                current_line.append(other_word)
                used[j] = True
                changed = True

    # Sort words in this line by X position
    current_line.sort(key=lambda w: w['xmin'])
    lines.append(current_line)
    print(f"  Final line has {len(current_line)} words: {[w['id'] for w in current_line]}")

print(f"\n=== Total lines detected: {len(lines)} ===")
for i, line in enumerate(lines):
    leftmost = line[0]
    rightmost = line[-1]
    print(f"Line {i}: {len(line)} words, left=({leftmost['xmin']}, {leftmost['center_y']:.1f}), right=({rightmost['xmax']}, {rightmost['center_y']:.1f})")
