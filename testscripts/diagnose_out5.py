#!/usr/bin/env python3
"""
Diagnostic script specifically for out5.png to understand the 6-line structure
"""

import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile

# Load image
image_path = "images/out5.png"
img = cv2.imread(image_path)
img_h, img_w, _ = img.shape

print(f"Image dimensions: {img_w} x {img_h}")

# Run text detection
print("\nRunning text detection...")
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

print(f"Total words detected: {len(words)}")

# Calculate median word height
word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
median_height = np.median(word_heights)
print(f"Median word height: {median_height:.1f}px")

# Get center Y for each word
word_centers = []
for i, (xmin, ymin, xmax, ymax, conf) in enumerate(words):
    center_y = (ymin + ymax) / 2
    height = ymax - ymin
    word_centers.append((center_y, i, xmin, ymin, xmax, ymax, height))

# Sort by center Y
word_centers.sort()

# Calculate gaps
gaps = []
for i in range(1, len(word_centers)):
    gap = word_centers[i][0] - word_centers[i-1][0]
    gaps.append(gap)

# Try different percentiles to find which gives 6 lines
print("\n" + "=" * 80)
print("Testing different percentiles:")
print("=" * 80)

for percentile in [75, 80, 83, 85, 87, 88, 89, 90, 91, 92, 95]:
    gap_threshold = np.percentile(gaps, percentile)

    # Apply safety bounds
    min_threshold = median_height * 0.20
    max_threshold = median_height * 0.60
    bounded_threshold = max(min_threshold, min(max_threshold, gap_threshold))

    # Cluster into lines
    lines = []
    current_line = [word_centers[0]]

    for i in range(1, len(word_centers)):
        prev_y = word_centers[i-1][0]
        curr_y = word_centers[i][0]
        y_gap = curr_y - prev_y

        if y_gap > bounded_threshold:
            lines.append(current_line)
            current_line = [word_centers[i]]
        else:
            current_line.append(word_centers[i])

    if current_line:
        lines.append(current_line)

    marker = "✓" if len(lines) == 6 else " "
    print(f"{marker} p{percentile}: threshold={bounded_threshold:.1f}px ({bounded_threshold/median_height:.3f}×) → {len(lines)} lines")

    # Show details for 6 lines
    if len(lines) == 6:
        print(f"\n  ✓ FOUND 6 LINES with percentile {percentile}!")
        for line_idx, line_words in enumerate(lines):
            avg_y = sum(w[0] for w in line_words) / len(line_words)
            word_count = len(line_words)
            min_x = min(w[2] for w in line_words)
            max_x = max(w[4] for w in line_words)
            print(f"  Line {line_idx+1}: y≈{avg_y:.1f}, {word_count} words, x=[{min_x}, {max_x}]")

        # Create visualization
        vis_img = img.copy()

        # Draw word rectangles
        for xmin, ymin, xmax, ymax, _ in words:
            cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

        # Draw lines
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        for line_idx, line_words in enumerate(lines):
            avg_y = int(sum(w[0] for w in line_words) / len(line_words))
            min_x = min(w[2] for w in line_words)
            max_x = max(w[4] for w in line_words)

            color = colors[line_idx % len(colors)]

            # Draw line
            cv2.line(vis_img, (min_x, avg_y), (max_x, avg_y), color, 2)

            # Draw circles at endpoints
            cv2.circle(vis_img, (min_x, avg_y), 5, color, -1)
            cv2.circle(vis_img, (max_x, avg_y), 5, color, -1)

            # Label
            cv2.putText(vis_img, f"L{line_idx+1}", (min_x - 30, avg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = "diagnostic_out5_lines.png"
        cv2.imwrite(output_path, vis_img)
        print(f"\n✓ Visualization saved to: {output_path}")
        break

print("\n" + "=" * 80)
print("Current p90 analysis:")
print("=" * 80)
p90 = np.percentile(gaps, 90)
bounded_p90 = max(median_height * 0.20, min(median_height * 0.60, p90))
print(f"p90 = {p90:.1f}px ({p90/median_height:.3f}×)")
print(f"After bounds: {bounded_p90:.1f}px ({bounded_p90/median_height:.3f}×)")

# Show the gap distribution
print(f"\nGap distribution (sorted):")
sorted_gaps = sorted(gaps)
for i in [0, len(gaps)//4, len(gaps)//2, 3*len(gaps)//4, len(gaps)-1]:
    print(f"  Gap[{i}]: {sorted_gaps[i]:.1f}px")
