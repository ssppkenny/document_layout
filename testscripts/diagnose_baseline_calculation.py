"""
Detailed diagnostic for out13.png baseline calculation
"""
import cv2
import numpy as np
from math import ceil
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from operator import itemgetter
import matplotlib.pyplot as plt

filename = 'notebooks/out13.png'
img = cv2.imread(filename)
img_h, img_w, _ = img.shape

print(f"Processing: {filename}")
print(f"Image size: {img_w}x{img_h}")

# Run detection
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images([filename])
result = model(docs)
words = result[0]["words"]

# Convert to absolute coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
words = words.astype(np.int32)

print(f"\n=== STEP 1: Word Detection ===")
print(f"Detected {len(words)} words")

# Simulate find_rects to get individual letters
line_words = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax, _ in words]

# Find connected components for each word
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

print(f"\n=== STEP 2: Letter Segmentation ===")
print(f"Found {len(line_letters)} individual letters")

# Filter to normal height letters
heights = [ymax - ymin for xmin, ymin, xmax, ymax in line_letters]
m_height = np.median(heights)
sd = np.std(heights)

print(f"Median height: {m_height:.1f}")
print(f"Std deviation: {sd:.1f}")

normal_letters = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in line_letters
                  if abs((ymax-ymin)-m_height) < sd]

print(f"Normal letters (within 1 SD): {len(normal_letters)}")

# Get bottom points
lower_points = [((xmin+xmax)/2, ymax) for xmin, ymin, xmax, ymax in normal_letters]

print(f"\n=== STEP 3: Baseline Fitting ===")
print(f"Fitting line to {len(lower_points)} bottom points")

if len(lower_points) >= 2:
    x_coords = [x for x, y in lower_points]
    y_coords = [y for x, y in lower_points]
    m, c = np.polyfit(x_coords, y_coords, 1)

    print(f"Fitted baseline: y = {m:.6f}*x + {c:.2f}")
    print(f"Slope: {m:.6f}")
    print(f"At x=0: y={c:.2f}")
    print(f"At x={img_w}: y={m*img_w + c:.2f}")
    print(f"Y difference over image width: {m*img_w:.2f} pixels")

    # Check baseline at left and right edges
    left_x = min(x_coords)
    right_x = max(x_coords)
    left_baseline_y = m * left_x + c
    right_baseline_y = m * right_x + c

    print(f"\nBaseline at leftmost letter (x={left_x:.0f}): y={left_baseline_y:.2f}")
    print(f"Baseline at rightmost letter (x={right_x:.0f}): y={right_baseline_y:.2f}")
    print(f"Baseline Y difference: {right_baseline_y - left_baseline_y:.2f} pixels")
    print(f"Baseline X span: {right_x - left_x:.0f} pixels")
    print(f"Baseline slope over span: {(right_baseline_y - left_baseline_y)/(right_x - left_x):.6f}")

    # Calculate bl (baseline shift) for each letter
    print(f"\n=== STEP 4: Baseline Shift Calculation ===")
    print("First 5 letters:")
    for i, (xmin, ymin, xmax, ymax) in enumerate(line_letters[:5]):
        x_center = (xmin + xmax) / 2
        expected_baseline_y = m * x_center + c
        bl = ymax - ceil(expected_baseline_y)
        print(f"  Letter {i}: x=[{xmin},{xmax}], ymax={ymax}, "
              f"expected_baseline={expected_baseline_y:.2f}, bl={bl}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

    # Top: Original image with detected baseline
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)

    # Draw all letters
    for xmin, ymin, xmax, ymax in line_letters:
        ax1.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                     fill=False, edgecolor='red', linewidth=1, alpha=0.5))

    # Draw normal letters (used for baseline)
    for xmin, ymin, xmax, ymax in normal_letters:
        ax1.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                     fill=False, edgecolor='blue', linewidth=2))

    # Draw bottom points
    for x, y in lower_points:
        ax1.plot(x, y, 'go', markersize=4)

    # Draw fitted baseline
    x_line = [0, img_w]
    y_line = [c, m * img_w + c]
    ax1.plot(x_line, y_line, 'g-', linewidth=3, label=f'Fitted baseline (slope={m:.6f})')

    ax1.legend()
    ax1.set_title('Baseline Detection')
    ax1.axis('tight')

    # Bottom: Show baseline deviations
    ax2.set_xlim(0, img_w)
    ax2.set_ylim(-10, 10)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Deviation from baseline (pixels)')
    ax2.set_title('Letter Bottom Deviation from Fitted Baseline')
    ax2.grid(True, alpha=0.3)

    # Plot deviations
    for xmin, ymin, xmax, ymax in line_letters:
        x_center = (xmin + xmax) / 2
        expected_y = m * x_center + c
        deviation = ymax - expected_y
        ax2.plot(x_center, deviation, 'ro', markersize=3)

    plt.tight_layout()
    plt.savefig('out13_baseline_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic visualization saved to: out13_baseline_diagnostic.png")

else:
    print("Not enough points for baseline fitting")
