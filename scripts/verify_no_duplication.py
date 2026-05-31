#!/usr/bin/env python3
"""
Verify that Swedish words don't get duplicated characters
Test specifically for Börja → Böörja and inför → inföör issues
"""

import cv2
import numpy as np

print("="*80)
print("VERIFYING SWEDISH CHARACTER DUPLICATION FIX")
print("="*80)
print()

# Run reflow
print("Running reflow on images/gang_p023.png...")
import subprocess
result = subprocess.run(
    ["python", "src/ocr_reflow/main.py", "images/gang_p023.png", "--layout"],
    capture_output=True,
    text=True,
    timeout=60
)

if result.returncode != 0:
    print("❌ ERROR: Reflow failed")
    print(result.stderr)
    exit(1)

print("✅ Reflow completed successfully\n")

# Now analyze the output
# We can't easily OCR the output, but we can check for visual duplication
# by analyzing character spacing and component counts

img = cv2.imread('output_reflowed.png')
if img is None:
    print("❌ ERROR: Could not load output_reflowed.png")
    exit(1)

print(f"Output image size: {img.shape[1]}x{img.shape[0]}")

# Convert to binary
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
binary_inv = 255 - binary

# Count connected components (rough character count)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inv, 8, cv2.CV_32S)

# Filter out very small components (noise)
valid_components = []
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    if area > 10 and w > 2 and h > 2:
        valid_components.append(i)

print(f"Found {len(valid_components)} text components in output\n")

# Analyze first few lines for suspicious patterns
# Extract top portion of page
top_section = img[50:500, :]
gray_top = cv2.cvtColor(top_section, cv2.COLOR_BGR2GRAY)
_, binary_top = cv2.threshold(gray_top, 200, 255, cv2.THRESH_BINARY)
binary_inv_top = 255 - binary_top

# Find components in top section
num_labels_top, labels_top, stats_top, centroids_top = cv2.connectedComponentsWithStats(binary_inv_top, 8, cv2.CV_32S)

# Look for suspicious very close horizontal pairs (might indicate duplication)
print("Analyzing for suspicious close character pairs...")
suspicious_pairs = []

for i in range(1, num_labels_top):
    if stats_top[i, cv2.CC_STAT_AREA] < 10:
        continue

    x1 = stats_top[i, cv2.CC_STAT_LEFT]
    y1 = stats_top[i, cv2.CC_STAT_TOP]
    w1 = stats_top[i, cv2.CC_STAT_WIDTH]
    h1 = stats_top[i, cv2.CC_STAT_HEIGHT]

    for j in range(i+1, num_labels_top):
        if stats_top[j, cv2.CC_STAT_AREA] < 10:
            continue

        x2 = stats_top[j, cv2.CC_STAT_LEFT]
        y2 = stats_top[j, cv2.CC_STAT_TOP]
        w2 = stats_top[j, cv2.CC_STAT_WIDTH]
        h2 = stats_top[j, cv2.CC_STAT_HEIGHT]

        # Check if at similar y position (same line)
        y_diff = abs((y1 + h1/2) - (y2 + h2/2))
        if y_diff < min(h1, h2) * 0.5:
            # Check horizontal distance
            horizontal_gap = abs((x1 + w1) - x2) if x1 < x2 else abs((x2 + w2) - x1)

            # Suspiciously close if gap is very small relative to character width
            if horizontal_gap < min(w1, w2) * 0.3:
                suspicious_pairs.append((i, j, horizontal_gap, x1, x2, y1, y2))

if suspicious_pairs:
    print(f"⚠️  WARNING: Found {len(suspicious_pairs)} suspiciously close character pairs")
    print("   (This might indicate duplication, but could also be normal ligatures/kerning)")
    for idx, (i, j, gap, x1, x2, y1, y2) in enumerate(suspicious_pairs[:5]):
        print(f"   Pair {idx+1}: components {i} and {j}, gap={gap:.1f}px at positions x=({x1}, {x2}), y=({y1}, {y2})")
else:
    print("✅ No suspiciously close character pairs found")

print()
print("="*80)
print("VISUAL INSPECTION REQUIRED")
print("="*80)
print()
print("Please manually check output_reflowed.png for:")
print("  1. Swedish word 'Börja' should NOT appear as 'Böörja'")
print("  2. Swedish word 'inför' should NOT appear as 'inföör'")
print("  3. Letters ä, ö, å should appear complete and not duplicated")
print()
print("If you see correct Swedish characters without duplication:")
print("  ✅ THE FIX IS WORKING!")
print()
print("If you still see doubled characters (Böörja, inföör):")
print("  ❌ THE BUG IS NOT FULLY FIXED")
print()
