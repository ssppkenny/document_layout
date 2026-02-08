#!/usr/bin/env python3
"""
Diagnose why Epilogue title appears rotated
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ocr_reflow.layout import layout

def diagnose_title_rotation(image_path='images/jtg_p033.png'):
    """Check if title region itself is rotated in the source image"""

    print("=" * 70)
    print("DIAGNOSING TITLE ROTATION")
    print("=" * 70)

    # Load original image
    img = cv2.imread(image_path)
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Get layout boxes
    layout_boxes = layout(image_path)
    title_boxes = [(b, t) for b, t in layout_boxes if t == 'title']

    # Sort by Y
    title_boxes_sorted = sorted(title_boxes, key=lambda item: item[0].bounds[1])

    print(f"\nFound {len(title_boxes_sorted)} title boxes")

    if len(title_boxes_sorted) < 2:
        print("Need at least 2 title boxes")
        return

    # Get the Epilogue title (second/last one)
    epilogue_box, _ = title_boxes_sorted[-1]
    xmin, ymin, xmax, ymax = map(int, epilogue_box.bounds)

    print(f"\nEpilogue title box: ({xmin}, {ymin}) → ({xmax}, {ymax})")
    print(f"  Size: {xmax-xmin}x{ymax-ymin}")

    # Extract the region
    title_region = img[ymin:ymax, xmin:xmax].copy()

    # Check if the extracted region looks rotated
    # Convert to grayscale
    gray = cv2.cvtColor(title_region, cv2.COLOR_BGR2GRAY)

    # Detect edges to see text orientation
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough lines to detect dominant orientation
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        if angles:
            median_angle = np.median(angles)
            print(f"\n  Detected text orientation: {median_angle:.2f}°")
            if abs(median_angle) > 2:
                print(f"  ⚠️  Title region IS ROTATED by ~{median_angle:.1f}°!")
            else:
                print(f"  ✓ Title region appears horizontal")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original full page with title box marked
    vis_full = img.copy()
    cv2.rectangle(vis_full, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    cv2.putText(vis_full, "Epilogue Title", (xmin, ymin-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    axes[0, 0].imshow(cv2.cvtColor(vis_full, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Full Page with Title Box')
    axes[0, 0].axis('off')

    # Extracted title region
    axes[0, 1].imshow(cv2.cvtColor(title_region, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Extracted Title Region')
    axes[0, 1].axis('off')

    # Grayscale
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title('Grayscale')
    axes[1, 0].axis('off')

    # Edges with detected lines
    vis_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axes[1, 1].imshow(vis_lines)
    axes[1, 1].set_title(f'Edge Detection & Lines (angle: {median_angle:.1f}° if detected)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('notebooks/title_rotation_diagnosis.png', dpi=150)
    print(f"\n✓ Saved to notebooks/title_rotation_diagnosis.png")
    plt.close()

    print("=" * 70)

if __name__ == '__main__':
    diagnose_title_rotation()
