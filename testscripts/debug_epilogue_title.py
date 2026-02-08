#!/usr/bin/env python3
"""
Debug the Epilogue title block - show binarized image and connected components
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout

def debug_epilogue_title(image_path='images/jtg_p033.png'):
    """Debug the Epilogue title processing"""

    print("=" * 70)
    print("DEBUGGING EPILOGUE TITLE BLOCK")
    print("=" * 70)

    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)

    print(f"Skew angle: {angle:.2f}°")

    # Get title boxes
    layout_boxes = layout(image_path)
    title_boxes = [(b, t) for b, t in layout_boxes if t == 'title']
    title_boxes_sorted = sorted(title_boxes, key=lambda item: item[0].bounds[1])

    # Get the one with higher Y (Epilogue)
    box_geom, _ = title_boxes_sorted[-1]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)

    print(f"Epilogue title at y={ymin}: ({xmin}, {ymin}) to ({xmax}, {ymax})")

    # Extract from ORIGINAL (not deskewed) first
    region_original = img[ymin:ymax, xmin:xmax].copy()
    region_h, region_w = region_original.shape[:2]

    print(f"Region size: {region_w}x{region_h}")

    # Extract from DESKEWED
    region_deskewed = deskewed_img[ymin:ymax, xmin:xmax].copy()

    # Binarize both
    gray_orig = cv2.cvtColor(region_original, cv2.COLOR_BGR2GRAY)
    _, binary_orig = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    gray_desk = cv2.cvtColor(region_deskewed, cv2.COLOR_BGR2GRAY)
    _, binary_desk = cv2.threshold(gray_desk, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components on ORIGINAL
    num_labels_orig, labels_orig, stats_orig, centroids_orig = cv2.connectedComponentsWithStats(binary_orig, 8, cv2.CV_32S)

    # Connected components on DESKEWED
    num_labels_desk, labels_desk, stats_desk, centroids_desk = cv2.connectedComponentsWithStats(binary_desk, 8, cv2.CV_32S)

    print(f"\nConnected components:")
    print(f"  Original: {num_labels_orig - 1}")
    print(f"  Deskewed: {num_labels_desk - 1}")

    # Show main components
    print(f"\nOriginal - main components:")
    for i in range(1, min(num_labels_orig, 11)):
        x = stats_orig[i, cv2.CC_STAT_LEFT]
        y = stats_orig[i, cv2.CC_STAT_TOP]
        w = stats_orig[i, cv2.CC_STAT_WIDTH]
        h = stats_orig[i, cv2.CC_STAT_HEIGHT]
        area = stats_orig[i, cv2.CC_STAT_AREA]
        if h >= region_h * 0.3:  # Main components
            print(f"  Comp {i}: x={x:3d}, y={y:3d}, w={w:3d}, h={h:3d}, area={area:5d}")

    # Visualize
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Original color
    axes[0, 0].imshow(cv2.cvtColor(region_original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Title Region')
    axes[0, 0].axis('off')

    # Deskewed color
    axes[0, 1].imshow(cv2.cvtColor(region_deskewed, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Deskewed (angle={angle:.2f}°)')
    axes[0, 1].axis('off')

    # Original binary
    axes[1, 0].imshow(binary_orig, cmap='gray')
    axes[1, 0].set_title(f'Original Binary ({num_labels_orig-1} components)')
    axes[1, 0].axis('off')

    # Deskewed binary
    axes[1, 1].imshow(binary_desk, cmap='gray')
    axes[1, 1].set_title(f'Deskewed Binary ({num_labels_desk-1} components)')
    axes[1, 1].axis('off')

    # Original with component boxes
    vis_orig = region_original.copy()
    for i in range(1, num_labels_orig):
        x = stats_orig[i, cv2.CC_STAT_LEFT]
        y = stats_orig[i, cv2.CC_STAT_TOP]
        w = stats_orig[i, cv2.CC_STAT_WIDTH]
        h = stats_orig[i, cv2.CC_STAT_HEIGHT]
        if w >= 5 and h >= 10:  # Filter small noise
            cv2.rectangle(vis_orig, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(vis_orig, str(i), (x+2, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    axes[2, 0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Original with Components')
    axes[2, 0].axis('off')

    # Deskewed with component boxes
    vis_desk = region_deskewed.copy()
    for i in range(1, num_labels_desk):
        x = stats_desk[i, cv2.CC_STAT_LEFT]
        y = stats_desk[i, cv2.CC_STAT_TOP]
        w = stats_desk[i, cv2.CC_STAT_WIDTH]
        h = stats_desk[i, cv2.CC_STAT_HEIGHT]
        if w >= 5 and h >= 10:
            cv2.rectangle(vis_desk, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(vis_desk, str(i), (x+2, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    axes[2, 1].imshow(cv2.cvtColor(vis_desk, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('Deskewed with Components')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig('notebooks/epilogue_debug_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to notebooks/epilogue_debug_comparison.png")
    plt.close()

    print("=" * 70)

if __name__ == '__main__':
    debug_epilogue_title()
