#!/usr/bin/env python3
"""
Debug find_rects to understand what's happening with component detection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout
from ocr_reflow.main import get_doctr_model
import tempfile

def debug_find_rects(image_path='images/jtg_p033.png'):
    """Debug the find_rects function step by step"""

    print("=" * 70)
    print("DEBUGGING FIND_RECTS FOR EPILOGUE")
    print("=" * 70)

    # Load and prepare image
    img = cv2.imread(image_path)
    deskewed_img, _ = detect_and_correct_skew(img)

    # Get title box
    layout_boxes = layout(image_path)
    title_boxes = [b for b, t in layout_boxes if t == 'title']

    if len(title_boxes) == 0:
        return

    box_geom = title_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()

    # Get words
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        region_h, region_w = region.shape[:2]
        padding = 15
        words = []
        for i in range(len(words_array)):
            wx1 = int(words_array[i, 0] * region_w) - padding
            wy1 = int(words_array[i, 1] * region_h) - padding
            wx2 = int(words_array[i, 2] * region_w) + padding
            wy2 = int(words_array[i, 3] * region_h) + padding
            words.append((max(0, wx1), max(0, wy1), min(region_w, wx2), min(region_h, wy2)))
    finally:
        import os
        os.unlink(tmp_path)

    # Find the largest word (Epilogue)
    word_sizes = [(i, (w[2]-w[0]) * (w[3]-w[1])) for i, w in enumerate(words)]
    word_sizes.sort(key=lambda x: x[1], reverse=True)
    epilogue_idx = word_sizes[0][0]

    xmin, ymin, xmax, ymax = words[epilogue_idx]
    word_height = ymax - ymin
    word_width = xmax - xmin

    print(f"Word box: {xmin},{ymin} → {xmax},{ymax}")
    print(f"Word size: {word_width}x{word_height}")

    # Manual find_rects logic with debugging
    r = region[ymin:ymax, xmin:xmax, :].copy()
    r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    _, r_binary = cv2.threshold(r_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r_binary, 8, cv2.CV_32S)

    print(f"\nConnected components: {num_labels - 1} (excluding background)")

    # Show all components
    print("\nAll components:")
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        print(f"  Comp {i}: x={x:3d}, y={y:3d}, w={w:3d}, h={h:3d}, area={area:5d}")

    # Check filtering
    main_components = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if w < 2 or h < 2 or area < 4:
            continue

        if h >= word_height * 0.3:
            main_components.append(i)

    print(f"\nMain components (h >= {word_height * 0.3:.1f}): {len(main_components)}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original word
    axes[0, 0].imshow(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Word Image: "Epilogue"')
    axes[0, 0].axis('off')

    # Binarized
    axes[0, 1].imshow(r_binary, cmap='gray')
    axes[0, 1].set_title('Binarized')
    axes[0, 1].axis('off')

    # All components
    vis_all = r.copy()
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(vis_all, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(vis_all, str(i), (x+2, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    axes[1, 0].imshow(cv2.cvtColor(vis_all, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'All Components: {num_labels-1}')
    axes[1, 0].axis('off')

    # Main components only
    vis_main = r.copy()
    for i in main_components:
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(vis_main, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(vis_main, str(i), (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    axes[1, 1].imshow(cv2.cvtColor(vis_main, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Main Components: {len(main_components)}')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('notebooks/debug_find_rects.png', dpi=150)
    print(f"\n✓ Saved to notebooks/debug_find_rects.png")
    plt.close()

    print("=" * 70)

if __name__ == '__main__':
    debug_find_rects()
