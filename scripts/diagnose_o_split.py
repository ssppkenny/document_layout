#!/usr/bin/env python3
"""
Diagnose the ö splitting problem in words like Börja and plötsligt.
"""

import cv2
import numpy as np
import sys

sys.path.insert(0, 'src/ocr_reflow')
from doctr.models import detection_predictor
import torch

def apply_otsu_binarization(img):
    """Apply Otsu binarization to image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert so text is white on black
    binary = cv2.bitwise_not(binary)
    return binary

def analyze_word_segmentation(img, word_idx, label):
    """Analyze how a word with ö is being segmented."""
    # Get word detection
    model = detection_predictor(arch='db_resnet50', pretrained=True, assume_straight_pages=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    result = model([img])

    words = result[0]['words']
    h, w = img.shape[:2]

    if word_idx >= len(words):
        print(f"Word index {word_idx} out of range (total: {len(words)})")
        return

    word = words[word_idx]
    xmin = int(word[0] * w)
    ymin = int(word[1] * h)
    xmax = int(word[2] * w)
    ymax = int(word[3] * h)

    print(f"\n{'='*80}")
    print(f"Analyzing {label} (W{word_idx+1})")
    print(f"Word box: ({xmin},{ymin}) -> ({xmax},{ymax}), size: {xmax-xmin}x{ymax-ymin}")
    print(f"{'='*80}")

    # Extract word region
    word_img = img[ymin:ymax, xmin:xmax].copy()

    # Apply binarization (same as in reflow)
    word_gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, word_binary = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find connected components (same as in find_rects)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word_binary, 8, cv2.CV_32S)

    print(f"\nConnected components: {num_labels-1}")
    print(f"\nComponent details:")

    # Analyze each component
    components_info = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_comp = stats[i, cv2.CC_STAT_WIDTH]
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        components_info.append({
            'id': i,
            'x': x,
            'y': y,
            'w': w_comp,
            'h': h_comp,
            'area': area
        })

    # Sort by x position
    components_info.sort(key=lambda c: c['x'])

    # Print components
    word_height = ymax - ymin
    median_height = np.median([c['h'] for c in components_info]) if components_info else word_height * 0.5

    print(f"\nWord height: {word_height}, Median component height: {median_height:.1f}")
    print(f"\nComponents (sorted left to right):")

    for comp in components_info:
        is_diacritic = (comp['h'] < median_height * 0.5 and
                       comp['w'] < median_height * 1.0 and
                       comp['w'] * comp['h'] < (median_height ** 2) * 0.4)

        comp_type = "DIACRITIC" if is_diacritic else "LETTER"

        print(f"  #{comp['id']:2d} [{comp_type:9s}]: x={comp['x']:3d} y={comp['y']:3d} "
              f"w={comp['w']:2d} h={comp['h']:2d} area={comp['area']:4d}")

    # Visualize
    vis = cv2.cvtColor(word_binary, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for idx, comp in enumerate(components_info):
        color = colors[idx % len(colors)]
        cv2.rectangle(vis, (comp['x'], comp['y']),
                     (comp['x'] + comp['w'], comp['y'] + comp['h']), color, 2)

        is_diacritic = (comp['h'] < median_height * 0.5 and
                       comp['w'] < median_height * 1.0 and
                       comp['w'] * comp['h'] < (median_height ** 2) * 0.4)

        label_text = f"D{comp['id']}" if is_diacritic else f"L{comp['id']}"
        cv2.putText(vis, label_text, (comp['x'], comp['y']-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    output_path = f"/tmp/o_split_w{word_idx+1}.png"
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Visualization saved to {output_path}")

def main():
    img_path = 'images/gang_p023_lines1.png'
    img = cv2.imread(img_path)

    if img is None:
        print(f"ERROR: Could not read {img_path}")
        return

    print(f"Analyzing ö splitting problem in {img_path}")
    print("\nUser reported problems:")
    print("  W51 - för (has split ö)")
    print("  W5 - något (has duplicate å)")
    print("  W?? - behöver (has split ö)")
    print()

    # Analyze the specific problematic words
    print("\n" + "="*80)
    print("ANALYZING PROBLEMATIC WORDS")
    print("="*80)

    analyze_word_segmentation(img, 50, "W51 - för (split ö)")
    analyze_word_segmentation(img, 4, "W5 - något (duplicate å)")

    # Search for "behöver" - it's likely 7-8 letters, around 70-90 pixels wide
    print("\n" + "="*80)
    print("SEARCHING FOR behöver")
    print("="*80)
    for i in range(40, 60):
        try:
            analyze_word_segmentation(img, i, f"W{i+1} - checking for behöver")
        except Exception as e:
            break

if __name__ == '__main__':
    main()
