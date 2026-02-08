#!/usr/bin/env python3
"""
Diagnose letter segmentation issues with dots on 'i' and 'j'
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout as detect_layout
from doctr.models import detection_predictor

def diagnose_word_segmentation(image_path, word_idx=0):
    """Diagnose letter segmentation for a specific word"""

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    print(f"Image shape: {img.shape}")

    # Detect skew and correct
    print("Detecting skew...")
    deskewed_img, angle = detect_and_correct_skew(img)
    print(f"Skew angle: {angle:.2f}°")

    # Detect layout
    print("Detecting layout...")
    layout_boxes, _ = detect_layout(str(image_path))  # Returns (boxes, image)

    # Filter plain text boxes
    plain_text_boxes = [box for box, box_type in layout_boxes if box_type == 'plain text']
    print(f"Found {len(plain_text_boxes)} plain text boxes")

    if len(plain_text_boxes) == 0:
        print("No plain text boxes found!")
        return

    # Use first plain text box
    box = plain_text_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box.bounds)
    print(f"Using plain text box: ({xmin}, {ymin}) -> ({xmax}, {ymax})")

    # Extract region
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()

    # Detect words
    print("Detecting words...")
    det_predictor = detection_predictor(arch='db_resnet50', pretrained=True)
    det_result = det_predictor([region])

    words = []
    for page in det_result:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geo = word.geometry
                    words.append((
                        int(geo[0][0] * region.shape[1]),
                        int(geo[0][1] * region.shape[0]),
                        int(geo[1][0] * region.shape[1]),
                        int(geo[1][1] * region.shape[0])
                    ))

    print(f"Found {len(words)} words")

    if word_idx >= len(words):
        print(f"Word index {word_idx} out of range (max: {len(words)-1})")
        word_idx = 0

    # Analyze specific word
    xmin, ymin, xmax, ymax = words[word_idx]
    print(f"\nAnalyzing word {word_idx}: ({xmin}, {ymin}) -> ({xmax}, {ymax})")

    word_height = ymax - ymin
    word_width = xmax - xmin
    print(f"Word dimensions: {word_width}x{word_height}")

    # Extract word region
    word_img = region[ymin:ymax, xmin:xmax, :].copy()

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

    print(f"\nFound {num_labels-1} connected components")

    # Analyze each component
    print("\nComponent analysis:")
    print("ID | X    Y    W    H    Area  | H/WordH | Status")
    print("-" * 65)

    valid_components = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        h_ratio = h / word_height

        # Current filter logic
        is_valid = (w >= 3 and h >= 3 and area >= 9 and h >= word_height * 0.2)

        status = "✓ VALID" if is_valid else "✗ FILTERED"
        if not is_valid:
            reasons = []
            if w < 3: reasons.append("w<3")
            if h < 3: reasons.append("h<3")
            if area < 9: reasons.append("area<9")
            if h < word_height * 0.2: reasons.append(f"h<{word_height*0.2:.1f}")
            status += f" ({', '.join(reasons)})"

        print(f"{i:2d} | {x:4d} {y:4d} {w:4d} {h:4d} {area:6d} | {h_ratio:6.2%} | {status}")

        if is_valid:
            valid_components.append(i)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original word
    axes[0, 0].imshow(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Word {word_idx} - Original')
    axes[0, 0].axis('off')

    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')

    # Binary
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Binary (after threshold)')
    axes[0, 2].axis('off')

    # All components
    colored_labels = np.zeros((*binary.shape, 3), dtype=np.uint8)
    colors = plt.cm.hsv(np.linspace(0, 1, num_labels))
    for i in range(1, num_labels):
        mask = labels == i
        colored_labels[mask] = (colors[i][:3] * 255).astype(np.uint8)

    axes[1, 0].imshow(colored_labels)
    axes[1, 0].set_title(f'All {num_labels-1} Components')
    axes[1, 0].axis('off')

    # Valid components only (current filter)
    valid_mask = np.zeros_like(binary)
    for i in valid_components:
        valid_mask[labels == i] = 255

    axes[1, 1].imshow(valid_mask, cmap='gray')
    axes[1, 1].set_title(f'Valid Components (current: {len(valid_components)})')
    axes[1, 1].axis('off')

    # Better filter - use vertical proximity
    better_components = []
    # Find main body components (tall enough)
    main_components = []
    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h >= word_height * 0.3:  # Main body is at least 30% of word height
            main_components.append(i)

    # Include all components that are vertically close to main components
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Skip tiny noise
        if w < 2 or h < 2 or area < 4:
            continue

        # Check if vertically aligned with any main component
        cy = y + h / 2
        for main_idx in main_components:
            main_y = stats[main_idx, cv2.CC_STAT_TOP]
            main_h = stats[main_idx, cv2.CC_STAT_HEIGHT]
            main_bottom = main_y + main_h

            # Check if this component is above the main component (like a dot)
            # or overlaps with it
            if (y < main_bottom + word_height * 0.3):  # Within reasonable distance above
                better_components.append(i)
                break

    better_mask = np.zeros_like(binary)
    for i in better_components:
        better_mask[labels == i] = 255

    axes[1, 2].imshow(better_mask, cmap='gray')
    axes[1, 2].set_title(f'Better Filter (proximity: {len(better_components)})')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('notebooks/letter_segmentation_diagnosis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to notebooks/letter_segmentation_diagnosis.png")
    plt.show()

if __name__ == '__main__':
    image_path = '../images/sedg_p598.png'
    word_idx = 0  # Change this to analyze different words

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        word_idx = int(sys.argv[2])

    diagnose_word_segmentation(image_path, word_idx)
