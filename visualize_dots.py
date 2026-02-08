#!/usr/bin/env python3
"""
Visualize letter segmentation for a specific word to check dots on 'i' and 'j'
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

def visualize_word_with_dots(image_path, word_text_to_find=''):
    """Find and visualize a word containing 'i' or 'j' to check dot preservation"""

    print(f"Analyzing: {image_path}")
    print("=" * 70)

    # Load and deskew
    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)
    print(f"✓ Detected skew: {angle:.2f}°")

    # Get layout
    layout_boxes = layout(image_path)
    plain_text_boxes = [b for b, t in layout_boxes if t == 'plain text']
    print(f"✓ Found {len(plain_text_boxes)} plain text boxes")

    if len(plain_text_boxes) == 0:
        print("No plain text boxes found!")
        return

    # Process first plain text box
    box_geom = plain_text_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()
    region_h, region_w = region.shape[:2]

    # Save region to temp file for doctr
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        # Detect words
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        # Convert to pixel coordinates
        words = []
        for i in range(len(words_array)):
            wx1 = int(words_array[i, 0] * region_w)
            wy1 = int(words_array[i, 1] * region_h)
            wx2 = int(words_array[i, 2] * region_w)
            wy2 = int(words_array[i, 3] * region_h)
            words.append((wx1, wy1, wx2, wy2))
    finally:
        import os
        os.unlink(tmp_path)

    print(f"✓ Detected {len(words)} words")

    # Find words with 'i' or 'j' by analyzing their letter segmentation
    # Look for words that have very short components (dots)
    target_words = []

    for word_idx, (wx1, wy1, wx2, wy2) in enumerate(words):
        word_height = wy2 - wy1
        word_width = wx2 - wx1
        word_img = region[wy1:wy2, wx1:wx2, :].copy()

        # Convert to binary
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

        # Check if there are small components (potential dots)
        has_small_component = False
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if h < word_height * 0.3:  # Small component
                has_small_component = True
                break

        if has_small_component and num_labels > 3:  # Has dots and multiple components
            target_words.append(word_idx)

    print(f"✓ Found {len(target_words)} words with potential dots (i, j, etc.)")

    # Visualize first few interesting words
    n_show = min(5, len(target_words))

    if n_show == 0:
        print("No words with dots found in this region")
        return

    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    from ocr_reflow.main import find_rects

    for plot_idx, word_idx in enumerate(target_words[:n_show]):
        wx1, wy1, wx2, wy2 = words[word_idx]
        word_height = wy2 - wy1
        word_width = wx2 - wx1
        word_img = region[wy1:wy2, wx1:wx2, :].copy()

        # Get letters for this word only
        letters = find_rects(region, [words[word_idx]])

        # Adjust letter coordinates to be relative to word
        letters_rel = [(lx1-wx1, ly1-wy1, lx2-wx1, ly2-wy1) for lx1, ly1, lx2, ly2 in letters]

        # Convert to binary
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

        # Original word
        axes[plot_idx, 0].imshow(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
        axes[plot_idx, 0].set_title(f'Word {word_idx} ({word_width}x{word_height})')
        axes[plot_idx, 0].axis('off')

        # Binary
        axes[plot_idx, 1].imshow(binary, cmap='gray')
        axes[plot_idx, 1].set_title(f'{num_labels-1} components')
        axes[plot_idx, 1].axis('off')

        # Segmented letters (current)
        letter_vis = word_img.copy()
        for lx1, ly1, lx2, ly2 in letters_rel:
            cv2.rectangle(letter_vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)

        axes[plot_idx, 2].imshow(cv2.cvtColor(letter_vis, cv2.COLOR_BGR2RGB))
        axes[plot_idx, 2].set_title(f'{len(letters)} letters (FIXED)')
        axes[plot_idx, 2].axis('off')

        # Component info
        info_text = f"Components:\n"
        main_count = 0
        dot_count = 0
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            if h >= word_height * 0.3:
                main_count += 1
                info_text += f"  Main: {w}x{h}\n"
            else:
                dot_count += 1
                info_text += f"  Dot: {w}x{h}\n"

        info_text += f"\nMain: {main_count}, Dots: {dot_count}"
        info_text += f"\nLetters: {len(letters)}"

        axes[plot_idx, 3].text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                              verticalalignment='center')
        axes[plot_idx, 3].set_title('Analysis')
        axes[plot_idx, 3].axis('off')

    plt.tight_layout()
    output_path = 'notebooks/dots_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved detailed analysis to {output_path}")
    plt.close()

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'images/sedg_p598.png'
    visualize_word_with_dots(image_path)
