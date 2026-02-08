#!/usr/bin/env python3
"""
Test letter segmentation fix for dots on 'i' and 'j'
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import detection_predictor

from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout
from ocr_reflow.main import find_rects, get_doctr_model
from shapely.geometry import box as shapely_box

def test_letter_segmentation(image_path):
    """Test letter segmentation with the fixed find_rects"""

    print(f"Testing letter segmentation on: {image_path}")
    print("=" * 70)

    # Load and deskew
    img = cv2.imread(image_path)
    print(f"✓ Loaded image: {img.shape}")

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
    print(f"✓ Processing box: ({xmin}, {ymin}) -> ({xmax}, {ymax})")

    region = deskewed_img[ymin:ymax, xmin:xmax].copy()
    region_h, region_w = region.shape[:2]

    # Save region to temp file for doctr
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        # Detect words using doctr (same as main.py)
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        # Convert to pixel coordinates
        words = []
        for i in range(len(words_array)):
            xmin = int(words_array[i, 0] * region_w)
            ymin = int(words_array[i, 1] * region_h)
            xmax = int(words_array[i, 2] * region_w)
            ymax = int(words_array[i, 3] * region_h)
            words.append((xmin, ymin, xmax, ymax))
    finally:
        import os
        os.unlink(tmp_path)

    print(f"✓ Detected {len(words)} words")

    # Extract letters
    letters = find_rects(region, words)
    print(f"✓ Extracted {len(letters)} letters")
    print(f"  Average letters per word: {len(letters) / len(words):.1f}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original region
    axes[0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Text Region ({len(words)} words)')
    axes[0].axis('off')

    # Words
    word_vis = region.copy()
    for xmin, ymin, xmax, ymax in words:
        cv2.rectangle(word_vis, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    axes[1].imshow(cv2.cvtColor(word_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detected Words ({len(words)})')
    axes[1].axis('off')

    # Letters
    letter_vis = region.copy()
    for xmin, ymin, xmax, ymax in letters:
        cv2.rectangle(letter_vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    axes[2].imshow(cv2.cvtColor(letter_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Detected Letters ({len(letters)}) - Fixed')
    axes[2].axis('off')

    plt.tight_layout()
    output_path = '../notebooks/letter_segmentation_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    plt.show()

    # Analyze some words in detail
    print("\n" + "=" * 70)
    print("Detailed analysis of first 3 words:")
    print("=" * 70)

    for word_idx in range(min(3, len(words))):
        wx1, wy1, wx2, wy2 = words[word_idx]

        # Count letters in this word
        letters_in_word = [
            (lx1, ly1, lx2, ly2)
            for lx1, ly1, lx2, ly2 in letters
            if lx1 >= wx1 and lx2 <= wx2 and ly1 >= wy1 and ly2 <= wy2
        ]

        print(f"\nWord {word_idx}: ({wx1}, {wy1}) -> ({wx2}, {wy2})")
        print(f"  Size: {wx2-wx1}x{wy2-wy1}")
        print(f"  Letters: {len(letters_in_word)}")

        if len(letters_in_word) > 0:
            heights = [ly2 - ly1 for _, ly1, _, ly2 in letters_in_word]
            print(f"  Letter heights: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'images/sedg_p598.png'
    test_letter_segmentation(image_path)
