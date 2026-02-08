#!/usr/bin/env python3
"""
Visualize how doctr detects words in the title to understand the over-segmentation
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

def visualize_word_detection(image_path='images/jtg_p033.png'):
    """Show how doctr breaks up the title text"""

    print("=" * 70)
    print("VISUALIZING DOCTR WORD DETECTION IN TITLE")
    print("=" * 70)

    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)

    # Get title box
    layout_boxes = layout(image_path)
    title_boxes = [b for b, t in layout_boxes if t == 'title']

    if len(title_boxes) == 0:
        return

    box_geom = title_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()
    region_h, region_w = region.shape[:2]

    print(f"Title region: {region_w}x{region_h}")

    # Get words WITHOUT padding first
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        words_no_padding = []
        for i in range(len(words_array)):
            wx1 = int(words_array[i, 0] * region_w)
            wy1 = int(words_array[i, 1] * region_h)
            wx2 = int(words_array[i, 2] * region_w)
            wy2 = int(words_array[i, 3] * region_h)
            words_no_padding.append((wx1, wy1, wx2, wy2))
    finally:
        import os
        os.unlink(tmp_path)

    print(f"\nDoctr detected {len(words_no_padding)} words in title")
    print("This explains the over-segmentation!")
    print("'Epilogue' (1 word, 8 letters) is being detected as multiple words")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # Original title region
    vis1 = region.copy()
    for i, (wx1, wy1, wx2, wy2) in enumerate(words_no_padding):
        cv2.rectangle(vis1, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)
        cv2.putText(vis1, f"W{i}", (wx1+5, wy1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    axes[0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Doctr Word Detection: {len(words_no_padding)} words (should be 1 word "Epilogue")')
    axes[0].axis('off')

    # Show word gaps
    if len(words_no_padding) > 1:
        word_centers = [(wx1 + wx2) / 2 for wx1, wy1, wx2, wy2 in words_no_padding]
        gaps = [word_centers[i+1] - word_centers[i] for i in range(len(word_centers)-1)]

        axes[1].bar(range(len(gaps)), gaps, color='red', alpha=0.7)
        axes[1].axhline(y=np.mean(gaps), color='blue', linestyle='--', label=f'Mean gap: {np.mean(gaps):.1f}')
        axes[1].set_xlabel('Between words')
        axes[1].set_ylabel('Gap (pixels)')
        axes[1].set_title('Gaps Between Detected Words')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        print(f"\nWord gaps: {gaps}")
        print(f"Mean gap: {np.mean(gaps):.1f} pixels")
        print(f"\nSOLUTION: Need to merge nearby word boxes in title text")
        print(f"         Words with gap < threshold should be merged before letter extraction")

    plt.tight_layout()
    plt.savefig('notebooks/doctr_word_detection.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to notebooks/doctr_word_detection.png")
    plt.close()

    print("=" * 70)

if __name__ == '__main__':
    visualize_word_detection()
