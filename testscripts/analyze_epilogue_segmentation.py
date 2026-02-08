#!/usr/bin/env python3
"""
Analyze letter segmentation for the word "Epilogue" in title block
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
from ocr_reflow.main import get_doctr_model, find_rects
import tempfile

def analyze_epilogue_segmentation(image_path='images/jtg_p033.png'):
    """Analyze how 'Epilogue' is being segmented"""

    print("=" * 70)
    print("ANALYZING 'EPILOGUE' SEGMENTATION IN TITLE")
    print("=" * 70)

    # Load image
    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)

    print(f"Image loaded: {img.shape[1]}x{img.shape[0]}")
    print(f"Skew angle: {angle:.2f}°")

    # Get title boxes
    layout_boxes = layout(image_path)
    title_boxes = [b for b, t in layout_boxes if t == 'title']

    if len(title_boxes) == 0:
        print("No title boxes found!")
        return

    print(f"\nFound {len(title_boxes)} title boxes")

    # Process first title box
    box_geom = title_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()

    print(f"Title region: {xmin},{ymin} → {xmax},{ymax} (size: {xmax-xmin}x{ymax-ymin})")

    # Get words from doctr
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        region_h, region_w = region.shape[:2]

        # Use title padding (15px)
        padding = 15
        words = []
        for i in range(len(words_array)):
            wx1 = int(words_array[i, 0] * region_w) - padding
            wy1 = int(words_array[i, 1] * region_h) - padding
            wx2 = int(words_array[i, 2] * region_w) + padding
            wy2 = int(words_array[i, 3] * region_h) + padding
            wx1 = max(0, wx1)
            wy1 = max(0, wy1)
            wx2 = min(region_w, wx2)
            wy2 = min(region_h, wy2)
            words.append((wx1, wy1, wx2, wy2))
    finally:
        import os
        os.unlink(tmp_path)

    print(f"\nDetected {len(words)} words")

    if len(words) == 0:
        print("No words detected!")
        return

    # Assume first/largest word is "Epilogue"
    # Sort by size to find it
    word_sizes = [(i, (w[2]-w[0]) * (w[3]-w[1])) for i, w in enumerate(words)]
    word_sizes.sort(key=lambda x: x[1], reverse=True)

    epilogue_idx = word_sizes[0][0]
    wx1, wy1, wx2, wy2 = words[epilogue_idx]

    print(f"\nAssuming word {epilogue_idx} is 'Epilogue': {wx1},{wy1} → {wx2},{wy2}")
    print(f"  Size: {wx2-wx1}x{wy2-wy1}")

    # Extract letters from this word
    word_img = region[wy1:wy2, wx1:wx2].copy()

    # Show word image
    print(f"\nWord image shape: {word_img.shape}")

    # Run find_rects on just this word
    letters = find_rects(region, [words[epilogue_idx]])

    print(f"\nExtracted {len(letters)} letters (expected ~8 for 'Epilogue')")

    if len(letters) > 8:
        print(f"⚠️  TOO MANY LETTERS! Got {len(letters)}, expected 8")
        print("   This causes the word to look strange")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Original word
    axes[0].imshow(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Word Image (should be "Epilogue" - 8 letters)')
    axes[0].axis('off')

    # With detected letters overlaid
    vis = region.copy()
    cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (255, 0, 0), 3)  # Word box in blue

    for i, (lx1, ly1, lx2, ly2) in enumerate(letters):
        color = (0, 255, 0) if i < 8 else (0, 0, 255)  # Green for first 8, red for extra
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), color, 2)
        cv2.putText(vis, str(i), (lx1+2, ly1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detected Letters: {len(letters)} (Green=expected, Red=extra)')
    axes[1].axis('off')

    # Show binarized image (how find_rects sees it)
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title('Binarized Word (what find_rects processes)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('notebooks/epilogue_segmentation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved analysis to notebooks/epilogue_segmentation_analysis.png")
    plt.close()

    # Analyze the letters in detail
    print(f"\nLetter details:")
    for i, (lx1, ly1, lx2, ly2) in enumerate(letters):
        w = lx2 - lx1
        h = ly2 - ly1
        area = w * h
        print(f"  Letter {i}: x={lx1-wx1:3d}, y={ly1-wy1:3d}, w={w:3d}, h={h:3d}, area={area:5d}")

    # Check for small fragments
    if len(letters) > 0:
        letter_areas = [(lx2-lx1)*(ly2-ly1) for lx1, ly1, lx2, ly2 in letters]
        median_area = np.median(letter_areas)

        print(f"\nArea statistics:")
        print(f"  Median area: {median_area:.0f}")
        print(f"  Min area: {min(letter_areas)}")
        print(f"  Max area: {max(letter_areas)}")

        small_fragments = [i for i, area in enumerate(letter_areas) if area < median_area * 0.3]
        if small_fragments:
            print(f"\n⚠️  Found {len(small_fragments)} small fragments (< 30% median area):")
            for i in small_fragments:
                print(f"     Letter {i}: area={letter_areas[i]}")
            print("  These are likely noise or letter parts that should be merged")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    analyze_epilogue_segmentation()
