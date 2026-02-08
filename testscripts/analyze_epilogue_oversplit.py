#!/usr/bin/env python3
"""
Detailed analysis of Epilogue segmentation to see if letters are being over-split
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

def analyze_epilogue_oversplit(image_path='images/jtg_p033.png'):
    """Check if Epilogue is being over-split"""

    print("=" * 70)
    print("ANALYZING EPILOGUE OVER-SPLITTING")
    print("=" * 70)

    # Load and prepare
    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)

    print(f"Skew angle: {angle:.2f}°")

    # Get title boxes
    layout_boxes = layout(image_path)
    title_boxes = [b for b, t in layout_boxes if t == 'title']

    if len(title_boxes) == 0:
        print("No title boxes found!")
        return

    print(f"Found {len(title_boxes)} title boxes")

    # Process first title (should contain Epilogue)
    box_geom = title_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()

    print(f"Title region: {xmax-xmin}x{ymax-ymin}")

    # Get words with title padding (15px)
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

    # Find largest word (likely Epilogue)
    word_sizes = [(i, (w[2]-w[0]) * (w[3]-w[1])) for i, w in enumerate(words)]
    word_sizes.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 largest words:")
    for i, (idx, size) in enumerate(word_sizes[:3]):
        wx1, wy1, wx2, wy2 = words[idx]
        print(f"  {i+1}. Word {idx}: {wx2-wx1}x{wy2-wy1} (area={size})")

    # Process the largest word
    epilogue_idx = word_sizes[0][0]

    # Extract letters using find_rects
    print(f"\n{'='*70}")
    print(f"Analyzing word {epilogue_idx} (assuming it's 'Epilogue')")
    print(f"{'='*70}")

    letters = find_rects(region, [words[epilogue_idx]])

    print(f"\nExtracted {len(letters)} letter components")
    print(f"Expected: ~8 letters for 'Epilogue'")

    if len(letters) > 12:
        print(f"\n⚠️  SEVERE OVER-SPLITTING!")
        print(f"   Got {len(letters)} components, expected 8")
        print(f"   Likely individual letters are being split into parts")
    elif len(letters) > 10:
        print(f"\n⚠️  MODERATE OVER-SPLITTING")
        print(f"   Got {len(letters)} components, expected 8")
    elif len(letters) < 6:
        print(f"\n⚠️  UNDER-SEGMENTATION")
        print(f"   Got {len(letters)} components, expected 8")
    else:
        print(f"\n✓ Reasonable segmentation")

    # Analyze letter widths and positions
    wx1, wy1, wx2, wy2 = words[epilogue_idx]
    word_width = wx2 - wx1
    word_height = wy2 - wy1

    letter_widths = []
    letter_heights = []
    letter_areas = []

    for lx1, ly1, lx2, ly2 in letters:
        w = lx2 - lx1
        h = ly2 - ly1
        letter_widths.append(w)
        letter_heights.append(h)
        letter_areas.append(w * h)

    if len(letters) > 0:
        median_width = np.median(letter_widths)
        median_height = np.median(letter_heights)
        median_area = np.median(letter_areas)

        print(f"\nLetter statistics:")
        print(f"  Widths:  min={min(letter_widths):3d}, max={max(letter_widths):3d}, median={median_width:.1f}")
        print(f"  Heights: min={min(letter_heights):3d}, max={max(letter_heights):3d}, median={median_height:.1f}")
        print(f"  Areas:   min={min(letter_areas):4d}, max={max(letter_areas):4d}, median={median_area:.1f}")

        # Check for very narrow components (likely over-split)
        narrow_threshold = median_width * 0.4  # Less than 40% of median width
        narrow_components = [i for i, w in enumerate(letter_widths) if w < narrow_threshold]

        if narrow_components:
            print(f"\n⚠️  Found {len(narrow_components)} VERY NARROW components (< 40% median width):")
            for i in narrow_components:
                w = letter_widths[i]
                h = letter_heights[i]
                print(f"     Component {i}: width={w}, height={h} (width={w/median_width:.1%} of median)")
            print(f"   These are likely PARTS of letters that were incorrectly split")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Word image
    word_img = region[wy1:wy2, wx1:wx2].copy()
    axes[0, 0].imshow(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Word Image (should be "Epilogue" - 8 letters)')
    axes[0, 0].axis('off')

    # With letter boxes
    vis = region.copy()
    cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (255, 0, 0), 3)

    for i, (lx1, ly1, lx2, ly2) in enumerate(letters):
        w = lx2 - lx1
        # Color code: green=normal width, red=narrow (over-split)
        color = (0, 255, 0) if w >= narrow_threshold else (0, 0, 255)
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), color, 2)
        cv2.putText(vis, str(i), (lx1+2, ly1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    axes[0, 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Detected: {len(letters)} components (Green=OK, Red=too narrow)')
    axes[0, 1].axis('off')

    # Width distribution
    if len(letter_widths) > 0:
        axes[1, 0].bar(range(len(letter_widths)), letter_widths,
                      color=['green' if w >= narrow_threshold else 'red' for w in letter_widths])
        axes[1, 0].axhline(y=median_width, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_width:.1f}')
        axes[1, 0].axhline(y=narrow_threshold, color='red', linestyle='--', linewidth=1, label=f'40% threshold: {narrow_threshold:.1f}')
        axes[1, 0].set_xlabel('Component index')
        axes[1, 0].set_ylabel('Width (pixels)')
        axes[1, 0].set_title('Component Widths (Red bars = likely over-split)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # List all components
    text_lines = [f"{'#':<3} {'X':<4} {'Y':<4} {'W':<4} {'H':<4} {'Area':<6} {'Status'}"]
    text_lines.append("-" * 50)
    for i, (lx1, ly1, lx2, ly2) in enumerate(letters):
        w = lx2 - lx1
        h = ly2 - ly1
        area = w * h
        status = "OK" if w >= narrow_threshold else "TOO NARROW"
        text_lines.append(f"{i:<3} {lx1-wx1:<4} {ly1-wy1:<4} {w:<4} {h:<4} {area:<6} {status}")

    text_content = "\n".join(text_lines)
    axes[1, 1].text(0.05, 0.95, text_content, fontsize=7, family='monospace',
                   verticalalignment='top', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('All Components Details')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('notebooks/epilogue_oversplit_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved analysis to notebooks/epilogue_oversplit_analysis.png")
    plt.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Return summary
    return {
        'num_components': len(letters),
        'expected': 8,
        'narrow_components': narrow_components if len(letters) > 0 else [],
        'median_width': median_width if len(letters) > 0 else 0
    }

if __name__ == '__main__':
    result = analyze_epilogue_oversplit()

    if result and result['num_components'] > 10:
        print("\n" + "!" * 70)
        print("CONCLUSION: COMPONENT SPLITTING IS TOO AGGRESSIVE")
        print("!" * 70)
        print(f"The algorithm is splitting individual letters (p, i, g, u) into parts.")
        print(f"This happens because the vertical projection finds valleys WITHIN letters,")
        print(f"not just between them.")
        print(f"\nNeed to adjust splitting algorithm to be more conservative.")
