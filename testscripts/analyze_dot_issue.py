#!/usr/bin/env python3
"""
Analyze the dot placement issue - dots detected as separate symbols
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

def analyze_dot_issue(image_path):
    """Analyze how dots are currently being handled"""

    print(f"Analyzing dot placement issue in: {image_path}")
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

    # Extract letters
    letters = find_rects(region, words)
    print(f"✓ Extracted {len(letters)} letters")

    # Analyze letter sizes to find dots vs main letters
    letter_heights = [(ly2 - ly1) for lx1, ly1, lx2, ly2 in letters]
    median_height = np.median(letter_heights)

    # Classify letters
    main_letters = []
    dots = []

    for letter in letters:
        lx1, ly1, lx2, ly2 = letter
        h = ly2 - ly1
        w = lx2 - lx1

        # Dots are typically < 40% of median letter height and small in area
        if h < median_height * 0.4 and w < median_height * 0.5 and (w * h) < (median_height ** 2) * 0.3:
            dots.append(letter)
        else:
            main_letters.append(letter)

    print(f"\nClassification:")
    print(f"  Main letters: {len(main_letters)}")
    print(f"  Dots/accents: {len(dots)}")
    print(f"  Median letter height: {median_height:.1f}")

    # Find dots that are above main letters (potential i, j dots)
    dot_letter_pairs = []

    for dot in dots:
        dx1, dy1, dx2, dy2 = dot
        dot_cx = (dx1 + dx2) / 2
        dot_cy = (dy1 + dy2) / 2

        # Find main letters below this dot
        candidates = []
        for main in main_letters:
            mx1, my1, mx2, my2 = main
            main_cy = (my1 + my2) / 2

            # Dot should be above (smaller y) and horizontally aligned
            if dy2 <= my1 + 5:  # Dot bottom above or just touching main top
                # Check horizontal alignment
                main_cx = (mx1 + mx2) / 2
                horizontal_distance = abs(dot_cx - main_cx)

                if horizontal_distance < median_height * 0.8:  # Within reasonable distance
                    vertical_distance = my1 - dy2
                    candidates.append((main, vertical_distance, horizontal_distance))

        if candidates:
            # Pick closest candidate
            best_main, v_dist, h_dist = min(candidates, key=lambda x: (x[1], x[2]))
            dot_letter_pairs.append((dot, best_main, h_dist))

    print(f"\nFound {len(dot_letter_pairs)} dot-letter pairs")

    # Visualize the issue
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original region
    axes[0, 0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Text Region')
    axes[0, 0].axis('off')

    # All letters
    vis_all = region.copy()
    for lx1, ly1, lx2, ly2 in letters:
        cv2.rectangle(vis_all, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)

    axes[0, 1].imshow(cv2.cvtColor(vis_all, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'All Letters ({len(letters)})')
    axes[0, 1].axis('off')

    # Classified: main vs dots
    vis_classified = region.copy()
    for lx1, ly1, lx2, ly2 in main_letters:
        cv2.rectangle(vis_classified, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)
    for dx1, dy1, dx2, dy2 in dots:
        cv2.rectangle(vis_classified, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)

    axes[0, 2].imshow(cv2.cvtColor(vis_classified, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Classified: Green=Main({len(main_letters)}), Red=Dots({len(dots)})')
    axes[0, 2].axis('off')

    # Dot-letter pairs
    vis_pairs = region.copy()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(dot_letter_pairs)))

    for idx, (dot, main, h_dist) in enumerate(dot_letter_pairs):
        color = (int(colors[idx][2] * 255), int(colors[idx][1] * 255), int(colors[idx][0] * 255))

        dx1, dy1, dx2, dy2 = dot
        mx1, my1, mx2, my2 = main

        # Draw rectangles
        cv2.rectangle(vis_pairs, (dx1, dy1), (dx2, dy2), color, 2)
        cv2.rectangle(vis_pairs, (mx1, my1), (mx2, my2), color, 2)

        # Draw line connecting them
        dot_center = ((dx1 + dx2) // 2, (dy1 + dy2) // 2)
        main_center = ((mx1 + mx2) // 2, (my1 + my2) // 2)
        cv2.line(vis_pairs, dot_center, main_center, color, 1)

        # Show horizontal misalignment
        cv2.putText(vis_pairs, f"{int(h_dist)}",
                   ((dx1 + mx1) // 2, (dy1 + my2) // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    axes[1, 0].imshow(cv2.cvtColor(vis_pairs, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Dot-Letter Pairs ({len(dot_letter_pairs)}) - Numbers show misalignment')
    axes[1, 0].axis('off')

    # Analysis text
    analysis = f"""PROBLEM ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dots detected as SEPARATE letters: {len(dots)}
Identified dot-letter pairs: {len(dot_letter_pairs)}

ISSUE:
During reflow, dots are placed independently
→ Can shift horizontally from their base letters
→ Results in misaligned dots above i, j

SOLUTION OPTIONS:

1) MERGE APPROACH (BETTER):
   - Merge dot with base letter into single bbox
   - Treat i, j as atomic units
   - During reflow: place as single unit
   ✓ Simpler reflow logic
   ✓ Guaranteed alignment
   ✓ More robust
   
2) PRECISE PLACEMENT APPROACH:
   - Keep dots separate
   - During reflow: calculate exact dot position
   - Must track base letter position
   ✗ Complex reflow logic
   ✗ Error-prone
   ✗ Still possible misalignment

RECOMMENDATION: Use MERGE approach
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    axes[1, 1].text(0.05, 0.95, analysis, fontsize=9, family='monospace',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    axes[1, 1].set_title('Problem Analysis')
    axes[1, 1].axis('off')

    # Show some example pairs in detail
    if len(dot_letter_pairs) > 0:
        # Pick first 3 pairs
        n_examples = min(3, len(dot_letter_pairs))
        example_width = region.shape[1] // n_examples

        combined = np.ones((150, example_width * n_examples, 3), dtype=np.uint8) * 255

        for i in range(n_examples):
            dot, main, h_dist = dot_letter_pairs[i]
            dx1, dy1, dx2, dy2 = dot
            mx1, my1, mx2, my2 = main

            # Create combined view
            merged_x1 = min(dx1, mx1)
            merged_y1 = dy1
            merged_x2 = max(dx2, mx2)
            merged_y2 = my2

            if merged_x2 > merged_x1 and merged_y2 > merged_y1:
                letter_img = region[merged_y1:merged_y2, merged_x1:merged_x2].copy()

                # Resize to fit
                target_h = 140
                target_w = int(letter_img.shape[1] * target_h / letter_img.shape[0])
                if target_w < example_width - 10:
                    letter_img = cv2.resize(letter_img, (target_w, target_h))

                    # Place in combined
                    x_offset = i * example_width + (example_width - target_w) // 2
                    combined[5:5+target_h, x_offset:x_offset+target_w] = letter_img

        axes[1, 2].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Example Dot-Letter Pairs (zoomed)')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'No pairs found', ha='center', va='center')
        axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = '../notebooks/dot_alignment_issue.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved analysis to {output_path}")
    plt.close()

    # Print statistics
    if dot_letter_pairs:
        h_distances = [h_dist for _, _, h_dist in dot_letter_pairs]
        print(f"\nHorizontal misalignment statistics:")
        print(f"  Min: {min(h_distances):.1f} pixels")
        print(f"  Max: {max(h_distances):.1f} pixels")
        print(f"  Mean: {np.mean(h_distances):.1f} pixels")
        print(f"  Median: {np.median(h_distances):.1f} pixels")

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'images/sedg_p598.png'
    analyze_dot_issue(image_path)
