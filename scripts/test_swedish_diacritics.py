#!/usr/bin/env python3
"""
Test Swedish diacritics merging - verify ä, ö, å are properly merged
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/ocr_reflow'))

from main import find_rects

def test_swedish_diacritics(image_path):
    """Test if Swedish diacritics are properly merged with base letters"""

    print(f"\n{'='*80}")
    print(f"TESTING SWEDISH DIACRITICS MERGING: {image_path}")
    print(f"{'='*80}\n")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image {image_path}")
        return

    # Get doctr words (using same method as main.py)
    from doctr.io import DocumentFile
    from doctr.models import detection_predictor

    model = detection_predictor(pretrained=True)
    docs = DocumentFile.from_images([image_path])
    result = model(docs)

    h, w = img.shape[:2]
    words_raw = result[0]["words"]

    # Convert to pixel coordinates with padding (same as main.py)
    words = []
    for word in words_raw:
        xmin = int(word[0] * w) - 5
        ymin = int(word[1] * h) - 5
        xmax = int(word[2] * w) + 5
        ymax = int(word[3] * h) + 5

        # Clamp to image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        words.append([xmin, ymin, xmax, ymax])

    print(f"✓ Detected {len(words)} words\n")

    # Test first 15 words
    print("Testing character segmentation on first 15 words:")
    print(f"{'='*80}\n")

    # Create visualization
    img_vis = img.copy()

    for idx, (xmin, ymin, xmax, ymax) in enumerate(words[:15]):
        word_height = ymax - ymin
        word_width = xmax - xmin

        # Extract word region
        box_img = img[ymin:ymax, xmin:xmax].copy()

        # Get rectangles using find_rects (tests the actual merging logic)
        try:
            rectangles = find_rects(box_img, [[xmin, ymin, xmax, ymax]])

            print(f"Word {idx+1}: ({xmin}, {ymin}) → ({xmax}, {ymax}), size: {word_width}x{word_height}")
            print(f"  → Segmented into {len(rectangles)} character(s)")

            # Check for potential diacritic issues
            # If a word has many small vertical slices, it might be split incorrectly
            if len(rectangles) > 0:
                rect_widths = [r[1] - r[0] for r in rectangles]
                rect_heights = [r[3] - r[2] for r in rectangles]
                avg_width = np.mean(rect_widths)
                avg_height = np.mean(rect_heights)

                # Check if there are suspiciously narrow components (might be split diacritics)
                narrow_rects = [w for w in rect_widths if w < avg_width * 0.3]
                if narrow_rects:
                    print(f"  ⚠ WARNING: {len(narrow_rects)} suspiciously narrow component(s) detected")
                    print(f"     Average width: {avg_width:.1f}px, narrow widths: {narrow_rects}")

                # Visualize rectangles
                for rect_idx, (rxmin, rxmax, rymin, rymax) in enumerate(rectangles):
                    cv2.rectangle(img_vis, (rxmin, rymin), (rxmax, rymax), (0, 255, 0), 1)

            # Draw word box
            cv2.rectangle(img_vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(img_vis, str(idx+1), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        except Exception as e:
            print(f"Word {idx+1}: ERROR - {e}")

        print()

    # Save visualization
    output_path = 'output_swedish_char_segmentation.png'
    cv2.imwrite(output_path, img_vis)
    print(f"✓ Saved character segmentation visualization to: {output_path}\n")

    print("="*80)
    print("TEST SUMMARY:")
    print("="*80)
    print("Check the output image to verify:")
    print("  1. Swedish letters ä, ö, å appear as single unified rectangles")
    print("  2. No split diacritics (dots separated from base letters)")
    print("  3. Characters are properly bounded without clipping")

if __name__ == '__main__':
    image_path = 'images/gang_p023.png'
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    test_swedish_diacritics(image_path)
