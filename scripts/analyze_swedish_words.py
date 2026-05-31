#!/usr/bin/env python3
"""
Analyze specific Swedish words to find why ö becomes öö, ä becomes äi, etc.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')

from doctr.models import detection_predictor
from main import find_rects

def extract_and_analyze_word(img, word_box, word_label):
    """Extract a word and analyze its character segmentation"""

    xmin, ymin, xmax, ymax = word_box
    print(f"\n{'='*80}")
    print(f"ANALYZING WORD: {word_label}")
    print(f"{'='*80}")
    print(f"Box: ({xmin}, {ymin}) → ({xmax}, {ymax}), size: {xmax-xmin}x{ymax-ymin}")

    # Extract word image
    word_img = img[ymin:ymax, xmin:xmax].copy()

    # Save original word
    cv2.imwrite(f"/tmp/swedish_{word_label}_original.png", word_img)
    print(f"Saved original to /tmp/swedish_{word_label}_original.png")

    # Run find_rects to see how characters are extracted
    box_img = img.copy()
    words_list = [[xmin, ymin, xmax, ymax]]

    try:
        rectangles = find_rects(box_img, words_list)
        print(f"\nExtracted {len(rectangles)} characters:")

        # Visualize each character
        vis_img = word_img.copy()
        for i, rect in enumerate(rectangles):
            x, y, w, h = rect.x - xmin, rect.y - ymin, rect.w, rect.h
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print(f"  Char {i}: ({rect.x}, {rect.y}) size={rect.w}x{rect.h} baseline={rect.baseline:.1f}")

        cv2.imwrite(f"/tmp/swedish_{word_label}_segmented.png", vis_img)
        print(f"Saved segmentation to /tmp/swedish_{word_label}_segmented.png")

    except Exception as e:
        print(f"ERROR in find_rects: {e}")
        import traceback
        traceback.print_exc()

    # Also do manual connected component analysis
    print(f"\nMANUAL CONNECTED COMPONENT ANALYSIS:")
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    word_height = ymax - ymin
    word_width = xmax - xmin

    components = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if w >= 3 and h >= 3 and area >= 9:
            components.append({
                'idx': i,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'rel_h': h / word_height,
                'rel_w': w / word_width
            })

    components.sort(key=lambda c: c['x'])

    if components:
        heights = [c['h'] for c in components]
        median_h = np.median(heights)
        print(f"Found {len(components)} components, median height: {median_h:.1f}px")

        print(f"\nComponents (left to right):")
        for c in components:
            # Standard diacritic test
            is_standard_diacritic = (c['h'] < median_h * 0.4 and
                                     c['w'] < median_h * 0.8 and
                                     c['w'] * c['h'] < (median_h ** 2) * 0.3 and
                                     c['h'] < word_height * 0.25 and
                                     c['w'] < word_width * 0.5)

            # Swedish/Scandinavian diacritic test: tall narrow components
            is_tall_narrow_diacritic = (c['w'] < c['h'] * 0.5 and  # Narrow relative to height
                                       c['w'] * c['h'] < (word_width * word_height) * 0.1 and  # Small area relative to word
                                       c['y'] < word_height * 0.6 and  # In top 60%
                                       c['w'] < word_width * 0.25 and  # Narrow relative to word
                                       c['h'] < word_height * 0.9)  # Not full height

            is_diacritic = is_standard_diacritic or is_tall_narrow_diacritic

            type_str = "DIACRITIC"
            if is_diacritic:
                if is_tall_narrow_diacritic and not is_standard_diacritic:
                    type_str = "DIACRITIC(TALL)"
            else:
                type_str = "MAIN"

            print(f"  #{c['idx']}: ({c['x']:3d}, {c['y']:3d}) size={c['w']:2d}x{c['h']:2d} "
                  f"rel_h={c['rel_h']:.2f} area={c['area']:4d} w/h={c['w']/c['h'] if c['h']>0 else 0:.2f} [{type_str}]")

        # Visualize components with colors
        vis_img2 = word_img.copy()
        for c in components:
            is_standard_diacritic = (c['h'] < median_h * 0.4 and
                                     c['w'] < median_h * 0.8 and
                                     c['w'] * c['h'] < (median_h ** 2) * 0.3 and
                                     c['h'] < word_height * 0.25 and
                                     c['w'] < word_width * 0.5)

            is_tall_narrow_diacritic = (c['w'] < c['h'] * 0.5 and
                                       c['w'] * c['h'] < (word_width * word_height) * 0.1 and
                                       c['y'] < word_height * 0.6 and
                                       c['w'] < word_width * 0.25 and
                                       c['h'] < word_height * 0.9)

            is_diacritic = is_standard_diacritic or is_tall_narrow_diacritic

            if is_diacritic:
                # Yellow for standard diacritics, Cyan for tall narrow (Swedish)
                color = (0, 255, 255) if is_standard_diacritic else (255, 255, 0)
            else:
                color = (255, 0, 255)  # Magenta for main

            cv2.rectangle(vis_img2, (c['x'], c['y']),
                         (c['x']+c['w'], c['y']+c['h']), color, 2)

        cv2.imwrite(f"/tmp/swedish_{word_label}_components.png", vis_img2)
        print(f"Saved components to /tmp/swedish_{word_label}_components.png")
        print(f"  (Yellow=standard diacritic, Cyan=tall narrow diacritic, Magenta=main)")


def main():
    image_path = "images/gang_p023.png"

    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load {image_path}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Run doctr to get word boxes
    print("\nRunning doctr word detection...")
    det_predictor = detection_predictor(arch='db_resnet50', pretrained=True)

    # Need to use DocumentFile format like in main.py
    from doctr.io import DocumentFile
    docs = DocumentFile.from_images([image_path])
    result = det_predictor(docs)

    # Result is a list where result[0] is a dict with "words" key
    words_array = result[0]["words"]

    img_h, img_w, _ = img.shape

    # Convert normalized coordinates to pixel coordinates
    words = []
    for word_box in words_array:
        xmin = int(word_box[0] * img_w)
        ymin = int(word_box[1] * img_h)
        xmax = int(word_box[2] * img_w)
        ymax = int(word_box[3] * img_h)
        words.append([xmin, ymin, xmax, ymax])

    print(f"Found {len(words)} words")

    # Manually identify words to analyze based on visual inspection
    # You mentioned: "Börja" → "Böörja", "inför" → "inföör"
    # Let's analyze several words from different parts of the page

    # Let's examine words that likely contain Swedish diacritics
    # Based on typical Swedish text patterns and the diagnostic output

    # Word 7 from diagnostic had 2 small components (likely ö with 2 dots)
    print(f"\n\n{'#'*80}")
    print(f"# EXAMINING WORD 7 (likely contains ö with 2 dots)")
    print(f"{'#'*80}")
    if len(words) > 7:
        extract_and_analyze_word(img, words[7], "word_07")

    # Word 4 from diagnostic had 1 small component
    print(f"\n\n{'#'*80}")
    print(f"# EXAMINING WORD 4 (likely contains ö or ä)")
    print(f"{'#'*80}")
    if len(words) > 4:
        extract_and_analyze_word(img, words[4], "word_04")

    # Word 6 from diagnostic had 1 small component
    print(f"\n\n{'#'*80}")
    print(f"# EXAMINING WORD 6 (likely contains ö or ä)")
    print(f"{'#'*80}")
    if len(words) > 6:
        extract_and_analyze_word(img, words[6], "word_06")

    # Word 9 from diagnostic had 1 small component
    print(f"\n\n{'#'*80}")
    print(f"# EXAMINING WORD 9 (likely contains ö or ä)")
    print(f"{'#'*80}")
    if len(words) > 9:
        extract_and_analyze_word(img, words[9], "word_09")

    # Word 11 from diagnostic had 3 small components
    print(f"\n\n{'#'*80}")
    print(f"# EXAMINING WORD 11 (complex case with 3 small components)")
    print(f"{'#'*80}")
    if len(words) > 11:
        extract_and_analyze_word(img, words[11], "word_11")

    print(f"\n\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Check /tmp/swedish_*.png files for visualizations")


if __name__ == "__main__":
    main()
