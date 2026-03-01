#!/usr/bin/env python3
"""
Comprehensive visualization of Börja and inför segmentation and reflow problem
Processes images/gang_p023_lines1.png
"""

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import detection_predictor
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def visualize_word_segmentation(img, word_box, word_name, line_num):
    """Visualize how a single word is segmented"""
    xmin, ymin, xmax, ymax = word_box

    # Extract word
    word_img = img[ymin:ymax, xmin:xmax]
    word_h, word_w = word_img.shape[:2]

    # Get components using find_rects
    words_list = [[xmin, ymin, xmax, ymax]]
    rectangles = find_rects(img, words_list, debug=False)

    # Get raw connected components
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    raw_components = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w >= 2 and h >= 2:
            raw_components.append((x, y, w, h))

    raw_components.sort(key=lambda c: c[0])

    # Create visualizations
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)
    ]

    # 1. Original word
    vis_original = word_img.copy()

    # 2. Raw components
    vis_raw = word_img.copy()
    for i, (x, y, w, h) in enumerate(raw_components):
        color = colors[i % len(colors)]
        cv2.rectangle(vis_raw, (x, y), (x+w, y+h), color, 1)
        cv2.putText(vis_raw, str(i+1), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # 3. After merging (find_rects output)
    vis_merged = word_img.copy()
    for i, rect in enumerate(rectangles):
        if hasattr(rect, 'x'):
            x, y, w, h = rect.x, rect.y, rect.w, rect.h
        else:
            x, y, w, h = rect
        color = colors[i % len(colors)]
        cv2.rectangle(vis_merged, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis_merged, str(i+1), (x+2, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Resize for visibility
    scale = 4
    vis_original = cv2.resize(vis_original, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    vis_raw = cv2.resize(vis_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    vis_merged = cv2.resize(vis_merged, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # Add labels
    label_h = 30
    label = np.ones((label_h, vis_original.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(label, 'ORIGINAL', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    vis_original = np.vstack([label, vis_original])

    label = np.ones((label_h, vis_raw.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(label, 'RAW COMPONENTS ({})'.format(len(raw_components)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    vis_raw = np.vstack([label, vis_raw])

    label = np.ones((label_h, vis_merged.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(label, 'AFTER MERGING ({})'.format(len(rectangles)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    vis_merged = np.vstack([label, vis_merged])

    # Stack horizontally
    max_h = max(vis_original.shape[0], vis_raw.shape[0], vis_merged.shape[0])

    # Pad to same height
    def pad_to_height(img, target_h):
        if img.shape[0] < target_h:
            pad = np.ones((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img = np.vstack([img, pad])
        return img

    vis_original = pad_to_height(vis_original, max_h)
    vis_raw = pad_to_height(vis_raw, max_h)
    vis_merged = pad_to_height(vis_merged, max_h)

    spacer = np.ones((max_h, 20, 3), dtype=np.uint8) * 200
    combined = np.hstack([vis_original, spacer, vis_raw, spacer, vis_merged])

    # Add title
    title_h = 40
    title = np.ones((title_h, combined.shape[1], 3), dtype=np.uint8) * 230
    text = 'Line {} - Word: "{}"'.format(line_num, word_name)
    cv2.putText(title, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    combined = np.vstack([title, combined])

    return combined, len(raw_components), len(rectangles)

def main():
    print("="*80)
    print("VISUALIZATION: Börja and inför segmentation problem")
    print("="*80)
    print()

    # Load image
    img = cv2.imread('images/gang_p023_lines1.png')

    if img is None:
        print("ERROR: Could not load images/gang_p023_lines1.png")
        return

    img_h, img_w = img.shape[:2]
    print("Image loaded: {}x{} pixels".format(img_w, img_h))
    print()

    # Detect words
    print("Step 1: Detecting words with doctr...")
    det_predictor = detection_predictor(arch='db_resnet50', pretrained=True)
    docs = DocumentFile.from_images(['images/gang_p023_lines1.png'])
    result = det_predictor(docs)
    words_array = result[0]['words']

    print("Found {} words".format(len(words_array)))
    print()

    # Convert to absolute coordinates
    words = []
    for word_box in words_array:
        xmin = int(word_box[0] * img_w)
        ymin = int(word_box[1] * img_h)
        xmax = int(word_box[2] * img_w)
        ymax = int(word_box[3] * img_h)
        words.append((xmin, ymin, xmax, ymax))

    # Sort by y then x (line by line, left to right)
    words.sort(key=lambda w: (w[1], w[0]))

    # Group into lines (words with similar y)
    lines = []
    current_line = []
    last_y = -100

    for word in words:
        xmin, ymin, xmax, ymax = word

        if abs(ymin - last_y) > 20:  # New line
            if current_line:
                lines.append(current_line)
            current_line = [word]
            last_y = ymin
        else:
            current_line.append(word)

    if current_line:
        lines.append(current_line)

    print("Organized into {} lines".format(len(lines)))
    print()

    # Identify target words
    print("Step 2: Looking for 'Börja' (line 4) and 'inför' (line 2)...")
    print()

    target_words = []

    # Line 2 should have "inför"
    if len(lines) >= 2:
        line2_words = lines[1]  # 0-indexed, so line 2 is index 1
        print("Line 2 has {} words".format(len(line2_words)))
        # inför is likely one of the first words
        for i, word in enumerate(line2_words[:3]):
            xmin, ymin, xmax, ymax = word
            width = xmax - xmin
            if 70 < width < 100:  # inför is ~5 chars
                print("  Candidate for 'inför': word {} at x={}".format(i+1, xmin))
                target_words.append((word, 'inför', 2))
                break

    # Line 4 should have "Börja"
    if len(lines) >= 4:
        line4_words = lines[3]  # 0-indexed, so line 4 is index 3
        print("Line 4 has {} words".format(len(line4_words)))
        # Börja is likely the first word
        if line4_words:
            word = line4_words[0]
            xmin, ymin, xmax, ymax = word
            print("  Candidate for 'Börja': word 1 at x={}".format(xmin))
            target_words.append((word, 'Börja', 4))

    print()

    if not target_words:
        print("ERROR: Could not find target words")
        print("Please check line numbering")
        return

    print("Step 3: Visualizing segmentation for each word...")
    print()

    visualizations = []

    for word_box, word_name, line_num in target_words:
        print("Processing '{}' (line {})...".format(word_name, line_num))
        vis, raw_count, merged_count = visualize_word_segmentation(img, word_box, word_name, line_num)
        visualizations.append(vis)

        print("  Raw components: {}".format(raw_count))
        print("  After merging: {}".format(merged_count))

        if merged_count == 1:
            print("  ✗ PROBLEM: Everything merged into 1 component!")
        elif raw_count < 7 and word_name == 'Börja':
            print("  ✗ PROBLEM: Not enough raw components (expected ~8 for Börja)")
        elif merged_count > 5 and word_name in ['Börja', 'inför']:
            print("  ✗ PROBLEM: Too many components, ö not merged correctly")
        else:
            print("  ✓ Component count looks OK")
        print()

    # Stack all visualizations vertically
    if visualizations:
        spacer = np.ones((20, visualizations[0].shape[1], 3), dtype=np.uint8) * 200

        result = visualizations[0]
        for vis in visualizations[1:]:
            # Make same width
            if vis.shape[1] != result.shape[1]:
                if vis.shape[1] < result.shape[1]:
                    pad = np.ones((vis.shape[0], result.shape[1] - vis.shape[1], 3), dtype=np.uint8) * 255
                    vis = np.hstack([vis, pad])
                else:
                    pad = np.ones((result.shape[0], vis.shape[1] - result.shape[1], 3), dtype=np.uint8) * 255
                    result = np.hstack([result, pad])

            result = np.vstack([result, spacer, vis])

        cv2.imwrite('/tmp/borja_infor_segmentation_analysis.png', result)
        print("Saved visualization to: /tmp/borja_infor_segmentation_analysis.png")
        print()

    # Now run full reflow
    print("Step 4: Running full reflow...")
    print()

    import subprocess
    subprocess.run([
        'pixi', 'run', 'python', 'src/ocr_reflow/main.py',
        'images/gang_p023_lines1.png', '--layout'
    ], capture_output=False)

    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print("Files created:")
    print("  1. /tmp/borja_infor_segmentation_analysis.png")
    print("     Shows side-by-side: original, raw components, after merging")
    print()
    print("  2. output_reflowed.png")
    print("     Shows the reflowed result")
    print()
    print("Compare these to understand the problem:")
    print("  - If 'after merging' shows 1 component: everything is being merged")
    print("  - If 'after merging' shows too many: ö dots not merging with o")
    print("  - Check output_reflowed.png to see actual rendering issue")
    print()

if __name__ == '__main__':
    main()
