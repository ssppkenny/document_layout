#!/usr/bin/env python3
"""
Visualize Börja and inför with correct word identification
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def visualize_word(img, word_box, word_name, output_name):
    """Create detailed visualization for one word"""
    xmin, ymin, xmax, ymax = word_box

    # Extract word
    word_img = img[ymin:ymax, xmin:xmax]
    word_h, word_w = word_img.shape[:2]

    print(f"Processing '{word_name}':")
    print(f"  Box: ({xmin},{ymin}) → ({xmax},{ymax})")
    print(f"  Size: {word_w}x{word_h}")

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

    print(f"  Raw components: {len(raw_components)}")
    for i, (x, y, w, h) in enumerate(raw_components):
        y_pct = (y / word_h) * 100
        print(f"    {i+1}. ({x:2d},{y:2d}) {w:2d}x{h:2d} y={y_pct:4.1f}%")

    print(f"  After merging: {len(rectangles)}")

    # Create visualizations
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (255, 0, 128)
    ]

    scale = 8

    # 1. Original
    vis_orig = cv2.resize(word_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 2. Raw components
    vis_raw = word_img.copy()
    for i, (x, y, w, h) in enumerate(raw_components):
        color = colors[i % len(colors)]
        cv2.rectangle(vis_raw, (x, y), (x+w, y+h), color, 1)
        cv2.putText(vis_raw, str(i+1), (x, y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    vis_raw = cv2.resize(vis_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 3. After merging
    vis_merged = word_img.copy()
    for i, rect in enumerate(rectangles):
        if hasattr(rect, 'x'):
            x, y, w, h = rect.x, rect.y, rect.w, rect.h
        else:
            x, y, w, h = rect
        color = colors[i % len(colors)]
        cv2.rectangle(vis_merged, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis_merged, str(i+1), (x+2, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    vis_merged = cv2.resize(vis_merged, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # Add labels and stack
    h_target = vis_orig.shape[0]
    w_max = max(vis_orig.shape[1], vis_raw.shape[1], vis_merged.shape[1])

    def add_label_and_pad(img, label_text, target_h, target_w):
        label_h = 40
        label = np.ones((label_h, target_w, 3), dtype=np.uint8) * 255
        cv2.putText(label, label_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Pad width if needed
        if img.shape[1] < target_w:
            pad = np.ones((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8) * 255
            img = np.hstack([img, pad])

        img_with_label = np.vstack([label, img])

        # Pad height if needed
        if img_with_label.shape[0] < target_h + label_h:
            pad = np.ones((target_h + label_h - img_with_label.shape[0], target_w, 3), dtype=np.uint8) * 255
            img_with_label = np.vstack([img_with_label, pad])

        return img_with_label

    vis_orig = add_label_and_pad(vis_orig, "ORIGINAL", h_target, w_max)
    vis_raw = add_label_and_pad(vis_raw, f"RAW ({len(raw_components)} components)", h_target, w_max)
    vis_merged = add_label_and_pad(vis_merged, f"MERGED ({len(rectangles)} components)", h_target, w_max)

    # Stack horizontally
    spacer = np.ones((vis_orig.shape[0], 30, 3), dtype=np.uint8) * 200
    combined = np.hstack([vis_orig, spacer, vis_raw, spacer, vis_merged])

    # Add title
    title_h = 60
    title = np.ones((title_h, combined.shape[1], 3), dtype=np.uint8) * 230
    cv2.putText(title, f"Word: {word_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    combined = np.vstack([title, combined])

    cv2.imwrite(output_name, combined)
    print(f"  Saved to: {output_name}")
    print()

    return len(raw_components), len(rectangles)

def main():
    print("="*80)
    print("VISUALIZATION: Börja and inför segmentation")
    print("="*80)
    print()

    img = cv2.imread('images/gang_p023_lines1.png')

    if img is None:
        print("ERROR: Could not load images/gang_p023_lines1.png")
        return

    print("Based on user input:")
    print("  - 'inför' is 3rd word on line 2")
    print("  - 'Börja' is 1st word on line 4")
    print()

    # From the detection output:
    # Line 4, word 1: x=37-102 (65px) y=121-169
    borja_box = (37, 121, 102, 169)

    # Line 2 only shows 2 words, but user says inför is 3rd word
    # Let me check all words near line 2 y-coordinates
    from doctr.io import DocumentFile
    from doctr.models import detection_predictor

    img_h, img_w = img.shape[:2]
    det_predictor = detection_predictor(arch='db_resnet50', pretrained=True)
    docs = DocumentFile.from_images(['images/gang_p023_lines1.png'])
    result = det_predictor(docs)
    words_array = result[0]['words']

    # Get all words around y=37-99 (line 2 area)
    line2_candidates = []
    for word_box in words_array:
        xmin = int(word_box[0] * img_w)
        ymin = int(word_box[1] * img_h)
        xmax = int(word_box[2] * img_w)
        ymax = int(word_box[3] * img_h)

        if 30 < ymin < 110:  # Line 2 area
            line2_candidates.append((xmin, ymin, xmax, ymax))

    line2_candidates.sort(key=lambda w: (w[1], w[0]))

    print("All word boxes in line 2 area (y=30-110):")
    for i, (xmin, ymin, xmax, ymax) in enumerate(line2_candidates):
        width = xmax - xmin
        print(f"  {i+1}. x={xmin}-{xmax} ({width}px) y={ymin}-{ymax}")
    print()

    # User says inför is 3rd word, so take index 2
    if len(line2_candidates) >= 3:
        infor_box = line2_candidates[2]
    else:
        print("WARNING: Less than 3 words found in line 2 area")
        print("Using last word as fallback")
        infor_box = line2_candidates[-1] if line2_candidates else (1204, 56, 1269, 99)

    print(f"Selected 'inför' box: {infor_box}")
    print(f"Selected 'Börja' box: {borja_box}")
    print()
    print("-"*80)
    print()

    # Visualize each word
    raw1, merged1 = visualize_word(img, borja_box, "Börja", "/tmp/borja_detailed.png")
    raw2, merged2 = visualize_word(img, infor_box, "inför", "/tmp/infor_detailed.png")

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Börja:  {raw1} raw → {merged1} merged")
    print(f"inför:  {raw2} raw → {merged2} merged")
    print()

    if merged1 == 1:
        print("✗ Börja: Everything merged into 1 component!")
    elif raw1 < 7:
        print("✗ Börja: Too few raw components (expected ~8)")
    elif merged1 > 5:
        print("✗ Börja: ö dots not merged correctly")
    else:
        print("✓ Börja: Component count OK")

    if merged2 == 1:
        print("✗ inför: Everything merged into 1 component!")
    elif raw2 < 7:
        print("✗ inför: Too few raw components (expected ~6-7)")
    elif merged2 > 5:
        print("✗ inför: ö dots not merged correctly")
    else:
        print("✓ inför: Component count OK")

    print()
    print("Files created:")
    print("  /tmp/borja_detailed.png")
    print("  /tmp/infor_detailed.png")
    print()

if __name__ == '__main__':
    main()
