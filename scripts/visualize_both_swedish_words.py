#!/usr/bin/env python3
"""
Comprehensive visualization of both Börja and inför
Shows original word, raw components, merged components, and reflowed output
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def create_word_visualization(img, word_box, word_name):
    """Create detailed visualization for one word"""
    xmin, ymin, xmax, ymax = word_box
    word_img = img[ymin:ymax, xmin:xmax]
    word_h, word_w = word_img.shape[:2]

    print(f"\n{'='*80}")
    print(f"WORD: {word_name}")
    print(f"{'='*80}")
    print(f"Box: ({xmin},{ymin}) → ({xmax},{ymax}), Size: {word_w}x{word_h}px")
    print()

    # Get raw components
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

    heights = [h for x, y, w, h in raw_components]
    median_h = np.median(heights)

    print(f"RAW COMPONENTS: {len(raw_components)}")
    print(f"Median height: {median_h:.1f}px")
    print("-" * 60)

    for i, (x, y, w, h) in enumerate(raw_components):
        y_pct = (y / word_h) * 100
        is_dot = (w < median_h * 0.8 and h < median_h * 0.4 and y < word_h * 0.4)
        label = "DOT" if is_dot else "LETTER"
        print(f"  {i+1:2d}. x={x:3d} y={y:3d} ({y_pct:4.1f}%) {w:2d}x{h:2d} [{label}]")

    # Get merged components
    words_list = [[xmin, ymin, xmax, ymax]]
    rectangles = find_rects(img, words_list, debug=False)

    print()
    print(f"MERGED COMPONENTS: {len(rectangles)}")
    print("-" * 60)

    for i, rect in enumerate(rectangles):
        rxmin, rymin, rxmax, rymax = rect
        w = rxmax - rxmin
        h = rymax - rymin
        rel_x = rxmin - xmin
        rel_y = rymin - ymin
        print(f"  {i+1:2d}. word-rel: x={rel_x:3d} y={rel_y:3d} size {w:2d}x{h:2d}")

    # Create visualizations
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (255, 0, 128)
    ]

    scale = 10

    # 1. Original
    vis_original = cv2.resize(word_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 2. Raw components
    vis_raw = word_img.copy()
    for i, (x, y, w, h) in enumerate(raw_components):
        color = colors[i % len(colors)]
        cv2.rectangle(vis_raw, (x, y), (x+w, y+h), color, 1)
        cv2.putText(vis_raw, str(i+1), (x+1, y+h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
    vis_raw = cv2.resize(vis_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 3. Merged components
    vis_merged = word_img.copy()
    for i, rect in enumerate(rectangles):
        rxmin, rymin, rxmax, rymax = rect
        rel_x1 = rxmin - xmin
        rel_y1 = rymin - ymin
        rel_x2 = rxmax - xmin
        rel_y2 = rymax - ymin

        color = colors[i % len(colors)]
        cv2.rectangle(vis_merged, (rel_x1, rel_y1), (rel_x2, rel_y2), color, 2)
        cv2.putText(vis_merged, str(i+1), (rel_x1+2, rel_y1+12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    vis_merged = cv2.resize(vis_merged, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # 4. Extract individual component images for reflow simulation
    vis_components_grid = np.ones((vis_merged.shape[0], vis_merged.shape[1], 3), dtype=np.uint8) * 255

    # Sort rectangles left to right
    sorted_rects = sorted(enumerate(rectangles), key=lambda x: x[1][0])

    current_x = 10
    baseline_y = vis_merged.shape[0] // 2

    for idx, (i, rect) in enumerate(sorted_rects):
        rxmin, rymin, rxmax, rymax = rect
        rel_x1 = rxmin - xmin
        rel_y1 = rymin - ymin
        rel_x2 = rxmax - xmin
        rel_y2 = rymax - ymin

        # Extract component from original
        if 0 <= rel_x1 < word_w and 0 <= rel_y1 < word_h and rel_x2 <= word_w and rel_y2 <= word_h:
            component = word_img[rel_y1:rel_y2, rel_x1:rel_x2]
            comp_h, comp_w = component.shape[:2]

            if comp_h > 0 and comp_w > 0:
                # Scale it
                scaled_comp = cv2.resize(component, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                sc_h, sc_w = scaled_comp.shape[:2]

                # Place it on the grid
                y_pos = baseline_y - sc_h + 10
                if y_pos < 0:
                    y_pos = 0
                if y_pos + sc_h > vis_components_grid.shape[0]:
                    y_pos = vis_components_grid.shape[0] - sc_h

                if current_x + sc_w < vis_components_grid.shape[1]:
                    vis_components_grid[y_pos:y_pos+sc_h, current_x:current_x+sc_w] = scaled_comp

                    # Draw number
                    cv2.putText(vis_components_grid, str(i+1),
                               (current_x, y_pos-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    current_x += sc_w + 5

    # Add labels
    label_h = 50

    def add_label(img, text):
        label = np.ones((label_h, img.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(label, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return np.vstack([label, img])

    vis_original = add_label(vis_original, f"ORIGINAL: {word_name}")
    vis_raw = add_label(vis_raw, f"RAW ({len(raw_components)} comp)")
    vis_merged = add_label(vis_merged, f"MERGED ({len(rectangles)} comp)")
    vis_components_grid = add_label(vis_components_grid, f"REFLOW SIMULATION")

    # Stack vertically
    max_w = max(vis_original.shape[1], vis_raw.shape[1], vis_merged.shape[1], vis_components_grid.shape[1])

    def pad_width(img, target_w):
        if img.shape[1] < target_w:
            pad = np.ones((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8) * 255
            img = np.hstack([img, pad])
        return img

    vis_original = pad_width(vis_original, max_w)
    vis_raw = pad_width(vis_raw, max_w)
    vis_merged = pad_width(vis_merged, max_w)
    vis_components_grid = pad_width(vis_components_grid, max_w)

    spacer = np.ones((20, max_w, 3), dtype=np.uint8) * 200

    combined = np.vstack([vis_original, spacer, vis_raw, spacer, vis_merged, spacer, vis_components_grid])

    return combined, len(raw_components), len(rectangles)

def main():
    print("="*80)
    print("COMPREHENSIVE VISUALIZATION: Börja and inför")
    print("="*80)

    img = cv2.imread('images/gang_p023_lines1.png')

    if img is None:
        print("✗ Could not load image")
        return

    # Word locations
    words = [
        ('Borja', (81, 174, 224, 240)),
        ('infor', (192, 63, 298, 111))
    ]

    visualizations = []

    for word_name, box in words:
        vis, raw_count, merged_count = create_word_visualization(img, box, word_name)
        visualizations.append(vis)

    # Stack both words side by side
    max_h = max(v.shape[0] for v in visualizations)

    padded_vis = []
    for vis in visualizations:
        if vis.shape[0] < max_h:
            pad = np.ones((max_h - vis.shape[0], vis.shape[1], 3), dtype=np.uint8) * 255
            vis = np.vstack([vis, pad])
        padded_vis.append(vis)

    spacer = np.ones((max_h, 40, 3), dtype=np.uint8) * 200
    final = np.hstack([padded_vis[0], spacer, padded_vis[1]])

    # Add main title
    title_h = 80
    title = np.ones((title_h, final.shape[1], 3), dtype=np.uint8) * 230
    cv2.putText(title, "Swedish Words: Borja and infor",
               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
    final = np.vstack([title, final])

    cv2.imwrite('/tmp/swedish_words_complete_visualization.png', final)

    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print("File saved: /tmp/swedish_words_complete_visualization.png")
    print()
    print("This shows for BOTH words:")
    print("  1. ORIGINAL - The word as it appears")
    print("  2. RAW - All connected components (numbered, colored boxes)")
    print("  3. MERGED - Components after diacritic merging")
    print("  4. REFLOW SIMULATION - How components appear when placed on new page")
    print()
    print("Please examine this file and tell me:")
    print("  - In REFLOW SIMULATION, do you see the 'parts of letters' issue?")
    print("  - Which components look wrong?")
    print("  - Are dots merged correctly with their base letters?")
    print()

if __name__ == '__main__':
    main()
