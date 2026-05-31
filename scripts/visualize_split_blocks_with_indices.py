#!/usr/bin/env python3
"""
Generate an annotated visualization of split layout blocks for dirak_p306.png
Each block overlays:
  - Index number
  - Block type (plain text, formula, ...)
  - [xmin, ymin, xmax, ymax] coordinates
  - Color-coded rectangle per type
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src/ocr_reflow'))

from layout import layout, find_grouped_bounding_boxes

def get_color(block_type):
    # Distinct color per type (plain text: green, formula: orange, other: blue, etc.)
    colors = {
        'plain text': (60, 200, 60),
        'isolate_formula': (40, 120, 255),
        'figure': (180, 0, 180),
        'table': (140, 180, 60),
        'figure_caption': (200, 180, 80),
        'table_caption': (80, 200, 200),
        'formula_caption': (255, 120, 40),
        'isolate_formula_and_caption': (0, 180, 180),
        'figure_and_caption': (180, 180, 0),
        # Add more as needed
    }
    return colors.get(block_type, (80, 80, 240))  # Default: blue

def main():
    image_path = 'images/dirak_p306.png'
    out_path = 'split_blocks_annotated.png'

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read input image: {image_path}")
        sys.exit(1)
    vis = img.copy()

    # Get split regions via the main layout splitter (with image_path for merging fix)
    results = layout(image_path)
    print("=== SPLIT BLOCKS RETURNED BY layout() ===")
    for idx, (geom, block_type) in enumerate(results):
        x1, y1, x2, y2 = map(int, geom.bounds)
        print(f"{idx}: {block_type} [{x1},{y1},{x2},{y2}]")
    print("")

    for idx, (geom, block_type) in enumerate(results):
        x1, y1, x2, y2 = map(int, geom.bounds)
        color = get_color(block_type)
        # Draw rectangle
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # Compose label: idx:type [x1,y1,x2,y2]
        label = f"{idx}: {block_type} [{x1},{y1},{x2},{y2}]"
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(vis, (x1, max(y1-20, 0)), (x1+tw+4, max(y1-2, 0)), color, -1)
        # Draw label text
        cv2.putText(vis, label, (x1+2, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)
    print(f"✅ Annotated visualization saved: {out_path}")
    print(f"Each box shows: index, type, [xmin, ymin, xmax, ymax]")

if __name__ == "__main__":
    main()
