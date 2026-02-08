#!/usr/bin/env python3
"""
Check if layout boxes have rotation information
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ocr_reflow.layout import layout

def check_box_geometry(image_path='images/jtg_p033.png'):
    """Check layout box geometry details"""

    print("=" * 70)
    print("CHECKING LAYOUT BOX GEOMETRY")
    print("=" * 70)

    layout_boxes = layout(image_path)
    title_boxes = [(b, t) for b, t in layout_boxes if t == 'title']
    title_boxes_sorted = sorted(title_boxes, key=lambda item: item[0].bounds[1])

    for i, (box_geom, box_type) in enumerate(title_boxes_sorted):
        print(f"\nTitle {i}: {box_type}")
        print(f"  Type: {type(box_geom)}")
        print(f"  Bounds: {box_geom.bounds}")
        print(f"  Geometry type: {box_geom.geom_type}")

        # Check if it's a Polygon (could be rotated) vs a box
        if hasattr(box_geom, 'exterior'):
            coords = list(box_geom.exterior.coords)
            print(f"  Exterior coordinates ({len(coords)} points):")
            for j, coord in enumerate(coords[:5]):  # Show first 5
                print(f"    {j}: {coord}")

            # Check if it's axis-aligned
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            # For axis-aligned box, should have only 2 unique X and 2 unique Y values
            unique_xs = set([round(x, 1) for x in xs])
            unique_ys = set([round(y, 1) for y in ys])

            print(f"  Unique X values: {len(unique_xs)}")
            print(f"  Unique Y values: {len(unique_ys)}")

            if len(unique_xs) == 2 and len(unique_ys) == 2:
                print(f"  ✓ Box is AXIS-ALIGNED (not rotated)")
            else:
                print(f"  ⚠️  Box might be ROTATED (not axis-aligned)")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    check_box_geometry()
