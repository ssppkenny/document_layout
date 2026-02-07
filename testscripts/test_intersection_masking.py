"""
Test script to verify that intersecting regions are properly masked out
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ocr_reflow.layout import layout

def test_intersection_masking(image_path):
    """Test that text boxes with intersecting non-text regions are properly masked"""
    print(f"Testing intersection masking on: {image_path}")
    print("=" * 70)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # Detect background color
    flat_img = img.reshape(-1, 3)
    background_color = np.median(flat_img, axis=0).astype(np.uint8)
    print(f"Background color (BGR): {background_color}")

    # Get layout
    layout_boxes = layout(image_path)
    layout_boxes_sorted = sorted(layout_boxes, key=lambda item: (item[0].bounds[1], item[0].bounds[0]))

    print(f"\nLayout boxes: {len(layout_boxes)}")

    # Identify text and non-text boxes
    text_boxes = [(g, t) for g, t in layout_boxes_sorted if t in ["plain text", "title"]]
    non_text_boxes = [(g, t) for g, t in layout_boxes_sorted if t not in ["plain text", "title"]]

    print(f"  Text boxes: {len(text_boxes)}")
    print(f"  Non-text boxes: {len(non_text_boxes)}")

    # Process each text box
    print(f"\nProcessing text boxes:")
    print("=" * 70)

    for idx, (box_geom, box_type) in enumerate(text_boxes):
        bounds = box_geom.bounds
        xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])

        print(f"\n{idx+1}. Text box ({box_type}):")
        print(f"   Position: ({xmin}, {ymin}) → ({xmax}, {ymax})")
        print(f"   Size: {xmax-xmin}×{ymax-ymin}px")

        # Extract region
        box_img = img[ymin:ymax, xmin:xmax].copy()
        original_box_img = box_img.copy()

        # Check for intersections and mask them
        intersections_found = 0
        total_masked_area = 0

        for other_geom, other_type in non_text_boxes:
            if box_geom.intersects(other_geom):
                intersection = box_geom.intersection(other_geom)
                inter_bounds = intersection.bounds

                # Convert to local coordinates
                local_xmin = max(0, int(inter_bounds[0] - xmin))
                local_ymin = max(0, int(inter_bounds[1] - ymin))
                local_xmax = min(box_img.shape[1], int(inter_bounds[2] - xmin))
                local_ymax = min(box_img.shape[0], int(inter_bounds[3] - ymin))

                if local_xmax > local_xmin and local_ymax > local_ymin:
                    # Mask it out
                    box_img[local_ymin:local_ymax, local_xmin:local_xmax] = background_color

                    masked_area = (local_xmax - local_xmin) * (local_ymax - local_ymin)
                    total_masked_area += masked_area
                    intersections_found += 1

                    print(f"   ✓ Masked intersection with {other_type}")
                    print(f"     Local coords: ({local_xmin}, {local_ymin}) → ({local_xmax}, {local_ymax})")
                    print(f"     Masked area: {masked_area:,} px² ({masked_area / (box_img.shape[0] * box_img.shape[1]) * 100:.1f}% of box)")

        if intersections_found > 0:
            total_box_area = box_img.shape[0] * box_img.shape[1]
            print(f"   Total: {intersections_found} intersections, {total_masked_area:,} px² masked ({total_masked_area / total_box_area * 100:.1f}% of box)")

            # Save comparison images
            output_dir = Path(image_path).parent / "debug_masking"
            output_dir.mkdir(exist_ok=True)

            orig_path = output_dir / f"text_box_{idx+1}_original.png"
            masked_path = output_dir / f"text_box_{idx+1}_masked.png"

            cv2.imwrite(str(orig_path), original_box_img)
            cv2.imwrite(str(masked_path), box_img)

            print(f"   Saved comparison: {orig_path.name} vs {masked_path.name}")
        else:
            print(f"   No intersections found")

    print("\n" + "=" * 70)
    print("✓ Test complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        test_image = "images/sedg_p598.png"
    else:
        test_image = sys.argv[1]

    test_intersection_masking(test_image)
