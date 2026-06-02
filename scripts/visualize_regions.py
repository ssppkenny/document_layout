"""Visualize layout regions from the ensemble detector on an image.

Usage:
    pixi run python scripts/visualize_regions.py <image> [options]

Examples:
    pixi run python scripts/visualize_regions.py images/gagarin.jpg
    pixi run python scripts/visualize_regions.py images/gagarin.jpg --deskew
    pixi run python scripts/visualize_regions.py images/gagarin.jpg --deskew --output /tmp/viz.png
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ocr_reflow.layout import layout_from_array
from ocr_reflow.skew_detection import detect_and_correct_skew

TYPE_COLORS = {
    "plain text":          (200,  80,   0),
    "title":               (  0, 140, 255),
    "figure":              ( 80,  80, 200),
    "figure_caption":      ( 80, 200, 200),
    "figure_and_caption":  ( 80, 200, 200),
    "table":               (120, 120,  80),
    "isolate_formula":     (120,  80, 120),
    "formula_caption":     (160, 100, 120),
    "abandon":             ( 80,  80,  80),
    "seal":                (  0, 200,   0),
    "titled_block_title":  (200,   0, 200),
    "titled_block_body":   (180,   0, 180),
}

DEFAULT_COLOR = (120, 120, 120)


def draw_regions(img, boxes, font_scale=0.5, thickness=2):
    vis = img.copy()
    counts = {}
    for geom, btype in boxes:
        xmin, ymin, xmax, ymax = map(int, geom.bounds)
        color = TYPE_COLORS.get(btype, DEFAULT_COLOR)
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(vis, btype, (xmin + 4, ymin + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        counts[btype] = counts.get(btype, 0) + 1
    return vis, counts


def main():
    parser = argparse.ArgumentParser(description="Visualize layout regions from ensemble detector")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--deskew", action="store_true",
                        help="Apply Hough-based skew correction before layout")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: <basename>_regions.png)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not load image: {args.image}")
        sys.exit(1)

    print(f"Loaded: {args.image}  ({img.shape[1]}x{img.shape[0]})")

    if args.deskew:
        print("Deskewing ...")
        img, skew_angle = detect_and_correct_skew(img, method="hough")
        print(f"  Skew angle: {skew_angle:.2f}°  (new size: {img.shape[1]}x{img.shape[0]})")

    print("Running ensemble layout ...")
    boxes = layout_from_array(img)
    print(f"  Detected {len(boxes)} regions")

    vis, counts = draw_regions(img, boxes)

    for t, c in sorted(counts.items()):
        print(f"    {t:20s} x{c}")

    out = args.output or os.path.splitext(args.image)[0] + "_regions.png"
    cv2.imwrite(out, vis)
    print(f"\nSaved: {out}  ({vis.shape[1]}x{vis.shape[0]})")


if __name__ == "__main__":
    main()
