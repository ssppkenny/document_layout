"""
Test script to verify text-region-based skew detection
"""
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ocr_reflow.layout import layout
from ocr_reflow.skew_detection import detect_skew, detect_skew_in_text_regions

def test_comparison(image_path):
    """Compare full-image vs text-region skew detection"""
    print(f"Testing: {image_path}")
    print("=" * 70)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Method 1: Full image detection (old method)
    print("\n1. Full-image skew detection (old method):")
    angle_full = detect_skew(img)
    print(f"   Detected: {angle_full:.2f}°")

    # Method 2: Text-region detection (new method)
    print("\n2. Text-region skew detection (new method):")
    try:
        layout_boxes = layout(image_path)
        text_boxes = [(geom, box_type) for geom, box_type in layout_boxes
                     if box_type in ["plain text", "title"]]
        print(f"   Found {len(text_boxes)} text regions (out of {len(layout_boxes)} total)")

        angle_text = detect_skew_in_text_regions(img, layout_boxes)
        print(f"   Detected: {angle_text:.2f}°")

        # Show difference
        diff = abs(angle_full - angle_text)
        print(f"\n3. Difference: {diff:.2f}°")
        if diff > 1.0:
            print(f"   ⚠️  Significant difference! Text-region method is more accurate.")
        else:
            print(f"   ✓ Both methods agree (difference < 1°)")

    except Exception as e:
        print(f"   Error: {e}")

    print("=" * 70)
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Test on default images
        test_images = [
            "images/sedg_p598.png",
            "images/dvurog_p017.png",
            "images/dvurog_p021.png",
        ]

        print("Testing text-region-based skew detection")
        print("=" * 70)
        print()

        for img_path in test_images:
            if os.path.exists(img_path):
                test_comparison(img_path)
            else:
                print(f"Warning: {img_path} not found, skipping")
    else:
        for img_path in sys.argv[1:]:
            test_comparison(img_path)
