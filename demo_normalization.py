#!/usr/bin/env python3
"""
Simple demonstration of normalization effect
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ocr_reflow'))
from binarization import normalize_image

def demo_normalization():
    """Demonstrate the effect of normalization on a test image"""

    test_img = 'images/dvurog_p017.png'

    if not os.path.exists(test_img):
        print(f"Error: {test_img} not found")
        return False

    print("Image Normalization Demonstration")
    print("="*50)

    # Load image
    img = cv2.imread(test_img)
    print(f"\n✓ Loaded: {test_img}")
    print(f"  Shape: {img.shape}")
    print(f"  Original range: [{img.min()}, {img.max()}]")
    print(f"  Original mean: {img.mean():.2f}")

    # Apply normalization
    print(f"\n✓ Applying cv2.normalize(NORM_MINMAX)...")
    normalized = normalize_image(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print(f"  Normalized range: [{normalized.min()}, {normalized.max()}]")
    print(f"  Normalized mean: {normalized.mean():.2f}")

    # Calculate difference
    diff = cv2.absdiff(img, normalized)
    diff_sum = int(diff.sum())

    print(f"\n✓ Difference: {diff_sum:,} (total absolute pixel difference)")

    # Save results
    cv2.imwrite('/tmp/demo_original.png', img)
    cv2.imwrite('/tmp/demo_normalized.png', normalized)

    if diff_sum > 0:
        # Scale difference for visibility
        diff_scaled = (diff.astype(float) / diff.max() * 255).astype(np.uint8)
        cv2.imwrite('/tmp/demo_difference.png', diff_scaled)
        print(f"\n✓ Saved files:")
        print(f"  - /tmp/demo_original.png")
        print(f"  - /tmp/demo_normalized.png")
        print(f"  - /tmp/demo_difference.png (scaled for visibility)")
    else:
        print(f"\n✓ Saved files:")
        print(f"  - /tmp/demo_original.png")
        print(f"  - /tmp/demo_normalized.png")
        print(f"  Note: No difference - image was already normalized")

    print("\n" + "="*50)
    print("Normalization brings pixel values to full [0, 255]")
    print("range, improving contrast for OCR and binarization.")
    print("="*50)

    return True

if __name__ == '__main__':
    success = demo_normalization()
    sys.exit(0 if success else 1)
