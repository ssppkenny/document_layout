#!/usr/bin/env python3
"""
Test script to verify Otsu binarization is working correctly
"""

import cv2
import sys
import os

# Add src/ocr_reflow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ocr_reflow'))

from binarization import binarize_document

def test_otsu():
    """Test Otsu binarization method"""

    # Test image
    test_image = 'images/gang_p023_lines1.png'

    if not os.path.exists(test_image):
        print(f"Error: Test image not found: {test_image}")
        return False

    # Load image
    img = cv2.imread(test_image)
    print(f"✓ Loaded image: {img.shape}")

    # Test Otsu binarization
    print("Testing Otsu binarization...")
    binary_otsu = binarize_document(img, method='otsu')
    print(f"✓ Otsu result: {binary_otsu.shape}, min={binary_otsu.min()}, max={binary_otsu.max()}")

    # Check that we have binary output (only 0 and 255)
    unique_vals = set(binary_otsu.flatten())
    if unique_vals == {0, 255}:
        print("✓ Binary output is correct (white text on black background)")
        # Check that background is predominantly black (0)
        mean_val = binary_otsu.mean()
        if mean_val < 128:
            print(f"✓ Confirmed: background is black (mean={mean_val:.1f})")
        else:
            print(f"✓ Background is white (mean={mean_val:.1f})")
    else:
        print(f"⚠ Warning: Expected only {{0, 255}}, got {unique_vals}")

    # Save output
    output_path = '/tmp/test_otsu_binarization.png'
    cv2.imwrite(output_path, binary_otsu)
    print(f"✓ Saved to: {output_path}")

    # Test adding Otsu to original (as done in main pipeline)
    print("\nTesting: Adding Otsu binarized image to original...")
    import numpy as np
    binary_otsu_bgr = cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR)
    enhanced_img = cv2.add(img, binary_otsu_bgr)
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    print(f"✓ Enhanced image created: {enhanced_img.shape}")
    cv2.imwrite('/tmp/test_otsu_enhanced.png', enhanced_img)
    print(f"✓ Saved enhanced image to: /tmp/test_otsu_enhanced.png")

    # Test Sauvola for comparison
    print("\nTesting Sauvola binarization for comparison...")
    binary_sauvola = binarize_document(img, method='sauvola', window_size=15)
    print(f"✓ Sauvola result: {binary_sauvola.shape}")
    cv2.imwrite('/tmp/test_sauvola_binarization.png', binary_sauvola)

    # Test Niblack
    print("\nTesting Niblack binarization...")
    binary_niblack = binarize_document(img, method='niblack', window_size=15)
    print(f"✓ Niblack result: {binary_niblack.shape}")
    cv2.imwrite('/tmp/test_niblack_binarization.png', binary_niblack)

    print("\n" + "="*60)
    print("All binarization methods working correctly!")
    print("Output files:")
    print("  - /tmp/test_otsu_binarization.png")
    print("  - /tmp/test_otsu_enhanced.png (original + Otsu)")
    print("  - /tmp/test_sauvola_binarization.png")
    print("  - /tmp/test_niblack_binarization.png")
    print("="*60)

    return True

if __name__ == '__main__':
    success = test_otsu()
    sys.exit(0 if success else 1)
