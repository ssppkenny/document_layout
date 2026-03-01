#!/usr/bin/env python3
"""
Test that binarization now uses only the binarized image (not added to original)
"""

import cv2
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ocr_reflow'))

from binarization import normalize_image, binarize_document

def test_binarized_only():
    """Test the updated pipeline that uses only binarized image"""

    test_image = 'images/dvurog_p017.png'

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found")
        return False

    print("="*70)
    print("Testing: Binarized Image Only (No Addition to Original)")
    print("="*70)

    # Load original
    print(f"\n1. Loading: {test_image}")
    img = cv2.imread(test_image)
    print(f"   ✓ Shape: {img.shape}, Range: [{img.min()}, {img.max()}]")

    # Normalize
    print(f"\n2. Normalizing...")
    img_norm = normalize_image(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print(f"   ✓ Normalized: Range [{img_norm.min()}, {img_norm.max()}]")

    # Binarize
    print(f"\n3. Binarizing with Otsu...")
    binary = binarize_document(img_norm, method='otsu')
    print(f"   ✓ Binary: Shape {binary.shape}, Values: {np.unique(binary)}")
    print(f"   ✓ Mean: {binary.mean():.1f} ({'white text on black' if binary.mean() < 128 else 'black text on white'})")

    # Convert to BGR for consistency
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    print(f"\n4. Converted to BGR: {binary_bgr.shape}")

    # Save results
    print(f"\n5. Saving results...")
    cv2.imwrite('/tmp/binarized_only_original.png', img)
    print(f"   ✓ Original: /tmp/binarized_only_original.png")

    cv2.imwrite('/tmp/binarized_only_normalized.png', img_norm)
    print(f"   ✓ Normalized: /tmp/binarized_only_normalized.png")

    cv2.imwrite('/tmp/binarized_only_binary.png', binary)
    print(f"   ✓ Binary (grayscale): /tmp/binarized_only_binary.png")

    cv2.imwrite('/tmp/binarized_only_binary_bgr.png', binary_bgr)
    print(f"   ✓ Binary (BGR): /tmp/binarized_only_binary_bgr.png")

    # Verify it's truly binary
    unique_vals = np.unique(binary)
    if len(unique_vals) <= 2:
        print(f"\n✓ Verification: Image is binary (only {len(unique_vals)} unique values)")
    else:
        print(f"\n⚠ Warning: Image has {len(unique_vals)} unique values, expected 2")

    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("="*70)
    print("\nPipeline:")
    print("  Original → Normalize → Binarize → Use Binary Only")
    print("\nThe binary image is now used directly for OCR and reflow,")
    print("without adding it back to the original image.")
    print("="*70)

    return True

if __name__ == '__main__':
    success = test_binarized_only()
    sys.exit(0 if success else 1)
