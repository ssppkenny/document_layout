#!/usr/bin/env python3
"""
Test script to demonstrate image normalization with Otsu binarization
"""

import cv2
import sys
import os
import numpy as np

# Add src/ocr_reflow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ocr_reflow'))

from binarization import normalize_image, binarize_document

def test_normalization():
    """Test image normalization and complete preprocessing pipeline"""

    test_image = 'images/dvurog_p017.png'

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found")
        return False

    print("="*70)
    print("Testing Image Normalization + Otsu Binarization Pipeline")
    print("="*70)

    # Load original image
    print(f"\n1. Loading original image: {test_image}")
    img_original = cv2.imread(test_image)
    print(f"   ✓ Original: {img_original.shape}")
    print(f"   ✓ Intensity range: [{img_original.min()}, {img_original.max()}]")
    print(f"   ✓ Mean intensity: {img_original.mean():.2f}")

    # Step 1: Normalize
    print(f"\n2. Applying normalization (NORM_MINMAX, range [0, 255])")
    img_normalized = normalize_image(img_original, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print(f"   ✓ Normalized: {img_normalized.shape}")
    print(f"   ✓ Intensity range: [{img_normalized.min()}, {img_normalized.max()}]")
    print(f"   ✓ Mean intensity: {img_normalized.mean():.2f}")

    # Step 2: Apply Otsu binarization
    print(f"\n3. Applying Otsu binarization")
    binary_img = binarize_document(img_normalized, method='otsu')
    print(f"   ✓ Binary: {binary_img.shape}")
    print(f"   ✓ Values: min={binary_img.min()}, max={binary_img.max()}")
    print(f"   ✓ Mean: {binary_img.mean():.2f} ({'mostly black' if binary_img.mean() < 128 else 'mostly white'})")

    # Step 3: Convert to BGR
    print(f"\n4. Converting to BGR")
    binary_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    print(f"   ✓ Binary BGR: {binary_bgr.shape}")

    # Step 4: Add to normalized original
    print(f"\n5. Adding binary to normalized original")
    enhanced_img = cv2.add(img_normalized, binary_bgr)
    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
    print(f"   ✓ Enhanced: {enhanced_img.shape}")
    print(f"   ✓ Intensity range: [{enhanced_img.min()}, {enhanced_img.max()}]")
    print(f"   ✓ Mean intensity: {enhanced_img.mean():.2f}")

    # Save all intermediate results
    print(f"\n6. Saving results")
    cv2.imwrite('/tmp/norm_test_1_original.png', img_original)
    print(f"   ✓ Saved: /tmp/norm_test_1_original.png")

    cv2.imwrite('/tmp/norm_test_2_normalized.png', img_normalized)
    print(f"   ✓ Saved: /tmp/norm_test_2_normalized.png")

    cv2.imwrite('/tmp/norm_test_3_binary.png', binary_img)
    print(f"   ✓ Saved: /tmp/norm_test_3_binary.png")

    cv2.imwrite('/tmp/norm_test_4_enhanced.png', enhanced_img)
    print(f"   ✓ Saved: /tmp/norm_test_4_enhanced.png")

    # Test other normalization types
    print(f"\n7. Testing other normalization types")

    # L2 normalization
    img_l2 = normalize_image(img_original, alpha=0, beta=1, norm_type=cv2.NORM_L2)
    img_l2_scaled = (img_l2 * 255).astype(np.uint8)
    cv2.imwrite('/tmp/norm_test_5_l2.png', img_l2_scaled)
    print(f"   ✓ L2 norm saved: /tmp/norm_test_5_l2.png")

    # Compare with non-normalized
    print(f"\n8. Comparison: Normalized vs Non-normalized")
    binary_direct = binarize_document(img_original, method='otsu')
    enhanced_direct = cv2.add(img_original, cv2.cvtColor(binary_direct, cv2.COLOR_GRAY2BGR))
    enhanced_direct = np.clip(enhanced_direct, 0, 255).astype(np.uint8)

    cv2.imwrite('/tmp/norm_test_6_without_norm.png', enhanced_direct)
    print(f"   ✓ Without normalization: /tmp/norm_test_6_without_norm.png")

    # Calculate difference
    diff = cv2.absdiff(enhanced_img, enhanced_direct)
    diff_sum = diff.sum()
    print(f"   ✓ Difference between normalized and non-normalized: {diff_sum:,} (total pixel difference)")

    if diff_sum > 0:
        cv2.imwrite('/tmp/norm_test_7_difference.png', diff)
        print(f"   ✓ Difference image saved: /tmp/norm_test_7_difference.png")

    print("\n" + "="*70)
    print("✓ Normalization test completed successfully!")
    print("="*70)
    print("\nPipeline steps:")
    print("  1. Load original image")
    print("  2. Normalize (brings pixel values to full [0, 255] range)")
    print("  3. Apply Otsu binarization (automatic threshold)")
    print("  4. Add binarized image to normalized original (contrast enhancement)")
    print("\nOutput files in /tmp:")
    print("  - norm_test_1_original.png")
    print("  - norm_test_2_normalized.png")
    print("  - norm_test_3_binary.png (Otsu)")
    print("  - norm_test_4_enhanced.png (final result)")
    print("  - norm_test_5_l2.png (L2 normalization)")
    print("  - norm_test_6_without_norm.png (comparison)")
    if diff_sum > 0:
        print("  - norm_test_7_difference.png (diff between normalized and non-normalized)")
    print("="*70)

    return True

if __name__ == '__main__':
    success = test_normalization()
    sys.exit(0 if success else 1)
