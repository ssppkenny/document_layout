#!/usr/bin/env python3
"""
Simple test to verify Otsu binarization with addition to original works
"""

import cv2
import sys
import os
import numpy as np

# Add src/ocr_reflow to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ocr_reflow'))

from binarization import binarize_document

def test_workflow():
    """Test the complete Otsu + original workflow"""

    test_image = 'images/dvurog_p017.png'

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found")
        return False

    # Load original image
    print(f"Loading {test_image}...")
    img = cv2.imread(test_image)
    print(f"✓ Original image: {img.shape}")

    # Apply Otsu binarization
    print("\nApplying Otsu binarization...")
    binary = binarize_document(img, method='otsu')
    print(f"✓ Binary image: {binary.shape}, min={binary.min()}, max={binary.max()}, mean={binary.mean():.1f}")

    # Convert to BGR
    print("\nConverting to BGR...")
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    print(f"✓ Binary BGR: {binary_bgr.shape}")

    # Add to original
    print("\nAdding binarized to original...")
    enhanced = cv2.add(img, binary_bgr)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    print(f"✓ Enhanced image: {enhanced.shape}")

    # Save results
    cv2.imwrite('/tmp/workflow_original.png', img)
    cv2.imwrite('/tmp/workflow_binary.png', binary)
    cv2.imwrite('/tmp/workflow_enhanced.png', enhanced)

    print("\n" + "="*60)
    print("✓ Workflow test successful!")
    print("Output files:")
    print("  - /tmp/workflow_original.png")
    print("  - /tmp/workflow_binary.png (Otsu binarized)")
    print("  - /tmp/workflow_enhanced.png (original + binary)")
    print("="*60)

    return True

if __name__ == '__main__':
    success = test_workflow()
    sys.exit(0 if success else 1)
