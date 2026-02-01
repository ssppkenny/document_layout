#!/usr/bin/env python3
"""
Test script to verify the margins() IndexError bug fix.
Tests the specific case that was failing: images/kf_p015.png
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_margins_with_few_words():
    """Test margins() function with small number of words."""
    from ocr_reflow.main import margins

    print("Testing margins() function with edge cases...")

    # Test case 1: Empty words array
    print("\n1. Testing with 0 words...")
    words = np.array([])
    try:
        left, right = margins(words)
        print(f"   ✅ Passed: returned {len(left)} left margins, {len(right)} right margins")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test case 2: Single word
    print("\n2. Testing with 1 word...")
    words = np.array([[10, 10, 50, 30, 0.9]])
    try:
        left, right = margins(words)
        print(f"   ✅ Passed: returned {len(left)} left margins, {len(right)} right margins")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test case 3: Few words (the problematic case)
    print("\n3. Testing with 28 words (the original failing case)...")
    words = np.array([
        [i*20, 10, i*20+15, 25, 0.9] for i in range(28)
    ])
    try:
        left, right = margins(words)
        print(f"   ✅ Passed: returned {len(left)} left margins, {len(right)} right margins")
    except IndexError as e:
        print(f"   ❌ Failed with IndexError: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Failed with other error: {e}")
        return False

    # Test case 4: Normal number of words
    print("\n4. Testing with 100 words (normal case)...")
    words = np.array([
        [i*10, (i//10)*20, i*10+8, (i//10)*20+15, 0.9] for i in range(100)
    ])
    try:
        left, right = margins(words)
        print(f"   ✅ Passed: returned {len(left)} left margins, {len(right)} right margins")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    print("\n✅ All margin tests passed!")
    return True

def test_full_image_processing():
    """Test the full processing with the problematic image."""
    from ocr_reflow.main import process_document_with_layout

    image_path = "images/kf_p015.png"

    if not os.path.exists(image_path):
        print(f"\n⚠️  Warning: Test image not found: {image_path}")
        print("   Skipping full image test.")
        return True

    print(f"\nTesting full processing with: {image_path}")
    print("This is the image that originally caused the IndexError...")

    try:
        result = process_document_with_layout(image_path)
        print(f"✅ Success! Processed image without errors.")
        print(f"   Output dimensions: {result.shape[1]}x{result.shape[0]}")

        # Save the result
        import cv2
        output_path = "test_margins_fix_output.png"
        cv2.imwrite(output_path, result)
        print(f"   Output saved to: {output_path}")

        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Testing margins() IndexError Bug Fix")
    print("=" * 70)

    # Test margins function with edge cases
    test1_passed = test_margins_with_few_words()

    # Test full image processing
    if test1_passed:
        test2_passed = test_full_image_processing()
    else:
        test2_passed = False

    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        print("The IndexError bug has been successfully fixed!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the error messages above.")
    print("=" * 70)

    sys.exit(0 if (test1_passed and test2_passed) else 1)
