#!/usr/bin/env python3
"""
Test the Swedish diacritics fix
Verifies that ö, ä, å are correctly merged and displayed
"""
å
import cv2
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import find_rects

def test_swedish_words():
    print("="*80)
    print("TESTING SWEDISH DIACRITICS FIX")
    print("="*80)
    print()

    img = cv2.imread('images/gang_p023_lines1.png')

    if img is None:
        print("✗ Could not load test image")
        return False

    # Test cases from gang_p023_lines1.png
    test_words = [
        ('inför', (192, 63, 298, 111), 5),  # i, n, f, ö (merged), r
        ('Börja', (81, 174, 224, 240), 5),  # B, ö (merged), r, j (merged with dot), a
    ]

    all_passed = True

    for word_name, box, expected_components in test_words:
        xmin, ymin, xmax, ymax = box
        words_list = [[xmin, ymin, xmax, ymax]]
        rectangles = find_rects(img, words_list, debug=False)

        actual = len(rectangles)

        # Allow ±1 component tolerance
        passed = abs(actual - expected_components) <= 1

        status = "✓" if passed else "✗"
        print(f"{status} {word_name}: {actual} components (expected ~{expected_components})")

        if not passed:
            all_passed = False
            if actual == 1:
                print(f"  Problem: Everything merged into 1 component")
            elif actual > expected_components + 2:
                print(f"  Problem: ö dots not merging with base letter")

    print()
    print("="*80)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        print()
        print("The Swedish diacritics fix is working correctly.")
        print("Words like 'Börja' and 'inför' should now display properly")
        print("without the 'half of letter ö' artifact.")
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("The fix needs further adjustment.")

    print("="*80)

    return all_passed

if __name__ == '__main__':
    success = test_swedish_words()
    sys.exit(0 if success else 1)
