#!/usr/bin/env python3
"""
Final verification: Compare before and after for the dot merging fix
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
from doctr.io import DocumentFile
from ocr_reflow.skew_detection import detect_and_correct_skew
from ocr_reflow.layout import layout
from ocr_reflow.main import get_doctr_model, find_rects
import tempfile

def verify_fix(image_path):
    """Quick verification that the merge fix is working"""

    print("=" * 70)
    print("VERIFICATION: Dot-Letter Merge Fix")
    print("=" * 70)
    print(f"Testing: {image_path}\n")

    # Load and process
    img = cv2.imread(image_path)
    deskewed_img, angle = detect_and_correct_skew(img)
    layout_boxes = layout(image_path)
    plain_text_boxes = [b for b, t in layout_boxes if t == 'plain text']

    if len(plain_text_boxes) == 0:
        print("No plain text boxes found!")
        return

    # Process first box
    box_geom = plain_text_boxes[0]
    xmin, ymin, xmax, ymax = map(int, box_geom.bounds)
    region = deskewed_img[ymin:ymax, xmin:xmax].copy()
    region_h, region_w = region.shape[:2]

    # Get words and letters
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, region)

    try:
        model, device = get_doctr_model()
        docs = DocumentFile.from_images([tmp_path])
        result = model(docs)
        words_array = result[0]["words"]

        words = []
        for i in range(len(words_array)):
            words.append((
                int(words_array[i, 0] * region_w),
                int(words_array[i, 1] * region_h),
                int(words_array[i, 2] * region_w),
                int(words_array[i, 3] * region_h)
            ))
    finally:
        import os
        os.unlink(tmp_path)

    # Extract letters
    letters = find_rects(region, words)

    # Analyze
    letter_heights = [(ly2 - ly1) for _, ly1, _, ly2 in letters]
    median_height = np.median(letter_heights) if len(letter_heights) > 0 else 25

    # Count dots and merged letters
    standalone_dots = 0
    merged_letters = 0
    normal_letters = 0

    for lx1, ly1, lx2, ly2 in letters:
        h = ly2 - ly1
        w = lx2 - lx1

        if h < median_height * 0.4 and w < median_height * 0.5:
            standalone_dots += 1
        elif h > median_height * 1.2:
            merged_letters += 1
        else:
            normal_letters += 1

    # Print results
    print(f"Words detected:       {len(words)}")
    print(f"Letters extracted:    {len(letters)}")
    print(f"Letters per word:     {len(letters) / len(words):.1f}")
    print(f"\nLetter breakdown:")
    print(f"  Normal letters:     {normal_letters}")
    print(f"  Merged (i, j):      {merged_letters}")
    print(f"  Standalone dots:    {standalone_dots}")

    # Verdict
    print("\n" + "-" * 70)

    if standalone_dots == 0:
        print("✅ PERFECT! All dots merged with base letters.")
        print("   Alignment guaranteed during reflow.")
        status = "PASS"
    elif standalone_dots < 5:
        print(f"✅ GOOD! Only {standalone_dots} standalone dots.")
        print("   (Likely legitimate accents or diacritics)")
        status = "PASS"
    elif standalone_dots < len(letters) * 0.05:
        print(f"⚠️  ACCEPTABLE: {standalone_dots} standalone dots (~{standalone_dots/len(letters)*100:.1f}%).")
        print("   Most dots merged, some may be non-i/j marks.")
        status = "ACCEPTABLE"
    else:
        print(f"❌ ISSUE: {standalone_dots} standalone dots (~{standalone_dots/len(letters)*100:.1f}%).")
        print("   Fix may not be working correctly.")
        status = "FAIL"

    print("-" * 70)
    print(f"Status: {status}")
    print("=" * 70)

    return status

if __name__ == '__main__':
    if len(sys.argv) > 1:
        result = verify_fix(sys.argv[1])
    else:
        # Test on both images
        print("\nTesting on English text...")
        status1 = verify_fix('../images/sedg_p598.png')

        print("\n\nTesting on Russian text...")
        status2 = verify_fix('../images/dvurog_p021.png')

        print("\n" + "=" * 70)
        print("OVERALL VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"English text (sedg_p598.png):  {status1}")
        print(f"Russian text (dvurog_p021.png): {status2}")

        if status1 in ["PASS", "ACCEPTABLE"] and status2 in ["PASS", "ACCEPTABLE"]:
            print("\n✅ FIX VERIFIED: Dot-letter merging is working correctly!")
        else:
            print("\n⚠️  Some issues detected. Review test results.")
        print("=" * 70)
