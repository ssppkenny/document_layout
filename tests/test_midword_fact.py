#!/usr/bin/env python3
"""
Test actual mid-word split where 'f' and 'act' have NO space between them
"""
import sys
sys.path.insert(0, '/src')

from reflow import Letter, create_page_with_word_wrapping
import numpy as np
import cv2

def create_test_case():
    """Create test where 'word fact' would cause 'fact' itself to split as 'f' + 'act'."""
    lines = []

    # Line 1: "word fact" - where page allows "word f" but splits "fact" mid-word
    line1 = [
        Letter(xmin=10, ymin=10, xmax=30, ymax=30, bl=5),   # w
        Letter(xmin=31, ymin=10, xmax=51, ymax=30, bl=5),   # o
        Letter(xmin=52, ymin=10, xmax=72, ymax=30, bl=5),   # r
        Letter(xmin=73, ymin=10, xmax=93, ymax=30, bl=5),   # d
        Letter(xmin=115, ymin=10, xmax=135, ymax=30, bl=5), # f (space before = 22)
        Letter(xmin=136, ymin=10, xmax=156, ymax=30, bl=5), # a (NO space, part of "fact")
        Letter(xmin=157, ymin=10, xmax=177, ymax=30, bl=5), # c (NO space, part of "fact")
        Letter(xmin=178, ymin=10, xmax=198, ymax=30, bl=5), # t (NO space, part of "fact")
    ]
    lines.append(line1)

    # Create a simple original image
    original_image = np.ones((40, 210, 3), dtype=np.uint8) * 255
    cv2.putText(original_image, "word fact", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return lines, original_image

def main():
    print("Testing mid-word 'f' + 'act' split prevention...")
    print("=" * 70)

    lines, original_image = create_test_case()

    zoom_factor = 1.5
    # Each letter ~20*1.5 = 30 pixels
    # "word f" = 5 letters + 1 word space = ~172 pixels
    # Set available to ~180 so "word f" fits but not "word fa"
    new_page_width = 220  # available = 180

    print(f"Zoom factor: {zoom_factor}")
    print(f"New page width: {new_page_width}")
    print(f"Left margin: 20, Right margin: 20")
    print(f"Available width: {new_page_width - 40}")
    print()
    print("Expected behavior:")
    print("- 'fact' should NOT be split as 'f' + 'act' (1 + 3 letters)")
    print("- Should see: 'word' on line 1, 'fact' on line 2")
    print("=" * 70)
    print()

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )

    # Save and display
    cv2.imwrite("/tests/test_midword_f_act_result.png", page_reflowed)
    print()
    print("Result saved to: test_midword_f_act_result.png")
    print("Look for: 'Word split check at letter 5: 1 letters on current line, 3 letters on next line - PREVENTING split'")
    print("(Letter 5 is 'a', which would be the first letter on next line, splitting 'fact' as 'f'+'act')")

if __name__ == "__main__":
    main()
