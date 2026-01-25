#!/usr/bin/env python3
"""
Direct test for 'f' + 'act' split
"""
import sys
sys.path.insert(0, '/src')

from reflow import Letter, create_page_with_word_wrapping
import numpy as np
import cv2

def create_test_case():
    """Create test where 'word f' on line, then 'act' would follow."""
    lines = []

    # Line 1: "is a fact" - where page is wide enough for "is a f" but not "is a fa"
    line1 = [
        Letter(xmin=10, ymin=10, xmax=25, ymax=30, bl=5),   # i
        Letter(xmin=26, ymin=10, xmax=41, ymax=30, bl=5),   # s
        Letter(xmin=60, ymin=10, xmax=75, ymax=30, bl=5),   # a (space before)
        Letter(xmin=95, ymin=10, xmax=110, ymax=30, bl=5),  # f (space before)
        Letter(xmin=111, ymin=10, xmax=126, ymax=30, bl=5), # a
        Letter(xmin=127, ymin=10, xmax=142, ymax=30, bl=5), # c
        Letter(xmin=143, ymin=10, xmax=158, ymax=30, bl=5), # t
    ]
    lines.append(line1)

    # Create a simple original image
    original_image = np.ones((40, 170, 3), dtype=np.uint8) * 255
    cv2.putText(original_image, "is a fact", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return lines, original_image

def main():
    print("Testing 'f' + 'act' split prevention directly...")
    print("=" * 70)

    lines, original_image = create_test_case()

    zoom_factor = 1.5
    # Each letter ~15*1.5 = 22.5 pixels
    # "is a f" = 5 letters + 2 word spaces (44 pixels) = ~157 pixels
    # Set available to ~120 so only "is a f" or even just "is a" fits
    new_page_width = 155  # available = 115, should force split

    print(f"Zoom factor: {zoom_factor}")
    print(f"New page width: {new_page_width}")
    print(f"Left margin: 20, Right margin: 20")
    print(f"Available width: {new_page_width - 40}")
    print()
    print("Expected behavior:")
    print("- 'fact' should NOT be split as 'f' + 'act' (1 + 3 letters)")
    print("- Should see: 'is a' on line 1, 'fact' on line 2")
    print("=" * 70)
    print()

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )

    # Save and display
    cv2.imwrite("/tests/test_f_act_result.png", page_reflowed)
    print()
    print("Result saved to: test_f_act_result.png")
    print("Look for: 'Word split check at letter 3: 1 letters on current line, 3 letters on next line - PREVENTING split'")

if __name__ == "__main__":
    main()
