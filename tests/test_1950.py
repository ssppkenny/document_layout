#!/usr/bin/env python3
"""
Test for number "1950" split as "195" + "0"
"""
import sys
sys.path.insert(0, '/src')

from reflow import Letter, create_page_with_word_wrapping
import numpy as np
import cv2

def create_test_case():
    """Create test where '1950' would split as '195' + '0'."""
    lines = []

    # Line 1: "year 1950" - where "1950" might be split
    line1 = [
        Letter(xmin=10, ymin=10, xmax=30, ymax=30, bl=5),   # y
        Letter(xmin=31, ymin=10, xmax=51, ymax=30, bl=5),   # e
        Letter(xmin=52, ymin=10, xmax=72, ymax=30, bl=5),   # a
        Letter(xmin=73, ymin=10, xmax=93, ymax=30, bl=5),   # r
        Letter(xmin=115, ymin=10, xmax=135, ymax=30, bl=5), # 1 (space before = 22)
        Letter(xmin=136, ymin=10, xmax=156, ymax=30, bl=5), # 9 (NO space)
        Letter(xmin=157, ymin=10, xmax=177, ymax=30, bl=5), # 5 (NO space)
        Letter(xmin=178, ymin=10, xmax=198, ymax=30, bl=5), # 0 (NO space)
    ]
    lines.append(line1)

    # Create a simple original image
    original_image = np.ones((40, 210, 3), dtype=np.uint8) * 255
    cv2.putText(original_image, "year 1950", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return lines, original_image

def main():
    print("Testing '1950' split as '195' + '0'...")
    print("=" * 70)

    lines, original_image = create_test_case()

    zoom_factor = 1.5
    # Each digit ~20*1.5 = 30 pixels
    # "195" = 3 digits = ~90 pixels
    # Set available to ~100 so "195" fits but not "1950"
    new_page_width = 140  # available = 100

    print(f"Zoom factor: {zoom_factor}")
    print(f"New page width: {new_page_width}")
    print(f"Left margin: 20, Right margin: 20")
    print(f"Available width: {new_page_width - 40}")
    print()
    print("Expected behavior:")
    print("- '1950' should NOT be split as '195' + '0' (3 + 1 digits)")
    print("- Should see: 'year' on line 1, '1950' on line 2")
    print("=" * 70)
    print()

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )

    # Save and display
    cv2.imwrite("/tests/test_1950_result.png", page_reflowed)
    print()
    print("Result saved to: test_1950_result.png")

if __name__ == "__main__":
    main()
