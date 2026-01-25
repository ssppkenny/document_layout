#!/usr/bin/env python3
"""
Test script to verify that "fact" is not split as "f" + "act"
"""
import sys
sys.path.insert(0, '/src')

from reflow import Letter, create_page_with_word_wrapping
import numpy as np
import cv2

def create_test_case():
    """Create test lines where 'fact' might be split as 'f' + 'act'."""
    lines = []

    # Line 1: "The word" - should fill most of the line
    line1 = [
        Letter(xmin=10, ymin=10, xmax=25, ymax=30, bl=5),   # T
        Letter(xmin=26, ymin=10, xmax=41, ymax=30, bl=5),   # h
        Letter(xmin=42, ymin=10, xmax=57, ymax=30, bl=5),   # e
        Letter(xmin=70, ymin=10, xmax=85, ymax=30, bl=5),   # w (space before)
        Letter(xmin=86, ymin=10, xmax=101, ymax=30, bl=5),  # o
        Letter(xmin=102, ymin=10, xmax=117, ymax=30, bl=5), # r
        Letter(xmin=118, ymin=10, xmax=133, ymax=30, bl=5), # d
    ]
    lines.append(line1)

    # Line 2: "fact" - this word should NOT be split as "f" + "act"
    line2 = [
        Letter(xmin=10, ymin=40, xmax=25, ymax=60, bl=5),   # f
        Letter(xmin=26, ymin=40, xmax=41, ymax=60, bl=5),   # a
        Letter(xmin=42, ymin=40, xmax=57, ymax=60, bl=5),   # c
        Letter(xmin=58, ymin=40, xmax=73, ymax=60, bl=5),   # t
    ]
    lines.append(line2)

    # Create a simple original image
    original_image = np.ones((70, 200, 3), dtype=np.uint8) * 255
    cv2.putText(original_image, "The word", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(original_image, "fact", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return lines, original_image

def main():
    print("Testing 'fact' word split prevention...")
    print("=" * 60)

    lines, original_image = create_test_case()

    # Use parameters that would normally force "fact" to split as "f" + "act"
    zoom_factor = 1.5
    # Calculate: each letter ~15 pixels * 1.5 = 22.5 pixels
    # "The word" = 7 letters = ~157 pixels
    # Set available width so that "f" fits but not "fa"
    # With margins of 20 each, available = 120 pixels
    # This should force just "f" on one line unless we prevent it
    new_page_width = 160  # Very narrow to force the split

    print(f"Zoom factor: {zoom_factor}")
    print(f"New page width: {new_page_width}")
    print(f"Left margin: 20, Right margin: 20")
    print(f"Available width: {new_page_width - 40}")
    print()
    print("Expected behavior:")
    print("- 'fact' should NOT be split as 'f' + 'act' (1 + 3 letters)")
    print("- Entire word 'fact' should move to next line")
    print("=" * 60)
    print()

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )

    # Save and display
    cv2.imwrite("/tests/test_fact_split_result.png", page_reflowed)
    print()
    print("Result saved to: test_fact_split_result.png")
    print("Check the console output above to see if word splitting was prevented.")
    print()
    print("Look for messages like:")
    print("  'Word split check at letter X: 1 letters on current line, 3 letters on next line'")
    print("  'Preventing single-letter split at letter X - moving word to next line'")

if __name__ == "__main__":
    main()
