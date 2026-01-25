#!/usr/bin/env python3
"""
Test script to verify that "fact" is not split as "f" + "act" more realistically
"""
import sys
sys.path.insert(0, '/src')

from reflow import Letter, create_page_with_word_wrapping
import numpy as np
import cv2

def create_test_case():
    """Create test lines where 'some other fact' would cause 'fact' to split as 'f' + 'act'."""
    lines = []

    # Line 1: "some other fact" - where "fact" might be split
    # Spacing: "some" (space) "other" (space) "fact"
    line1 = [
        Letter(xmin=10, ymin=10, xmax=25, ymax=30, bl=5),   # s
        Letter(xmin=26, ymin=10, xmax=41, ymax=30, bl=5),   # o
        Letter(xmin=42, ymin=10, xmax=57, ymax=30, bl=5),   # m
        Letter(xmin=58, ymin=10, xmax=73, ymax=30, bl=5),   # e
        Letter(xmin=90, ymin=10, xmax=105, ymax=30, bl=5),  # o (space before)
        Letter(xmin=106, ymin=10, xmax=121, ymax=30, bl=5), # t
        Letter(xmin=122, ymin=10, xmax=137, ymax=30, bl=5), # h
        Letter(xmin=138, ymin=10, xmax=153, ymax=30, bl=5), # e
        Letter(xmin=154, ymin=10, xmax=169, ymax=30, bl=5), # r
        Letter(xmin=186, ymin=10, xmax=201, ymax=30, bl=5), # f (space before)
        Letter(xmin=202, ymin=10, xmax=217, ymax=30, bl=5), # a
        Letter(xmin=218, ymin=10, xmax=233, ymax=30, bl=5), # c
        Letter(xmin=234, ymin=10, xmax=249, ymax=30, bl=5), # t
    ]
    lines.append(line1)

    # Create a simple original image
    original_image = np.ones((40, 260, 3), dtype=np.uint8) * 255
    cv2.putText(original_image, "some other fact", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return lines, original_image

def main():
    print("Testing 'fact' word split prevention (realistic scenario)...")
    print("=" * 70)

    lines, original_image = create_test_case()

    # Use parameters that would normally force "fact" to split as "f" + "act"
    zoom_factor = 1.5
    # Calculate: each letter ~15 pixels * 1.5 = 22.5 pixels
    # "some other f" = 11 letters + 2 spaces = ~293 pixels
    # Set available width so that "some other f" fits but not "fa"
    # With margins of 20 each, we need page width that allows "some other f" but not "fa"
    # Available = page_width - 40
    # "some other f" = approx 300 pixels, so need available = 300, page_width = 340
    new_page_width = 345  # Should force "f" on one line, "act" on next if not prevented

    print(f"Zoom factor: {zoom_factor}")
    print(f"New page width: {new_page_width}")
    print(f"Left margin: 20, Right margin: 20")
    print(f"Available width: {new_page_width - 40}")
    print()
    print("Expected behavior:")
    print("- 'fact' should NOT be split as 'f' + 'act' (1 + 3 letters)")
    print("- Entire word 'fact' should move to next line")
    print("- Should see: 'some other' on line 1, 'fact' on line 2")
    print("=" * 70)
    print()

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )

    # Save and display
    cv2.imwrite("/tests/test_fact_realistic_result.png", page_reflowed)
    print()
    print("Result saved to: test_fact_realistic_result.png")
    print("Check the console output above to see if word splitting was prevented.")
    print()
    print("Look for a message like:")
    print("  'Word split check at letter 9: 1 letters on current line, 3 letters on next line - PREVENTING split'")
    print("  (where letter 9 is the 'f' in 'fact')")

if __name__ == "__main__":
    main()
