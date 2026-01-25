#!/usr/bin/env python3
"""
Test script to verify that words are not split when only 1 letter would remain on either line.
"""
import sys
sys.path.insert(0, '/src')

from reflow import Letter, create_page_with_word_wrapping
import numpy as np
import cv2

# Create a test case where a word would be split with only 1 letter on one line
# Word: "testing" (7 letters)
# Scenario: If we have limited space, we should not split as "t" + "esting" or "testin" + "g"

def create_test_case():
    """Create test lines with a word that might be split badly."""
    lines = []

    # Line 1: "The quick"
    line1 = [
        Letter(xmin=10, ymin=10, xmax=25, ymax=30, bl=5),   # T
        Letter(xmin=26, ymin=10, xmax=41, ymax=30, bl=5),   # h
        Letter(xmin=42, ymin=10, xmax=57, ymax=30, bl=5),   # e
        Letter(xmin=70, ymin=10, xmax=85, ymax=30, bl=5),   # q (space before)
        Letter(xmin=86, ymin=10, xmax=101, ymax=30, bl=5),  # u
        Letter(xmin=102, ymin=10, xmax=117, ymax=30, bl=5), # i
        Letter(xmin=118, ymin=10, xmax=133, ymax=30, bl=5), # c
        Letter(xmin=134, ymin=10, xmax=149, ymax=30, bl=5), # k
    ]
    lines.append(line1)

    # Line 2: "testing" - this word might be split
    line2 = [
        Letter(xmin=10, ymin=40, xmax=25, ymax=60, bl=5),   # t
        Letter(xmin=26, ymin=40, xmax=41, ymax=60, bl=5),   # e
        Letter(xmin=42, ymin=40, xmax=57, ymax=60, bl=5),   # s
        Letter(xmin=58, ymin=40, xmax=73, ymax=60, bl=5),   # t
        Letter(xmin=74, ymin=40, xmax=89, ymax=60, bl=5),   # i
        Letter(xmin=90, ymin=40, xmax=105, ymax=60, bl=5),  # n
        Letter(xmin=106, ymin=40, xmax=121, ymax=60, bl=5), # g
    ]
    lines.append(line2)

    # Create a simple original image
    original_image = np.ones((70, 200, 3), dtype=np.uint8) * 255
    cv2.putText(original_image, "The quick", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(original_image, "testing", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return lines, original_image

def main():
    print("Testing word split prevention...")
    print("=" * 60)

    lines, original_image = create_test_case()

    # Use a narrow page width that would normally force a bad split
    # With zoom_factor=1.5, each letter is ~15*1.5 = 22.5 pixels wide
    # "The quick" = 8 letters * 22.5 = 180 pixels
    # If we set page width to allow "The qui" but would split "testing" as "t"+"esting"
    # That should now be prevented

    zoom_factor = 1.5
    new_page_width = 180  # Narrow width to force wrapping

    print(f"Zoom factor: {zoom_factor}")
    print(f"New page width: {new_page_width}")
    print(f"Left margin: 20, Right margin: 20")
    print(f"Available width: {new_page_width - 40}")
    print()
    print("Expected behavior:")
    print("- Words should NOT be split with only 1 letter on either line")
    print("- If 'testing' would be split as 't' + 'esting', move entire word to next line")
    print("- If 'testing' would be split as 'testin' + 'g', move entire word to next line")
    print("=" * 60)

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )

    # Save and display
    cv2.imwrite("/test_word_split_result.png", page_reflowed)
    print()
    print("Result saved to: test_word_split_result.png")
    print("Check the console output above to see if word splitting was prevented.")

if __name__ == "__main__":
    main()
