#!/usr/bin/env python3
"""
Test case to simulate huge line spacing caused by one outlier letter
"""
import sys
sys.path.insert(0, '/src')

from reflow import Letter, create_page_with_word_wrapping
import numpy as np
import cv2

def create_test_with_outlier():
    """Create test with normal letters and one outlier with huge baseline."""
    lines = []

    # Line 1: Normal text "hello"
    line1 = [
        Letter(xmin=10, ymin=10, xmax=30, ymax=30, bl=5),   # h
        Letter(xmin=31, ymin=10, xmax=51, ymax=30, bl=5),   # e
        Letter(xmin=52, ymin=10, xmax=72, ymax=30, bl=5),   # l
        Letter(xmin=73, ymin=10, xmax=93, ymax=30, bl=5),   # l
        Letter(xmin=94, ymin=10, xmax=114, ymax=30, bl=5),  # o
    ]
    lines.append(line1)

    # Line 2: Text with ONE OUTLIER letter - simulating bad baseline detection
    line2 = [
        Letter(xmin=10, ymin=40, xmax=30, ymax=60, bl=5),   # w
        Letter(xmin=31, ymin=40, xmax=51, ymax=60, bl=5),   # o
        Letter(xmin=52, ymin=40, xmax=72, ymax=60, bl=5),   # r
        Letter(xmin=73, ymin=40, xmax=93, ymax=60, bl=5),   # l
        Letter(xmin=94, ymin=40, xmax=114, ymax=60, bl=200), # d - OUTLIER with huge bl=200!
    ]
    lines.append(line2)

    # Line 3: Normal text "test"
    line3 = [
        Letter(xmin=10, ymin=70, xmax=30, ymax=90, bl=5),   # t
        Letter(xmin=31, ymin=70, xmax=51, ymax=90, bl=5),   # e
        Letter(xmin=52, ymin=70, xmax=72, ymax=90, bl=5),   # s
        Letter(xmin=73, ymin=70, xmax=93, ymax=90, bl=5),   # t
    ]
    lines.append(line3)

    # Create a simple original image
    original_image = np.ones((100, 150, 3), dtype=np.uint8) * 255
    cv2.putText(original_image, "hello", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(original_image, "world", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(original_image, "test", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return lines, original_image

def main():
    print("Testing line spacing with outlier letter (huge baseline)...")
    print("=" * 70)
    print("Line 2 has a letter with bl=200 (normal is bl=5)")
    print("Without outlier filtering, this would cause HUGE line spacing")
    print("=" * 70)
    print()

    lines, original_image = create_test_with_outlier()

    zoom_factor = 1.5
    new_page_width = 400

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )

    # Save and display
    cv2.imwrite("/tests/test_outlier_spacing.png", page_reflowed)
    print()
    print("Result saved to: test_outlier_spacing.png")
    print("Check if line spacing is reasonable (should be ~45-50 pixels, not 200+)")

if __name__ == "__main__":
    main()
