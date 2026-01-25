"""
Simple usage example of the ocr-reflow package.

This script demonstrates how to process a scanned document
and reflow the text to a new page.
"""

from ocr_reflow import process_document
import cv2
from pathlib import Path
import sys


def main():
    # Check if an image path was provided
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        # Default example image
        input_image = "../images/dvurog_p007.png"

    # Check if file exists
    if not Path(input_image).exists():
        print(f"Error: File not found: {input_image}")
        print("\nUsage: python simple_example.py <path_to_image.png>")
        return 1

    print(f"Processing: {input_image}")

    # Process the document
    # This will detect text, extract characters, and reflow them to a new page
    result = process_document(input_image)

    # Generate output filename
    input_path = Path(input_image)
    output_image = f"{input_path.stem}_reflowed{input_path.suffix}"

    # Save the result
    cv2.imwrite(output_image, result)

    print(f"✓ Success! Reflowed document saved to: {output_image}")
    print(f"  Output image size: {result.shape[1]}x{result.shape[0]} pixels")

    return 0


if __name__ == "__main__":
    sys.exit(main())
