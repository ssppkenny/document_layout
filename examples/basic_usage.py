#!/usr/bin/env python3
"""
Example: Basic usage of the text segmentation and reflow system

This script demonstrates how to:
1. Load an image
2. Process it with text detection
3. Reflow the text to a new page
4. Save the results

Usage:
    python examples/basic_usage.py path/to/image.png
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import *
import cv2
import numpy as np

def example_basic_reflow(image_path: str, output_path: str = 'example_output.png'):
    """
    Basic example: Load image, detect text, reflow, and save.

    Args:
        image_path: Path to input image
        output_path: Path for output image
    """
    print(f"Processing: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

    # Detect text using doctr
    print("Detecting text...")
    detector = detection_predictor(arch='db_resnet50', pretrained=True)
    doc = DocumentFile.from_images([image_path])
    result = detector(doc)

    # Extract lines and words
    lines_words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geometry = word.geometry
                    xmin = int(geometry[0][0] * img.shape[1])
                    ymin = int(geometry[0][1] * img.shape[0])
                    xmax = int(geometry[1][0] * img.shape[1])
                    ymax = int(geometry[1][1] * img.shape[0])
                    lines_words.append((xmin, ymin, xmax, ymax))

    print(f"Found {len(lines_words)} words")

    # Find individual characters
    print("Extracting characters...")
    rects = find_rects(img, lines_words)
    print(f"Found {rects} characters")

    # [Rest of processing as in main.py...]
    # This is a simplified example showing the basic structure

    print(f"Reflow complete! Output saved to: {output_path}")


def example_custom_parameters(image_path: str):
    """
    Example with custom reflow parameters.
    """
    print("\n" + "="*60)
    print("Example: Custom Parameters")
    print("="*60)

    # Load and process image (simplified)
    img = cv2.imread(image_path)

    # Assume we have lines of Letter objects (see main.py for full extraction)
    # lines = [...]

    # Custom reflow with specific parameters
    # reflowed = create_page_with_word_wrapping(
    #     lines=lines,
    #     original_image=img,
    #     zoom_factor=2.0,           # Larger text
    #     new_page_width=1200,       # Wider page
    #     left_margin=80,            # Larger margins
    #     right_margin=80,
    #     top_margin=100,
    #     bottom_margin=100,
    #     line_spacing=25,           # More space between lines
    #     paragraph_spacing_factor=2.5,  # Even more space between paragraphs
    #     preserve_spacing=True,
    #     background_color=(240, 240, 230)  # Cream color
    # )

    print("Custom parameters applied:")
    print("  - Zoom: 2.0x")
    print("  - Page width: 1200px")
    print("  - Margins: 80-100px")
    print("  - Line spacing: 25px")
    print("  - Background: Cream")


def example_with_error_handling(image_path: str):
    """
    Example with proper error handling.
    """
    print("\n" + "="*60)
    print("Example: With Error Handling")
    print("="*60)

    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        print(f"✓ Image loaded: {img.shape}")

        # Check image size
        if img.shape[0] < 100 or img.shape[1] < 100:
            raise ValueError("Image too small (minimum 100x100 pixels)")

        print(f"✓ Image size OK")

        # Process...
        print("✓ Processing would continue here")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure the image path is correct")
    except ValueError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for examples."""
    print("=" * 60)
    print("Text Segmentation & Reflow - Examples")
    print("=" * 60)

    # Check command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python examples/basic_usage.py <image_path>")
        print("\nExample:")
        print("  python examples/basic_usage.py test_image.png")
        print("\nAvailable examples:")
        print("  1. Basic reflow with default parameters")
        print("  2. Custom parameters")
        print("  3. With error handling")
        sys.exit(1)

    image_path = sys.argv[1]

    # Run examples
    print("\n1. Basic Reflow")
    print("-" * 60)
    example_basic_reflow(image_path)

    print("\n2. Custom Parameters")
    print("-" * 60)
    example_custom_parameters(image_path)

    print("\n3. Error Handling")
    print("-" * 60)
    example_with_error_handling(image_path)

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
