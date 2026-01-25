#!/usr/bin/env python3
"""
Complete demonstration of the ocr-reflow package.

This script shows all the different ways to use the package:
1. Basic usage
2. Batch processing
3. Error handling
4. Custom output paths
"""

import sys
from pathlib import Path
from ocr_reflow import process_document
import cv2


def demo_basic_usage(image_path: str):
    """Demonstrate basic usage."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Usage")
    print("=" * 60)

    if not Path(image_path).exists():
        print(f"⚠ Skipping: {image_path} not found")
        return

    print(f"Processing: {image_path}")
    result = process_document(image_path)

    output_path = "demo_output_1.png"
    cv2.imwrite(output_path, result)

    print(f"✓ Success! Saved to: {output_path}")
    print(f"  Input size: {cv2.imread(image_path).shape}")
    print(f"  Output size: {result.shape}")


def demo_batch_processing(image_dir: str):
    """Demonstrate batch processing multiple images."""
    print("\n" + "=" * 60)
    print("Demo 2: Batch Processing")
    print("=" * 60)

    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"⚠ Skipping: {image_dir} not found")
        return

    # Find all PNG images
    images = list(image_dir.glob("*.png"))[:3]  # Limit to 3 for demo

    if not images:
        print(f"⚠ No images found in {image_dir}")
        return

    print(f"Found {len(images)} images to process")

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        try:
            result = process_document(str(img_path))
            output_path = f"demo_batch_{img_path.stem}_reflowed.png"
            cv2.imwrite(output_path, result)
            print(f"  ✓ Saved to: {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n" + "=" * 60)
    print("Demo 3: Error Handling")
    print("=" * 60)

    test_cases = [
        "nonexistent_file.png",
        "README.md",  # Not an image
    ]

    for test_file in test_cases:
        print(f"\nTrying to process: {test_file}")
        try:
            result = process_document(test_file)
            print(f"  ✓ Success (unexpected!)")
        except FileNotFoundError:
            print(f"  ✓ Correctly caught FileNotFoundError")
        except Exception as e:
            print(f"  ✓ Caught error: {type(e).__name__}: {e}")


def demo_custom_output():
    """Demonstrate custom output paths."""
    print("\n" + "=" * 60)
    print("Demo 4: Custom Output Paths")
    print("=" * 60)

    image_path = "images/dvurog_p007.png"

    if not Path(image_path).exists():
        print(f"⚠ Skipping: {image_path} not found")
        return

    output_configs = [
        ("demo_output_custom1.png", "Custom filename"),
        ("output/demo.png", "Custom directory (will fail if dir doesn't exist)"),
    ]

    result = process_document(image_path)

    for output_path, description in output_configs:
        print(f"\n{description}: {output_path}")
        try:
            # Create directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, result)
            print(f"  ✓ Saved successfully")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def demo_info():
    """Display package information."""
    print("\n" + "=" * 60)
    print("Package Information")
    print("=" * 60)

    from ocr_reflow import __version__
    print(f"Version: {__version__}")

    # List available functions
    import ocr_reflow
    public_attrs = [attr for attr in dir(ocr_reflow) if not attr.startswith('_')]
    print(f"\nAvailable exports:")
    for attr in public_attrs:
        print(f"  - {attr}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("OCR Reflow Package - Complete Demonstration")
    print("=" * 60)

    # Display package info
    demo_info()

    # Run demos
    demo_basic_usage("images/dvurog_p007.png")
    demo_batch_processing("images")
    demo_error_handling()
    demo_custom_output()

    # Summary
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check the generated demo_*.png files")
    print("  2. Try the Jupyter notebook: notebooks/example_usage.ipynb")
    print("  3. Read the guides: docs/INSTALL.md, docs/JUPYTER_GUIDE.md")
    print("  4. Use the CLI: ocr-reflow <image.png>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
