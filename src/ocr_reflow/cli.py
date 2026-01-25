"""Command-line interface for ocr-reflow."""

import sys
from pathlib import Path
from .main import process_document
import cv2


def main():
    """Main entry point for the CLI."""
    # Handle help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("Usage: ocr-reflow <image_file> [output_file]")
        print("\nProcess a scanned document image and reflow the text.")
        print("\nArguments:")
        print("  image_file    Path to the input document image (PNG, JPG, etc.)")
        print("  output_file   Optional path for the output image")
        print("                (default: <input>_reflowed.<ext>)")
        print("\nExamples:")
        print("  ocr-reflow document.png")
        print("  ocr-reflow document.png output.png")
        print("  ocr-reflow scans/page1.jpg results/page1_reflowed.jpg")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Error: No input file specified")
        print("\nUsage: ocr-reflow <image_file> [output_file]")
        print("Use 'ocr-reflow --help' for more information")
        sys.exit(1)

    filename = sys.argv[1]

    # Check if input file exists
    if not Path(filename).exists():
        print(f"Error: File not found: {filename}")
        sys.exit(1)

    # Determine output filename
    if len(sys.argv) >= 3:
        output_filename = sys.argv[2]
    else:
        # Default: add '_reflowed' suffix
        input_path = Path(filename)
        output_filename = f"{input_path.stem}_reflowed{input_path.suffix}"

    print(f"Processing: {filename}")
    try:
        page_with_letters = process_document(filename)
        cv2.imwrite(output_filename, page_with_letters)
        print(f"✓ Success! Output saved to: {output_filename}")
    except Exception as e:
        print(f"✗ Error processing document: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
