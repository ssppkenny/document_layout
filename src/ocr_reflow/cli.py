"""Command-line interface for ocr-reflow."""

import sys
import logging
from pathlib import Path
from docs.main import process_document
import cv2

# Set up logger for this module
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    # Handle help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        help_text = """Usage: ocr-reflow <image_file> [output_file]

Process a scanned document image and reflow the text.

Arguments:
  image_file    Path to the input document image (PNG, JPG, etc.)
  output_file   Optional path for the output image
                (default: <input>_reflowed.<ext>)

Examples:
  ocr-reflow document.png
  ocr-reflow document.png output.png
  ocr-reflow scans/page1.jpg results/page1_reflowed.jpg"""
        logger.info(help_text)
        sys.exit(0)

    if len(sys.argv) < 2:
        error_text = """Error: No input file specified

Usage: ocr-reflow <image_file> [output_file]
Use 'ocr-reflow --help' for more information"""
        logger.error(error_text)
        sys.exit(1)

    filename = sys.argv[1]

    # Check if input file exists
    if not Path(filename).exists():
        logger.error(f"Error: File not found: {filename}")
        sys.exit(1)

    # Determine output filename
    if len(sys.argv) >= 3:
        output_filename = sys.argv[2]
    else:
        # Default: add '_reflowed' suffix
        input_path = Path(filename)
        output_filename = f"{input_path.stem}_reflowed{input_path.suffix}"

    logger.info(f"Processing: {filename}")
    try:
        page_with_letters = process_document(filename)
        cv2.imwrite(output_filename, page_with_letters)
        logger.info(f"✓ Success! Output saved to: {output_filename}")
    except Exception as e:
        logger.error(f"✗ Error processing document: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
