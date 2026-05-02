"""Command-line interface for ocr-reflow."""

import sys
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="ocr-reflow",
        description=(
            "OCR text reflow tool. Extracts and reflows text from scanned "
            "document images, PDF files, and DjVu files."
        ),
    )
    parser.add_argument(
        "input_file",
        help=(
            "Input file to process. Supported formats: "
            "image (PNG, JPG, TIFF, BMP, WEBP), PDF (.pdf), DjVu (.djvu)."
        ),
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        help=(
            "Output image file path. Defaults to <input_stem>_reflowed<ext> "
            "for images, or <input_stem>_p<page>_reflowed.png for PDF/DjVu."
        ),
    )
    parser.add_argument(
        "-p", "--page",
        type=int,
        default=0,
        metavar="N",
        help="0-based page number to process (for PDF and DjVu files). Default: 0.",
    )

    args = parser.parse_args()

    filename = args.input_file
    page_number = args.page

    if not Path(filename).exists():
        logger.error("Error: File not found: %s", filename)
        sys.exit(1)

    # Determine output filename
    if args.output_file:
        output_filename = args.output_file
    else:
        input_path = Path(filename)
        suffix = input_path.suffix.lower()
        if suffix in (".pdf", ".djvu"):
            output_filename = f"{input_path.stem}_p{page_number}_reflowed.png"
        else:
            output_filename = f"{input_path.stem}_reflowed{input_path.suffix}"

    logger.info("Processing: %s (page %d)", filename, page_number)

    try:
        from ocr_reflow.document_loader import load_page
        from ocr_reflow.main import process_document
        import cv2

        img = load_page(filename, page_number)
        page_with_letters = process_document(img)
        cv2.imwrite(output_filename, page_with_letters)
        logger.info("Success! Output saved to: %s", output_filename)
    except Exception as e:
        logger.error("Error processing document: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
