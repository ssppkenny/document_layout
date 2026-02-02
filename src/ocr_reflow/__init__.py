"""OCR Reflow - Document text reflow tool.

This package provides tools for extracting text from scanned document images
and reflowing it onto a new page with improved formatting, proper line wrapping,
and consistent spacing.
"""

import logging

from docs.main import process_document, Letter, find_rects, margins, visualize_detected_lines
from .reflow import create_page_with_word_wrapping
from .divide_conquer_4d import Point4D, divide_conquer_4d
from .layout import layout

__version__ = "0.1.0"
__all__ = [
    "process_document",
    "Letter",
    "find_rects",
    "margins",
    "visualize_detected_lines",
    "create_page_with_word_wrapping",
]

# Configure logging - disabled by default (level set to WARNING)
# Users can enable debug logging with: logging.getLogger('ocr_reflow').setLevel(logging.DEBUG)
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.getLogger(__name__).setLevel(logging.WARNING)

