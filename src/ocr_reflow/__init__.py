"""OCR Reflow - Document text reflow tool.

This package provides tools for extracting text from scanned document images
and reflowing it onto a new page with improved formatting, proper line wrapping,
and consistent spacing.
"""

import logging

from .main import process_document, process_document_with_layout, Letter, find_rects, margins, visualize_detected_lines
from .reflow import create_page_with_word_wrapping
from .divide_conquer_4d import Point4D, divide_conquer_4d
from .layout import layout
from .skew_detection import detect_and_correct_skew, detect_skew, rotate_image, detect_skew_in_text_regions

__version__ = "0.2.0"
__all__ = [
    "process_document",
    "process_document_with_layout",
    "Letter",
    "find_rects",
    "margins",
    "visualize_detected_lines",
    "create_page_with_word_wrapping",
    "detect_and_correct_skew",
    "detect_skew",
    "rotate_image",
    "detect_skew_in_text_regions",
]

# Logging is configured per-entry-point via log_setup.setup_logging().
# The package-level logger defaults to NOTSET so messages propagate to
# the root logger (which is configured by the entry point).
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.getLogger(__name__).setLevel(logging.NOTSET)

