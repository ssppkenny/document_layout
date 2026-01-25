"""OCR Reflow - Document text reflow tool.

This package provides tools for extracting text from scanned document images
and reflowing it onto a new page with improved formatting, proper line wrapping,
and consistent spacing.
"""

from .main import process_document, Letter, find_rects, margins
from .reflow import create_page_with_word_wrapping

__version__ = "0.1.0"
__all__ = [
    "process_document",
    "Letter",
    "find_rects",
    "margins",
    "create_page_with_word_wrapping",
]
