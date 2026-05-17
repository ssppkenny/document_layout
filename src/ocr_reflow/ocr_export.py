"""Compatibility shim — all logic has moved to ocr_export_layout.py.

Importing from this module continues to work unchanged.
"""

try:
    from ocr_reflow.ocr_export_layout import (
        _get_device,
        _get_lightonocr,
        ocr_page_to_html,
        ocr_page_to_html_simple,
        ocr_page_block_generator,
    )
except ImportError:
    from ocr_export_layout import (
        _get_device,
        _get_lightonocr,
        ocr_page_to_html,
        ocr_page_to_html_simple,
        ocr_page_block_generator,
    )

__all__ = [
    "_get_device",
    "_get_lightonocr",
    "ocr_page_to_html",
    "ocr_page_to_html_simple",
    "ocr_page_block_generator",
]
