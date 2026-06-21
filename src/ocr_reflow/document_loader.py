"""Document loader supporting images, PDF, and DjVu files.

All formats are rendered at a minimum of 300 DPI and returned as
NumPy BGR arrays compatible with OpenCV.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Minimum render resolution
MIN_DPI = 300

# Supported plain image extensions (handled by OpenCV)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Document cache — keep PDF/DjVu handles open across load_page calls to avoid
# re-opening and re-decoding the index on every page.
_doc_cache: dict[str, object] = {}


def _get_cached_doc(path: Path, opener):
    """Open or return cached document handle for ``path``."""
    key = str(path.resolve())
    doc = _doc_cache.get(key)
    if doc is not None:
        return doc
    doc = opener()
    _doc_cache[key] = doc
    return doc


def load_page(filepath: str, page_number: int = 0, min_dpi: int = 300) -> np.ndarray:
    """Load a single page from an image, PDF, or DjVu file.

    Args:
        filepath: Path to an image file, PDF (.pdf), or DjVu (.djvu).
        page_number: 0-based page index. Ignored for plain image files.
        min_dpi: Minimum render DPI (DJVU only — PDF renders at default DPI).

    Returns:
        NumPy array in BGR format (height x width x 3), dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the page number is out of range or the format is unsupported.
        RuntimeError: If rendering fails.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf_page(path, page_number)
    elif suffix == ".djvu":
        return _load_djvu_page(path, page_number, min_dpi)
    elif suffix in IMAGE_EXTENSIONS:
        return _load_image(path)
    else:
        # Try OpenCV as a fallback for unknown extensions
        logger.warning(
            "Unknown file extension '%s', attempting to load as image.", suffix
        )
        return _load_image(path)


# ---------------------------------------------------------------------------
# PDF rendering via PyMuPDF (fitz)
# ---------------------------------------------------------------------------

def _load_pdf_page(path: Path, page_number: int) -> np.ndarray:
    """Render a PDF page to a BGR numpy array at >= 300 DPI."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "pymupdf is required to read PDF files. "
            "Install it with: pip install pymupdf"
        ) from exc

    doc = _get_cached_doc(path, lambda: fitz.open(str(path)))
    n_pages = len(doc)
    if page_number < 0 or page_number >= n_pages:
        raise ValueError(
            f"Page number {page_number} is out of range for '{path}' "
            f"({n_pages} page(s))."
        )

    page = doc[page_number]

    # PyMuPDF default resolution is 72 DPI; scale to reach MIN_DPI.
    scale = MIN_DPI / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)

    # pix.samples is a bytes object: rows of RGB triplets
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    bgr = rgb[:, :, ::-1].copy()
    logger.debug(
        "PDF page %d rendered at %.0f DPI: %dx%d px",
        page_number,
        MIN_DPI,
        pix.width,
        pix.height,
    )
    return bgr


# ---------------------------------------------------------------------------
# DjVu rendering via python-djvulibre
# ---------------------------------------------------------------------------

class _DjVuContext(object):
    """Minimal djvu.decode.Context subclass that logs errors."""

    _instance = None

    @classmethod
    def get(cls):
        """Get the singleton DjVu context, creating it if needed."""
        if cls._instance is None:
            import djvu.decode  # noqa: PLC0415
            class _Ctx(djvu.decode.Context):
                def handle_message(self, message):
                    """Suppress DjVu context error messages to stderr."""
                    import djvu.decode as _d
                    if isinstance(message, _d.ErrorMessage):
                        logger.error("DjVu: %s", message)
            cls._instance = _Ctx()
        return cls._instance


def _load_djvu_page(path: Path, page_number: int, min_dpi: int = MIN_DPI) -> np.ndarray:
    """Render a DjVu page to a BGR numpy array at >= min_dpi."""
    try:
        import djvu.decode
    except ImportError as exc:
        raise ImportError(
            "python-djvulibre is required to read DjVu files. "
            "Install it with: pip install python-djvulibre"
        ) from exc

    ctx = _DjVuContext.get()
    doc = _get_cached_doc(path, lambda: ctx.new_document(djvu.decode.FileURI(str(path))))
    doc.decoding_job.wait()

    n_pages = len(doc.pages)
    if page_number < 0 or page_number >= n_pages:
        raise ValueError(
            f"Page number {page_number} is out of range for '{path}' "
            f"({n_pages} page(s))."
        )

    page = doc.pages[page_number]
    page_job = page.decode(wait=True)

    # DjVu pages store their native DPI; scale to reach min_dpi.
    native_dpi = page_job.dpi  # typically 300, 400, or 600
    if native_dpi and native_dpi > 0:
        scale = max(1.0, min_dpi / native_dpi)
    else:
        scale = 1.0

    native_width, native_height = page_job.size
    render_width = max(1, round(native_width * scale))
    render_height = max(1, round(native_height * scale))

    pixel_format = djvu.decode.PixelFormatRgb()
    pixel_format.rows_top_to_bottom = 1
    pixel_format.y_top_to_bottom = 1

    page_rect = (0, 0, render_width, render_height)
    render_rect = (0, 0, render_width, render_height)
    row_size = render_width * 3
    buf = bytearray(render_height * row_size)

    page_job.render(
        djvu.decode.RENDER_COLOR,
        page_rect,
        render_rect,
        pixel_format,
        row_alignment=row_size,
        buffer=buf,
    )

    rgb = np.frombuffer(buf, dtype=np.uint8).reshape(render_height, render_width, 3)
    bgr = rgb[:, :, ::-1].copy()
    effective_dpi = native_dpi * scale if native_dpi else min_dpi
    logger.debug(
        "DjVu page %d rendered at %.0f DPI (native %s DPI, scale %.2f): %dx%d px",
        page_number,
        effective_dpi,
        native_dpi,
        scale,
        render_width,
        render_height,
    )
    return bgr


# ---------------------------------------------------------------------------
# Plain image loading via OpenCV
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    """Load a plain image file with OpenCV."""
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    return img
