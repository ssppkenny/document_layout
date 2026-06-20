"""Language auto-detection for document OCR.

Downloads fastText lid.176.bin from the official Facebook URL on first use,
caches it in the project's ``models/fasttext/`` directory.

Usage::

    from ocr_reflow.language_detection import detect
    lang = detect("book.djvu", 300)   # → "ru"
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

_FASTTEXT_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
_MODEL_FILENAME = "lid.176.bin"


# ---------------------------------------------------------------------------
# Model download / caching
# ---------------------------------------------------------------------------

def _get_model_path() -> Path:
    """Get the fastText language detection model path from model_manager."""
    from ocr_reflow.model_manager import get_models_dir

    model_dir = get_models_dir() / "fasttext"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / _MODEL_FILENAME


def _ensure_model() -> Path:
    """Download ``lid.176.bin`` on first use, return cached path."""
    model_path = _get_model_path()
    if model_path.exists():
        return model_path

    logger.info("Downloading fastText language identification model (176 MB)...")
    logger.info("  from: %s", _FASTTEXT_URL)
    logger.info("  to:   %s", model_path)

    urllib.request.urlretrieve(_FASTTEXT_URL, model_path)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info("  downloaded %d MB", size_mb)
    return model_path


# ---------------------------------------------------------------------------
# Sampling helper — OCR a full page with the LightOnOCR VLM
# ---------------------------------------------------------------------------

def _ocr_full_page(source_path: str, page_num: int) -> str:
    """OCR an entire page image with the LightOnOCR VLM.

    Relies on the module-level singleton in ``ocr_export_layout`` (first call
    lazy-loads the 2 GB model; subsequent calls reuse it).
    """
    try:
        from document_loader import load_page  # type: ignore[no-redef]
        from ocr_export_layout import (  # type: ignore[no-redef]
            _bgr_to_pil,
            _get_lightonocr,
            _resize_for_ocr,
        )
    except ImportError:
        from ocr_reflow.document_loader import load_page
        from ocr_reflow.ocr_export_layout import (
            _bgr_to_pil,
            _get_lightonocr,
            _resize_for_ocr,
        )

    import torch

    img = load_page(source_path, page_num - 1)  # 0-based
    h, w = img.shape[:2]
    img = cv2.resize(img, (1024, int(h * 1024 / w)))
    pil = _resize_for_ocr(_bgr_to_pil(img))

    processor, model = _get_lightonocr()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    conv = [{"role": "user", "content": [{"type": "image", "image": pil}]}]
    text_prompt = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text_prompt], images=[pil], padding=True, return_tensors="pt"
    )
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            top_k=0,
        )
    input_len = inputs["input_ids"].shape[1]
    gen = ids[0, input_len:]
    return processor.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(source_path: str, total_pages: int) -> str:
    """Auto-detect the document language by OCR-sampling 3 mid-book pages.

    Positions: ~25 %, ~50 %, ~75 % into the document.

    Returns an ISO 639-1 language code (e.g. ``"ru"``, ``"en"``,
    ``"de"``, …).

    Falls back to ``"en"`` when no text can be read from the sample pages.
    """
    if total_pages < 3:
        # Not enough pages to sample — default to English
        return "en"

    positions = sorted(
        {max(1, min(int(total_pages * p), total_pages)) for p in (0.25, 0.5, 0.75)}
    )

    logger.info(
        "Auto-detecting language: OCR-sampling pages %s...",
        ", ".join(str(p) for p in positions),
    )

    texts: list[str] = []
    for pg in positions:
        try:
            text = _ocr_full_page(source_path, pg)
            if text:
                texts.append(text)
        except Exception as e:
            logger.warning("language_detection page %d: %s", pg, e)

    combined = "\n".join(texts)
    if not combined.strip():
        logger.warning("Warning: no text in sample pages; defaulting to 'en'")
        return "en"

    import numpy as np

    # fasttext 0.9.2 uses np.array(probs, copy=False) which raises on
    # NumPy >= 2 when a copy is needed.  Monkey-patch to use np.asarray.
    if not hasattr(np, "_fasttext_patched"):
        _orig_array = np.array

        def _patched_array(obj, copy=True, **kwargs):
            """Patch for fastText: np.array with copy=False fallback."""
            if copy is False:
                return np.asarray(obj, **kwargs)
            return _orig_array(obj, copy=copy, **kwargs)

        np.array = _patched_array
        np._fasttext_patched = True

    import fasttext

    model_path = _ensure_model()
    model = fasttext.load_model(str(model_path))
    predictions = model.predict(combined.replace("\n", " ").strip(), k=1)
    lang = predictions[0][0].replace("__label__", "")

    logger.info("Detected language: %s", lang)
    return lang
