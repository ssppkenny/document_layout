"""
extractor.py — page extraction pipeline for ScanReader.

Runs YOLO layout analysis + EasyOCR on a page image and returns a
PageData-compatible dict consumed by the Android/web clients.
"""

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Map doclayout_yolo class names (after find_grouped_bounding_boxes) → BlockType
_YOLO_TO_BLOCK_TYPE = {
    'plain text':                  'TEXT',
    'title':                       'HEADING',
    'figure':                      'PICTURE',
    'figure_and_caption':          'PICTURE',
    'figure_caption':              'TEXT',     # unpaired caption
    'isolate_formula':             'FORMULA',
    'isolate_formula_and_caption': 'FORMULA',
    'table':                       'TABLE',
    'table_and_caption':           'TABLE',
    'table_caption':               'TEXT',     # unpaired caption
    'table_footnote':              'TEXT',     # unpaired footnote
    'abandon':                     'FOOTER',
    'reference':                   'REFERENCE',
}

# EasyOCR Reader cache keyed by sorted language tuple
_READER_CACHE: dict = {}


def _get_reader(languages: list):
    import easyocr
    key = tuple(sorted(languages))
    if key not in _READER_CACHE:
        logger.info(f"Loading EasyOCR reader for languages: {list(key)}")
        _READER_CACHE[key] = easyocr.Reader(list(key), gpu=False)
        logger.info("EasyOCR reader loaded.")
    return _READER_CACHE[key]


def warmup_readers(languages=None):
    """Pre-load EasyOCR models. Call from a background thread at startup."""
    _get_reader(languages or ['ru', 'en'])


def extract_page_data(
    image_source,
    page_num: int,
    book_id: str,
    languages=None,
    blocks_dir: Path = None,
) -> dict:
    """
    Run layout analysis + OCR on a page image.

    Args:
        image_source: raw JPEG/PNG bytes, or a path-like object
        page_num:     0-based page number
        book_id:      book identifier
        languages:    EasyOCR language codes (default: ['ru', 'en'])
        blocks_dir:   if provided, PICTURE crops are saved here as {block_id}.jpg

    Returns:
        PageData dict compatible with the Android/web client models.
    """
    from .layout import layout
    from .skew_detection import detect_and_correct_skew

    langs = languages or ['ru', 'en']

    # ── 1. Decode image ───────────────────────────────────────────────────────
    if isinstance(image_source, (bytes, bytearray)):
        arr = np.frombuffer(image_source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(image_source))
    if img is None:
        raise ValueError("Failed to decode image")

    img_h, img_w = img.shape[:2]

    # ── 2. Skew correction ────────────────────────────────────────────────────
    try:
        corrected, angle = detect_and_correct_skew(img)
        if abs(angle) > 0.1:
            logger.debug(f"Skew corrected by {angle:.2f}°")
            img = corrected
    except Exception as e:
        logger.debug(f"Skew correction skipped: {e}")

    # ── 3. Write temp file for YOLO (needs a file path) ───────────────────────
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cv2.imwrite(str(tmp_path), img)
        # ── 4. Layout analysis ────────────────────────────────────────────────
        layout_boxes = layout(tmp_path)  # [(shapely_geometry, type_str), ...]
    finally:
        tmp_path.unlink(missing_ok=True)

    # Sort top-to-bottom then left-to-right
    layout_boxes.sort(key=lambda item: (item[0].bounds[1], item[0].bounds[0]))

    # ── 5. OCR ────────────────────────────────────────────────────────────────
    reader = _get_reader(langs)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ocr_results = reader.readtext(img_rgb, paragraph=False)
    # Each entry: ([[tl,tr,br,bl], text, confidence])

    ocr_words = []
    for pts, text, _conf in ocr_results:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        ocr_words.append({
            'cx': (x1 + x2) / 2,
            'cy': (y1 + y2) / 2,
            'bbox': {'x': x1, 'y': y1, 'width': max(1, x2 - x1), 'height': max(1, y2 - y1)},
            'text': text,
        })

    # ── 6. Assign words to blocks by centre-point containment ─────────────────
    assigned: set = set()
    blocks = []

    for block_idx, (geom, yolo_type) in enumerate(layout_boxes):
        block_type = _YOLO_TO_BLOCK_TYPE.get(yolo_type, 'TEXT')
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        block_id = f"b{block_idx}"

        block_words = []
        for i, word in enumerate(ocr_words):
            if i in assigned:
                continue
            if geom.contains(Point(word['cx'], word['cy'])):
                block_words.append(word)
                assigned.add(i)

        image_url = None
        if block_type == 'PICTURE' and blocks_dir is not None:
            image_url = _save_block_crop(img, bounds, blocks_dir, book_id, page_num, block_id)

        blocks.append({
            'id': block_id,
            'type': block_type,
            'bbox': {
                'x': int(bounds[0]),
                'y': int(bounds[1]),
                'width': int(bounds[2] - bounds[0]),
                'height': int(bounds[3] - bounds[1]),
            },
            'text': ' '.join(w['text'] for w in block_words) or None,
            'words': [
                {'bbox': w['bbox'], 'text': w['text'], 'baseline': None}
                for w in block_words
            ],
            'imageUrl': image_url,
        })

    # ── 7. Leftover OCR words → extra TEXT block ──────────────────────────────
    leftover = [ocr_words[i] for i in range(len(ocr_words)) if i not in assigned]
    if leftover:
        xs  = [w['bbox']['x']                      for w in leftover]
        ys  = [w['bbox']['y']                      for w in leftover]
        x2s = [w['bbox']['x'] + w['bbox']['width'] for w in leftover]
        y2s = [w['bbox']['y'] + w['bbox']['height'] for w in leftover]
        blocks.append({
            'id': f'b{len(blocks)}_extra',
            'type': 'TEXT',
            'bbox': {
                'x': int(min(xs)),
                'y': int(min(ys)),
                'width': int(max(x2s) - min(xs)),
                'height': int(max(y2s) - min(ys)),
            },
            'text': ' '.join(w['text'] for w in leftover),
            'words': [
                {'bbox': w['bbox'], 'text': w['text'], 'baseline': None}
                for w in leftover
            ],
            'imageUrl': None,
        })

    return {
        'bookId': book_id,
        'pageNum': page_num,
        'width': img_w,
        'height': img_h,
        'blocks': blocks,
    }


def _save_block_crop(img, bounds, blocks_dir: Path, book_id: str, page_num: int, block_id: str):
    """Save a cropped picture block and return its serving URL."""
    try:
        h, w = img.shape[:2]
        x1 = max(0, int(bounds[0]))
        y1 = max(0, int(bounds[1]))
        x2 = min(w, int(bounds[2]))
        y2 = min(h, int(bounds[3]))
        crop = img[y1:y2, x1:x2]
        out_path = blocks_dir / f"{block_id}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop)
        return f"/images/{book_id}/{page_num}/{block_id}"
    except Exception as e:
        logger.warning(f"Could not save block crop for {block_id}: {e}")
        return None
