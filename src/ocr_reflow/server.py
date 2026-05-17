"""
server.py — FastAPI endpoint for page layout analysis and word detection.

POST /page
  Query params:
    page_width   (int,   default 2000)
    zoom_factor  (float, default 2.5)
    lang         (str,   optional)
    bin          (bool,  default false)
    toc_algorithm (str,  default "layoutlm")
  Body: multipart/form-data { image: <JPG/PNG file> }

Returns JSON with all information needed by a client to reflow the page:
  - skew_angle (client must rotate its local image by this angle before cropping)
  - layout blocks with block_type, bbox, gap_before_px
  - per-block lines with per-word bounding boxes and baseline info (bl, above)
  - page-level metadata (background_color, margins, image dimensions)

All coordinates are in the skew-corrected image space.
Word coordinates are absolute in the full image (not relative to block).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import time
import uuid
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load OCR model eagerly at startup so first request is fast."""
    import asyncio
    loop = asyncio.get_event_loop()
    logger.info("Startup: loading LightOnOCR model…")
    try:
        from ocr_reflow.ocr_export import _get_lightonocr
    except ImportError:
        from ocr_export import _get_lightonocr
    await loop.run_in_executor(None, _get_lightonocr)
    logger.info("Startup: LightOnOCR model ready.")
    yield


app = FastAPI(title="OCR Reflow Server", version="1.0.0", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Lazy-loaded globals (models loaded once on first request)
# ---------------------------------------------------------------------------

_doctr_model = None
_layout_model_loaded = False

# ---------------------------------------------------------------------------
# In-memory OCR result cache  (cache_key -> html_str)
# ---------------------------------------------------------------------------

_ocr_cache: dict[str, str] = {}


def _get_doctr_model():
    global _doctr_model
    if _doctr_model is None:
        logger.info("Loading DocTR model…")
        try:
            from ocr_reflow.main import get_doctr_model
            _doctr_model, _ = get_doctr_model()
        except ImportError:
            from main import get_doctr_model
            _doctr_model, _ = get_doctr_model()
        logger.info("DocTR model ready.")
    return _doctr_model


def _analyze_layout(image_path: str):
    """Run DocLayout-YOLO on image_path, return grouped layout boxes."""
    try:
        from ocr_reflow.layout import layout as _layout_fn
    except ImportError:
        from layout import layout as _layout_fn
    return _layout_fn(image_path)


def _analyze_layout_array(img_bgr: np.ndarray):
    """Run DocLayout-YOLO on an in-memory BGR array — no temp file needed."""
    try:
        from ocr_reflow.layout import layout_from_array as _layout_fn
    except ImportError:
        from layout import layout_from_array as _layout_fn
    return _layout_fn(img_bgr)


def _detect_skew(img_bgr: np.ndarray, layout_boxes) -> float:
    """Return skew angle in degrees using HoughLinesP (~45ms)."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return 0.0
        angles = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15:  # keep near-horizontal lines only
                angles.append(angle)
        if not angles:
            return 0.0
        return float(np.median(angles))
    except Exception as e:
        logger.warning(f"Skew detection failed: {e}")
        return 0.0


def _rotate_image(img_bgr: np.ndarray, angle: float) -> np.ndarray:
    try:
        try:
            from ocr_reflow.skew_detection import rotate_image
        except ImportError:
            from skew_detection import rotate_image
        return rotate_image(img_bgr, angle)
    except Exception as e:
        logger.warning(f"Rotation failed: {e}")
        return img_bgr


def _detect_toc(image_path: str, toc_algorithm: str) -> bool:
    """Return True if the page is a Table of Contents."""
    try:
        if toc_algorithm == "layoutlm":
            try:
                from ocr_reflow.layoutlm_toc_detector import detect_toc_with_layoutlm
            except ImportError:
                try:
                    from layoutlm_toc_detector import detect_toc_with_layoutlm
                except ImportError:
                    detect_toc_with_layoutlm = None
            if detect_toc_with_layoutlm is not None:
                is_toc, _, _ = detect_toc_with_layoutlm(image_path, min_toc_entries=4)
                return is_toc
            # fall through to original
        if toc_algorithm in ("original", "layoutlm"):
            try:
                from ocr_reflow.toc_detection import detect_toc_page
            except ImportError:
                from toc_detection import detect_toc_page
            return detect_toc_page(image_path)
        if toc_algorithm == "mtd":
            try:
                from ocr_reflow.toc_detection_mtd import detect_toc_page_mtd
            except ImportError:
                from toc_detection_mtd import detect_toc_page_mtd
            return detect_toc_page_mtd(image_path)
    except Exception as e:
        logger.warning(f"TOC detection failed: {e}")
    return False


def _run_doctr_on_image(model, img_bgr: np.ndarray) -> np.ndarray:
    """
    Run DocTR detection on a single BGR numpy image (block crop).
    Returns array of shape (N, 5): [xmin_norm, ymin_norm, xmax_norm, ymax_norm, conf]
    in normalized [0,1] coordinates.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = model([img_rgb])
    return result[0]["words"]  # shape (N, 5), normalized coords


def _run_doctr_batch(model, crops_bgr: List[np.ndarray]) -> List[np.ndarray]:
    """
    Run DocTR detection on a list of BGR numpy crops in a single batched call.
    Returns a list of (N_i, 5) arrays, one per crop, in normalized [0,1] coords.
    """
    crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
    results = model(crops_rgb)
    return [r["words"] for r in results]


def _words_norm_to_abs(
    words_norm: np.ndarray,
    img_w: int,
    img_h: int,
    padding: int = 5,
) -> np.ndarray:
    """
    Convert normalized DocTR word boxes to absolute integer pixel coords with padding.
    Returns int32 array of shape (N, 5): [xmin, ymin, xmax, ymax, conf].
    """
    if len(words_norm) == 0:
        return np.zeros((0, 5), dtype=np.int32)
    w = words_norm.copy().astype(np.float32)
    w[:, 0] = w[:, 0] * img_w - padding
    w[:, 1] = w[:, 1] * img_h - padding
    w[:, 2] = w[:, 2] * img_w + padding
    w[:, 3] = w[:, 3] * img_h + padding
    w[:, 0] = np.maximum(w[:, 0], 0)
    w[:, 1] = np.maximum(w[:, 1], 0)
    w[:, 2] = np.minimum(w[:, 2], img_w)
    w[:, 3] = np.minimum(w[:, 3], img_h)
    return w.astype(np.int32)


def _group_words_into_lines(
    words_abs: np.ndarray,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Group absolute word boxes into text lines using margins() + merge_close_lines().
    words_abs: int32 array (N, 5) [xmin, ymin, xmax, ymax, conf].
    Returns list of lines; each line is a sorted list of (xmin, ymin, xmax, ymax).
    """
    from shapely.geometry import box as shapely_box, LineString

    try:
        from ocr_reflow.main import margins, merge_close_lines
    except ImportError:
        from main import margins, merge_close_lines

    # margins() expects list of (xmin, ymin, xmax, ymax, conf)
    words_list = [tuple(int(x) for x in row) for row in words_abs]
    if len(words_list) < 2:
        return [[(w[0], w[1], w[2], w[3]) for w in words_list]] if words_list else []

    left_margins, right_margins = margins(words_list)
    left_margins, right_margins = merge_close_lines(
        left_margins, right_margins, words_list, y_threshold=30
    )

    if not left_margins:
        return []

    rectangles = {
        shapely_box(w[0], w[1], w[2], w[3]): (w[0], w[1], w[2], w[3])
        for w in words_list
    }

    lines = []
    for l, r in zip(left_margins, right_margins):
        line_geom = LineString([(l[0], l[1]), (r[0], r[1])])
        line_words = [v for b, v in rectangles.items() if line_geom.intersects(b)]
        if line_words:
            lines.append(sorted(line_words))
    return lines


def _compute_wordlines(
    lines: List[List[Tuple[int, int, int, int]]],
) -> List[List[dict]]:
    """
    Convert raw line word tuples to Word-like dicts with bl and above fields.
    Uses words_to_wordlines from reflow_words.
    Returns list of lines; each line is a list of word dicts.
    """
    try:
        from ocr_reflow.reflow_words import words_to_wordlines, Word
    except ImportError:
        from reflow_words import words_to_wordlines, Word

    word_lines = words_to_wordlines(lines)
    result = []
    for line in word_lines:
        line_dicts = []
        for w in line:
            line_dicts.append({
                "xmin": int(w.xmin),
                "ymin": int(w.ymin),
                "xmax": int(w.xmax),
                "ymax": int(w.ymax),
                "bl": int(w.bl),
                "above": int(w.above),
            })
        result.append(line_dicts)
    return result


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/page")
async def analyze_page(
    image: UploadFile = File(...),
    page_width: int = Query(default=2000, ge=100, le=10000),
    zoom_factor: float = Query(default=2.5, ge=0.1, le=20.0),
    lang: Optional[str] = Query(default=None),
    bin: bool = Query(default=False),
    toc_algorithm: str = Query(default="none", pattern="^(layoutlm|original|mtd|none)$"),
):
    """
    Analyze a page image and return all data needed for client-side reflow.

    The client must:
    1. Rotate its local copy of the image by skew_angle (counter-clockwise)
       before using the returned coordinates for cropping.
    2. Use the returned word bounding boxes (absolute, in skew-corrected space)
       to crop word images from the rotated local image.
    3. Call its local reflow engine with the returned block/line/word structure.
    """
    # --- Read uploaded image ---
    t_start = time.perf_counter()
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    logger.warning(f"[timing] decode: {time.perf_counter()-t_start:.3f}s  size={len(raw)//1024}KB  shape={img_bgr.shape}")

    # --- Optional binarization ---
    if bin:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    img_h, img_w = img_bgr.shape[:2]

    # --- Step 1: Layout analysis directly from array (no tmp file) ---
    t1 = time.perf_counter()
    try:
        initial_layout = _analyze_layout_array(img_bgr)
    except Exception as e:
        logger.warning(f"Initial layout analysis failed: {e}")
        initial_layout = []
    logger.warning(f"[timing] YOLO layout: {time.perf_counter()-t1:.3f}s  boxes={len(initial_layout)}")

    # --- Step 2: Skew detection ---
    t1 = time.perf_counter()
    skew_angle = 0.0
    if initial_layout:
        skew_angle = _detect_skew(img_bgr, initial_layout)
    logger.warning(f"[timing] skew detect: {time.perf_counter()-t1:.3f}s  angle={skew_angle:.3f}°")

    layout_boxes = initial_layout
    layout_boxes_sorted = sorted(
        layout_boxes, key=lambda item: (item[0].bounds[1], item[0].bounds[0])
    )

    # --- Step 3: Background color ---
    bg_color = np.median(img_bgr.reshape(-1, 3), axis=0).astype(int).tolist()

    # --- Step 4: Margins ---
    text_geoms = [g for g, t in layout_boxes_sorted if t in ("plain text", "title")]
    if text_geoms:
        min_xmin = min(g.bounds[0] for g in text_geoms)
        max_xmax = max(g.bounds[2] for g in text_geoms)
        left_margin = max(1, int((min_xmin / img_w) * page_width))
        right_margin = max(1, int(((img_w - max_xmax) / img_w) * page_width))
    else:
        left_margin = int(page_width * 0.025)
        right_margin = int(page_width * 0.025)

    # --- Step 5: TOC detection (still needs a file path) ---
    t1 = time.perf_counter()
    is_toc = False
    if toc_algorithm != "none":
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, img_bgr)
        try:
            is_toc = _detect_toc(tmp_path, toc_algorithm)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    logger.warning(f"[timing] TOC detect: {time.perf_counter()-t1:.3f}s  algo={toc_algorithm}  is_toc={is_toc}")

    # --- Step 6: Median plain-text block width (for is_narrow) ---
    plain_widths = [
        g.bounds[2] - g.bounds[0]
        for g, t in layout_boxes_sorted
        if t in ("plain text", "titled_block_body")
    ]
    median_plain_w = float(np.median(plain_widths)) if plain_widths else img_w

    # --- Step 7: Collect text block metadata and crops for batched DocTR ---
    model = _get_doctr_model()

    # Build per-block metadata first (no inference yet)
    block_meta = []
    prev_y2 = 0
    for idx, (box_geom, box_type) in enumerate(layout_boxes_sorted):
        bounds = box_geom.bounds
        xmin = int(bounds[0])
        ymin = int(bounds[1])
        xmax = int(bounds[2])
        ymax = int(bounds[3])
        block_w = xmax - xmin
        is_narrow = (
            box_type in ("plain text", "titled_block_body")
            and block_w < median_plain_w * 0.65
        )
        gap_before_px = max(0, int((ymin - prev_y2) * zoom_factor))
        prev_y2 = ymax
        needs_doctr = box_type in ("plain text", "title", "titled_block_body") and not is_narrow
        block_meta.append({
            "idx": idx,
            "box_type": box_type,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "is_narrow": is_narrow,
            "gap_before_px": gap_before_px,
            "needs_doctr": needs_doctr,
        })

    # Collect crops for blocks that need DocTR
    doctr_indices = [i for i, m in enumerate(block_meta) if m["needs_doctr"]]
    crops = [
        img_bgr[block_meta[i]["ymin"]:block_meta[i]["ymax"],
                block_meta[i]["xmin"]:block_meta[i]["xmax"]]
        for i in doctr_indices
    ]

    # Single batched DocTR call for all crops
    t1 = time.perf_counter()
    batch_results = _run_doctr_batch(model, crops) if crops else []
    logger.warning(f"[timing] DocTR batch ({len(crops)} blocks): {time.perf_counter()-t1:.3f}s")

    # Map results back to block_meta
    for list_pos, meta_i in enumerate(doctr_indices):
        block_meta[meta_i]["words_norm"] = batch_results[list_pos]

    # --- Step 8: Build block list from metadata + DocTR results ---
    blocks = []
    all_words_abs: List[List[int]] = []

    for m in block_meta:
        xmin, ymin, xmax, ymax = m["xmin"], m["ymin"], m["xmax"], m["ymax"]
        box_type = m["box_type"]
        lines_data: List[dict] = []

        if m["needs_doctr"]:
            words_norm = m.get("words_norm", np.zeros((0, 5)))
            bh = ymax - ymin
            bw = xmax - xmin

            if len(words_norm) > 0:
                padding = 35 if box_type == "title" else 5
                words_local = words_norm.copy().astype(np.float32)
                words_local[:, 0] = np.maximum(words_local[:, 0] * bw - padding, 0)
                words_local[:, 1] = np.maximum(words_local[:, 1] * bh - padding, 0)
                words_local[:, 2] = np.minimum(words_local[:, 2] * bw + padding, bw)
                words_local[:, 3] = np.minimum(words_local[:, 3] * bh + padding, bh)
                words_local = words_local.astype(np.int32)

                if box_type == "title" and len(words_local) > 1:
                    extra = 20
                    words_local = np.array([[
                        max(0, int(words_local[:, 0].min()) - extra),
                        max(0, int(words_local[:, 1].min()) - extra),
                        min(bw, int(words_local[:, 2].max()) + extra),
                        min(bh, int(words_local[:, 3].max()) + extra),
                        100,
                    ]], dtype=np.int32)

                for row in words_local:
                    all_words_abs.append([
                        int(row[0]) + xmin,
                        int(row[1]) + ymin,
                        int(row[2]) + xmin,
                        int(row[3]) + ymin,
                    ])

                raw_lines = _group_words_into_lines(words_local)
                word_lines_local = _compute_wordlines(raw_lines)

                for line in word_lines_local:
                    abs_line = []
                    for w in line:
                        abs_line.append({
                            "xmin": w["xmin"] + xmin,
                            "ymin": w["ymin"] + ymin,
                            "xmax": w["xmax"] + xmin,
                            "ymax": w["ymax"] + ymin,
                            "bl": w["bl"],
                            "above": w["above"],
                        })
                    lines_data.append({"words": abs_line})

        blocks.append({
            "index": m["idx"],
            "block_type": box_type,
            "bbox": [xmin, ymin, xmax, ymax],
            "is_narrow": m["is_narrow"],
            "gap_before_px": m["gap_before_px"],
            "lines": lines_data,
        })

    logger.warning(f"[timing] TOTAL: {time.perf_counter()-t_start:.3f}s")
    return JSONResponse({
        "image_width": img_w,
        "image_height": img_h,
        "background_color": bg_color,
        "skew_angle": float(skew_angle),
        "is_toc": bool(is_toc),
        "zoom_factor": float(zoom_factor),
        "page_width": int(page_width),
        "left_margin": int(left_margin),
        "right_margin": int(right_margin),
        "word_boxes": all_words_abs,
        "blocks": blocks,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ocr_page")
async def ocr_page_endpoint(
    request: Request,
    image: UploadFile = File(...),
    no_pix2tex: bool = Query(default=False),  # kept for backward compat, ignored
    cache_key: str = Query(default=""),
    batched: bool = Query(default=True),
):
    """
    Run layout analysis + LightOnOCR on a page image.
    Returns a self-contained HTML string with inline base64 images and MathJax.
    If cache_key is provided, the result is stored in _ocr_cache for later retrieval.
    """
    import asyncio

    t_start = time.perf_counter()

    t_read = time.perf_counter()
    raw = await image.read()
    t_read_done = time.perf_counter()

    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    t_decode_done = time.perf_counter()

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    logger.warning(
        f"[timing] ocr_page: image read={t_read_done-t_read:.3f}s "
        f"decode={t_decode_done-t_read_done:.3f}s "
        f"size={len(raw)//1024}KB "
        f"shape={img_bgr.shape[1]}x{img_bgr.shape[0]}"
    )

    try:
        if batched:
            from ocr_reflow.ocr_export_layout import ocr_page_to_html
        else:
            from ocr_reflow.ocr_export_layout import ocr_page_to_html_simple as ocr_page_to_html
    except ImportError:
        if batched:
            from ocr_export_layout import ocr_page_to_html
        else:
            from ocr_export_layout import ocr_page_to_html_simple as ocr_page_to_html

    base_url = str(request.base_url).rstrip("/")
    loop = asyncio.get_event_loop()

    t_ocr = time.perf_counter()
    html_str = await loop.run_in_executor(
        None, lambda: ocr_page_to_html(img_bgr, base_url=base_url)
    )
    t_ocr_done = time.perf_counter()
    logger.warning(f"[timing] ocr_page: ocr_page_to_html={t_ocr_done-t_ocr:.3f}s")

    t_cache = time.perf_counter()
    if cache_key:
        _ocr_cache[cache_key] = html_str
    t_end = time.perf_counter()
    logger.warning(
        f"[timing] ocr_page: cache+response={t_end-t_cache:.3f}s  "
        f"html_size={len(html_str)//1024}KB"
    )
    logger.warning(f"[timing] ocr_page TOTAL: {t_end-t_start:.3f}s")

    return Response(content=html_str, media_type="text/html")


@app.get("/ocr_result/{cache_key}")
async def ocr_result(cache_key: str):
    """Serve a previously cached OCR HTML result by key."""
    if cache_key not in _ocr_cache:
        raise HTTPException(status_code=404, detail="OCR result not found. Re-run OCR.")
    return Response(content=_ocr_cache[cache_key], media_type="text/html")


# ---------------------------------------------------------------------------
# Streaming OCR endpoints
# ---------------------------------------------------------------------------

# In-memory job registry: token -> asyncio.Queue
# The generator thread puts (event, payload) dicts; None signals completion.
_stream_jobs: dict[str, asyncio.Queue] = {}


@app.post("/ocr_page_stream")
async def ocr_page_stream_start(
    request: Request,
    image: UploadFile = File(...),
):
    """Accept a page image, start background OCR, return a stream URL immediately.

    Returns JSON: {"stream_url": "/ocr_stream/{token}", "token": "..."}
    The client should navigate its WebView to stream_url.
    """
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    token = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _stream_jobs[token] = queue

    base_url = str(request.base_url).rstrip("/")
    loop = asyncio.get_event_loop()

    def _run_generator():
        try:
            from ocr_reflow.ocr_export_layout import ocr_page_block_generator
        except ImportError:
            from ocr_export_layout import ocr_page_block_generator
        try:
            for event, payload in ocr_page_block_generator(img_bgr, base_url=base_url):
                asyncio.run_coroutine_threadsafe(queue.put((event, payload)), loop)
        except Exception as exc:
            logger.exception("ocr_page_block_generator failed: %s", exc)
            asyncio.run_coroutine_threadsafe(queue.put(("error", {"message": str(exc)})), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    loop.run_in_executor(None, _run_generator)

    stream_url = f"{base_url}/ocr_stream/{token}"
    return JSONResponse({"stream_url": stream_url, "token": token})


@app.get("/ocr_stream/{token}")
async def ocr_stream_page(token: str, request: Request):
    """Serve the host HTML skeleton page for a streaming OCR job."""
    if token not in _stream_jobs:
        raise HTTPException(status_code=404, detail="Stream token not found.")
    base_url = str(request.base_url).rstrip("/")
    html_page = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']], packages: {{'[+]': ['ams']}} }},
  startup: {{
    ready() {{
      MathJax.startup.defaultReady();
      // Typeset any blocks that arrived before MathJax was ready
      MathJax.startup.promise.then(() => {{
        const page = document.getElementById("ocr-page");
        if (page) MathJax.typesetPromise([page]).catch(console.error);
      }});
    }}
  }}
}};
</script>
<script async src="{base_url}/static/mathjax/tex-svg-full.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ width: 100%; background: #f5f0e8; font-family: serif; font-size: calc(4vw * var(--font-scale, 1)); line-height: 1.6; padding: 0 1em; }}
  #ocr-status {{ font-family: sans-serif; font-size: 14px; color: #666; padding: 8px 0; }}
  .block {{ margin-bottom: 0.8em; }}
  .block p {{ margin: 0.4em 0; }}
  .block.formula {{ overflow-x: auto; }}
  .block.figure img {{ max-width: 100%; height: auto; display: block; margin: 0.5em 0; }}
  mjx-container[display="true"] {{ display: block; overflow-x: auto; }}
</style>
</head>
<body>
<div id="ocr-status">Analysing layout\u2026</div>
<div id="page-container"></div>
<script>
  const evtSource = new EventSource("{base_url}/ocr_stream/{token}/events");
  const container = document.getElementById("page-container");
  const status = document.getElementById("ocr-status");

  evtSource.addEventListener("layout_done", e => {{
    const d = JSON.parse(e.data);
    status.textContent = `Found ${{d.n_blocks}} blocks, running OCR\u2026`;
    container.innerHTML = `<div id="ocr-page"></div>`;
  }});

  evtSource.addEventListener("block", e => {{
    const d = JSON.parse(e.data);
    const page = document.getElementById("ocr-page");
    if (!page) return;
    const existing = document.getElementById("block-" + d.index);
    if (existing) {{
      existing.outerHTML = d.html;
    }} else {{
      page.insertAdjacentHTML("beforeend", d.html);
    }}
    if (window.MathJax) {{
      MathJax.typesetPromise([document.getElementById("block-" + d.index)]).catch(console.error);
    }}
  }});

  evtSource.addEventListener("done", e => {{
    status.textContent = "OCR complete.";
    evtSource.close();
    setTimeout(() => {{ status.style.display = "none"; }}, 5000);
    // Signal Flet app that rendering is complete
    window.location.hash = "ocr-done";
  }});

  evtSource.addEventListener("error", e => {{
    const d = JSON.parse(e.data || '{{"message":"unknown error"}}');
    status.textContent = "OCR error: " + d.message;
    evtSource.close();
    window.location.hash = "ocr-done";
  }});

  evtSource.onerror = () => {{
    status.textContent = "Connection lost.";
    evtSource.close();
    window.location.hash = "ocr-done";
  }};
</script>
</body>
</html>"""
    return Response(content=html_page, media_type="text/html")


@app.get("/ocr_stream/{token}/events")
async def ocr_stream_events(token: str):
    """SSE endpoint that streams OCR block events for a given job token."""
    if token not in _stream_jobs:
        raise HTTPException(status_code=404, detail="Stream token not found.")

    queue = _stream_jobs[token]

    async def event_generator():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    # Job complete — clean up
                    _stream_jobs.pop(token, None)
                    yield f"event: done\ndata: {{}}\n\n"
                    return
                event, payload = item
                import json
                yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            _stream_jobs.pop(token, None)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/test")
async def test_page(request: Request):
    """Diagnostic page to check JS and MathJax rendering in WebView."""
    base_url = str(request.base_url).rstrip("/")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WebView Test</title>
<script>
MathJax = {{
  tex: {{ displayMath: [['$$', '$$']], inlineMath: [['$', '$']] }},
  startup: {{
    ready() {{
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => MathJax.typesetPromise());
    }}
  }}
}};
</script>
<script src="{base_url}/static/mathjax/tex-chtml-full.js"></script>
<style>body {{ font-family: sans-serif; padding: 1em; }}</style>
</head>
<body>
<h2>WebView Diagnostic</h2>
<p>JS status: <strong id="js-check" style="color:red">OFF</strong></p>
<script>document.getElementById('js-check').textContent = 'ON'; document.getElementById('js-check').style.color = 'green';</script>
<p>MathJax test (should render as formula):</p>
<p>$$ x^2 + y^2 = z^2 $$</p>
<p>Inline: the formula $E = mc^2$ should be rendered.</p>
</body>
</html>"""
    return Response(content=html, media_type="text/html")
