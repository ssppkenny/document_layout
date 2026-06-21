"""EPUB exporter: convert a PDF or DjVu file to EPUB 3.

Pipeline per page:
  1. load_page()          — render to BGR image
  2. layout_from_array()  — detect blocks (YOLO dual-pass)
  3. Crop images/figures  — saved as PNG files inside the EPUB
  4. Batch OCR            — LightOnOCR for text + formula blocks
  5. Math pre-render      — Playwright + MathJax SVG for all LaTeX
  6. Assemble XHTML       — one big content.xhtml, sections per page
  7. Pack EPUB            — stdlib zipfile, OPF + NAV + NCX

Resumable: each completed page is checkpointed as JSON in --checkpoint-dir.
On re-run with --resume, already-done pages are skipped.

Usage:
    python -m ocr_reflow.epub_export INPUT [OPTIONS]
    python epub_export.py INPUT [OPTIONS]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label sets (epub-local; includes table_and_caption fix)
# ---------------------------------------------------------------------------

_TEXT_LABELS = {"plain text", "title", "titled_block_title", "titled_block_body", "figure_caption"}
_FORMULA_LABELS = {"isolate_formula", "isolate_formula_and_caption"}
_IMAGE_LABELS = {
    "figure", "figure_and_caption",
    "table", "table_and_caption", "table_caption", "table_footnote",
}
_SKIP_LABELS = {"abandon"}
_OCR_LABELS = _TEXT_LABELS | _FORMULA_LABELS
_TITLE_LABELS = {"title", "titled_block_title"}

# ---------------------------------------------------------------------------
# Tiny static HTTP server (serves MathJax JS to Playwright)
# ---------------------------------------------------------------------------

class _SilentHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses log output (used for MathJax static server)."""
    def log_message(self, *args):
        pass


_static_server: HTTPServer | None = None
_static_server_port: int | None = None
_static_server_lock = threading.Lock()


def _ensure_static_server(static_dir: Path) -> int:
    """Start a one-shot HTTP server serving static_dir; return port."""
    global _static_server, _static_server_port
    with _static_server_lock:
        if _static_server is not None:
            return _static_server_port

        os.chdir(static_dir)
        server = HTTPServer(("127.0.0.1", 0), _SilentHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        _static_server = server
        _static_server_port = port
        logger.debug("Static server started on port %d serving %s", port, static_dir)
        return port


# ---------------------------------------------------------------------------
# SVG ID uniquifier — prevents duplicate-id errors when multiple SVGs are
# inlined into a single XHTML document.
# ---------------------------------------------------------------------------

_svg_counter = 0

# Precompiled patterns for SVG ID uniquification — single pass over the string
# instead of 3 full-string scans per ID.
_RE_SVG_ID = re.compile(r'\bid="([^"]+)"')
_RE_SVG_HREF = re.compile(r'href="#([^"]+)"')
_RE_SVG_URL = re.compile(r'url\(#([^)]+)\)')


def _uniquify_svg_ids(svg: str) -> str:
    """Rewrite all id= attributes and their href/#... / url(#...) references
    inside an SVG string so they are globally unique across the document."""
    global _svg_counter
    _svg_counter += 1
    prefix = f"mjx{_svg_counter}-"

    # Collect all ids, build a mapping, then do 3 single-pass regex subs.
    ids = _RE_SVG_ID.findall(svg)
    if not ids:
        return svg
    id_map = {old: prefix + old for old in ids}

    svg = _RE_SVG_ID.sub(lambda m: f'id="{id_map[m.group(1)]}"', svg)
    svg = _RE_SVG_HREF.sub(lambda m: f'href="#{id_map.get(m.group(1), m.group(1))}"', svg)
    svg = _RE_SVG_URL.sub(lambda m: f'url(#{id_map.get(m.group(1), m.group(1))})', svg)
    return svg


# ---------------------------------------------------------------------------
# MathRenderer — Playwright + MathJax SVG
# ---------------------------------------------------------------------------

class MathRenderer:
    """Pre-render LaTeX to SVG using a headless Chromium + MathJax.

    One browser instance is reused for the entire export job.
    Results are cached so identical formulas are only rendered once.
    """

    def __init__(self, mathjax_js_path: Path):
        self._mathjax_js_path = mathjax_js_path
        self._cache: dict[tuple[str, bool], str] = {}
        self._browser = None
        self._page = None
        self._port: int | None = None

    def _start(self):
        from playwright.sync_api import sync_playwright
        static_dir = self._mathjax_js_path.parent
        self._port = _ensure_static_server(static_dir)

        self._pw_ctx = sync_playwright()
        self._pw = self._pw_ctx.__enter__()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page()

        mathjax_url = f"http://127.0.0.1:{self._port}/{self._mathjax_js_path.name}"
        bootstrap_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    packages: {{'[+]': ['ams']}}
  }},
  svg: {{ fontCache: 'local' }},
  startup: {{
    typeset: false
  }}
}};
</script>
<script src="{mathjax_url}"></script>
</head><body id="container"></body></html>"""

        self._page.set_content(bootstrap_html)
        # Wait for MathJax to finish loading
        self._page.wait_for_function("() => window.MathJax && window.MathJax.typesetPromise")
        logger.debug("MathRenderer: MathJax loaded")

    def _ensure_started(self):
        if self._page is None:
            self._start()

    def render(self, latex: str, display: bool) -> str:
        """Render latex to an SVG string. display=True for block math."""
        key = (latex, display)
        safe_latex = html.escape(latex)
        if key in self._cache:
            cached = self._cache[key]
            if cached is None:
                return f'<code class="math-fallback" data-latex="{safe_latex}">{safe_latex}</code>'
            result = _uniquify_svg_ids(cached)
            return result.replace("<svg", f'<svg data-latex="{safe_latex}"', 1)

        self._ensure_started()

        if display:
            math_str = f"$$\n{latex}\n$$"
        else:
            math_str = f"${latex}$"

        svg = self._page.evaluate(
            """async ([mathStr]) => {
                const container = document.getElementById('container');
                container.innerHTML = '';
                const el = document.createElement('div');
                el.textContent = mathStr;
                container.appendChild(el);
                await MathJax.typesetPromise([container]);
                const svg = container.querySelector('svg');
                return svg ? svg.outerHTML : null;
            }""",
            [math_str],
        )

        if svg is None:
            self._cache[key] = None
            return f'<code class="math-fallback" data-latex="{safe_latex}">{safe_latex}</code>'
        else:
            self._cache[key] = svg
            result = _uniquify_svg_ids(svg)
            return result.replace("<svg", f'<svg data-latex="{safe_latex}"', 1)

    def render_batch(self, formulas: list[tuple[str, bool]]) -> list[str]:
        """Render multiple LaTeX formulas in a single Playwright round-trip.

        ``formulas`` is a list of (latex, display) pairs.  Returns a list of
        SVG strings (or fallback <code> elements) in the same order.
        Cache hits are resolved without a round-trip; only cache misses are
        sent to MathJax in one batched ``page.evaluate()``.
        """
        results: list[str] = [None] * len(formulas)
        misses: list[int] = []
        miss_inputs: list[str] = []

        for idx, (latex, display) in enumerate(formulas):
            key = (latex, display)
            safe_latex = html.escape(latex)
            if key in self._cache:
                cached = self._cache[key]
                if cached is None:
                    results[idx] = f'<code class="math-fallback" data-latex="{safe_latex}">{safe_latex}</code>'
                else:
                    r = _uniquify_svg_ids(cached)
                    results[idx] = r.replace("<svg", f'<svg data-latex="{safe_latex}"', 1)
            else:
                misses.append(idx)
                if display:
                    miss_inputs.append(f"$$\n{latex}\n$$")
                else:
                    miss_inputs.append(f"${latex}$")

        if not miss_inputs:
            return results

        self._ensure_started()

        svgs = self._page.evaluate(
            """async ([mathStrs]) => {
                const container = document.getElementById('container');
                container.innerHTML = '';
                const els = mathStrs.map(s => {
                    const el = document.createElement('div');
                    el.textContent = s;
                    container.appendChild(el);
                    return el;
                });
                await MathJax.typesetPromise(els);
                return els.map(el => {
                    const svg = el.querySelector('svg');
                    return svg ? svg.outerHTML : null;
                });
            }""",
            [miss_inputs],
        )

        for j, idx in enumerate(misses):
            latex, display = formulas[idx]
            safe_latex = html.escape(latex)
            svg = svgs[j]
            if svg is None:
                self._cache[(latex, display)] = None
                results[idx] = f'<code class="math-fallback" data-latex="{safe_latex}">{safe_latex}</code>'
            else:
                self._cache[(latex, display)] = svg
                r = _uniquify_svg_ids(svg)
                results[idx] = r.replace("<svg", f'<svg data-latex="{safe_latex}"', 1)

        return results

    def close(self):
        if self._browser:
            self._browser.close()
            self._pw_ctx.__exit__(None, None, None)
            self._browser = None
            self._page = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# LaTeX extraction and substitution helpers
# ---------------------------------------------------------------------------

# Matches $$...$$  (display) — non-greedy, allows newlines
_RE_DISPLAY = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
# Matches $...$  (inline) — greedy so $0\leq n\leq9$ stays as one expression
_RE_INLINE = re.compile(r'\$([^\$\n]+)\$')


def _render_math_in_html(fragment: str, renderer: MathRenderer) -> str:
    """Replace all $...$ and $$...$$ in an HTML fragment with inline SVG.

    Collects all formulas in the fragment and renders them in a single
    batched Playwright round-trip (via ``render_batch``) instead of one
    ``page.evaluate()`` per formula.
    """

    # Collect all formula matches with their positions, then batch-render.
    matches: list[tuple[str, bool, int, int, int]] = []  # (latex, display, start, end, order)
    for m in _RE_DISPLAY.finditer(fragment):
        matches.append((m.group(1).strip(), True, m.start(), m.end(), len(matches)))
    for m in _RE_INLINE.finditer(fragment):
        matches.append((m.group(1).strip(), False, m.start(), m.end(), len(matches)))

    if not matches:
        return fragment

    # Sort by position so we can rebuild the string left-to-right
    matches.sort(key=lambda x: x[2])

    formulas = [(m[0], m[1]) for m in matches]
    svgs = renderer.render_batch(formulas)

    # Rebuild fragment with SVGs inserted
    parts = []
    last_end = 0
    for i, m in enumerate(matches):
        latex, display, start, end, _ = m
        parts.append(fragment[last_end:start])
        parts.append(svgs[i])
        last_end = end
    parts.append(fragment[last_end:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Reuse OCR helpers from ocr_export_layout (import, don't copy)
# ---------------------------------------------------------------------------

# Module-level caches — avoid try/except ImportError per call.
_layout_fn = None
_ocr_helpers = None
_metrics_fn = None


def _get_layout():
    """Import and return the layout detection function from layout.py."""
    global _layout_fn
    if _layout_fn is None:
        try:
            from layout import layout_from_array
        except ImportError:
            from ocr_reflow.layout import layout_from_array
        _layout_fn = layout_from_array
    return _layout_fn


def _get_ocr_helpers():
    """Import and return OCR helper functions from ocr_export_layout."""
    global _ocr_helpers
    if _ocr_helpers is None:
        try:
            from ocr_export_layout import (
                _crop, _ocr_blocks_batch, _lightonocr_to_html, _split_plain_text_crop,
            )
        except ImportError:
            from ocr_reflow.ocr_export_layout import (
                _crop, _ocr_blocks_batch, _lightonocr_to_html, _split_plain_text_crop,
            )
        _ocr_helpers = (_crop, _ocr_blocks_batch, _lightonocr_to_html, _split_plain_text_crop)
    return _ocr_helpers


def _get_metrics_fn():
    """Return ocr_export_layout.block_hallucination_metrics (cached)."""
    global _metrics_fn
    if _metrics_fn is None:
        try:
            from ocr_export_layout import block_hallucination_metrics
        except ImportError:
            from ocr_reflow.ocr_export_layout import block_hallucination_metrics
        _metrics_fn = block_hallucination_metrics
    return _metrics_fn


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def _is_grayscale(img_bgr: np.ndarray, sat_threshold: int = 20) -> bool:
    """Return True if the image has no meaningful color (scanned grayscale)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return int(hsv[:, :, 1].max()) < sat_threshold


def _normalize_to_white(img_bgr: np.ndarray) -> np.ndarray:
    """Scale pixel values so the background (95th percentile) maps to white."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_level = np.percentile(gray, 95)
    if bg_level < 10:
        return img_bgr  # very dark image — don't touch
    scale = 255.0 / bg_level
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.clip(rgb * scale, 0, 255).astype(np.uint8)


def _bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    if _is_grayscale(img_bgr):
        rgb = _normalize_to_white(img_bgr)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # cv2.imencode is ~1.5-3x faster than PIL for megapixel images.
    success, buf = cv2.imencode(".png", rgb)
    if not success:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Language aliases: --lang ru  →  ru_RU, etc.
# ---------------------------------------------------------------------------

_LANG_ALIASES: dict[str, str] = {
    "ru": "ru_RU",
    "en": "en_US",
    "en_us": "en_US",
    "en_gb": "en_GB",
    "de": "de_DE",
    "fr": "fr_FR",
    "es": "es_ES",
}


def _resolve_lang(lang: str) -> str:
    """Map short aliases to hunspell dict names; pass through if already full."""
    return _LANG_ALIASES.get(lang.lower(), lang)


def _resolve_langs(lang_str: str) -> str:
    """Resolve a comma-separated lang string, e.g. 'ru,en' → 'ru_RU,en_US'."""
    parts = [_resolve_lang(p.strip()) for p in lang_str.split(",") if p.strip()]
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Spell check helpers
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def _preserve_case(original: str, replacement: str) -> str:
    """Apply the capitalisation pattern of *original* to *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original.istitle() or (original and original[0].isupper()):
        return replacement[0].upper() + replacement[1:] if replacement else replacement
    return replacement


# Regex to split text into plain / math segments.
# Groups: (plain, math) alternating; math = $...$ or ```latex...```
_MATH_SPLIT_RE = re.compile(
    r'(```latex.*?```|\$\$.*?\$\$|\$[^$\n]+?\$)',
    re.DOTALL,
)


def _spellcheck_text(text: str, hunspell_lang: str) -> str:
    """Run hunspell on plain-text segments of *text* — only 100%-certain fixes.

    A correction is accepted only when:
    - hunspell returns exactly one suggestion, AND
    - edit distance ≤ 1 for words shorter than 8 chars, ≤ 2 for words ≥ 8 chars.
    Math segments ($...$, ```latex```) are left untouched.

    Note: this is a conservative inline pass used during OCR. Ambiguous
    corrections (multiple suggestions) are deferred to the advanced post-hoc
    pass at the end of main().
    """
    segments = _MATH_SPLIT_RE.split(text)
    # _MATH_SPLIT_RE.split() returns [plain, math, plain, math, ...]
    result_parts: list[str] = []
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            # Math segment — pass through unchanged
            result_parts.append(seg)
            continue
        if not seg.strip():
            result_parts.append(seg)
            continue
        result_parts.append(_spellcheck_plain(seg, hunspell_lang))
    return "".join(result_parts)


def _spellcheck_plain(text: str, hunspell_lang: str) -> str:
    """Apply hunspell corrections to a plain-text (non-math) string."""
    import subprocess

    # Step 1: find misspelled words
    try:
        proc = subprocess.run(
            ["hunspell", "-d", hunspell_lang, "-l"],
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        logger.warning(f"hunspell -l failed: {e}")
        return text

    misspelled = set(proc.stdout.split())
    if not misspelled:
        return text

    # Step 2: get suggestions for each misspelled word
    corrections: dict[str, str] = {}
    for word in misspelled:
        try:
            proc2 = subprocess.run(
                ["hunspell", "-d", hunspell_lang],
                input=word,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"hunspell suggestion failed for '{word}': {e}")
            continue

        # hunspell output lines:
        #   "& word count offset: sug1, sug2, ..."  → misspelled with suggestions
        #   "# word offset"                          → no suggestions
        #   "* word"                                 → correct
        suggestions: list[str] = []
        for line in proc2.stdout.splitlines():
            if line.startswith("& "):
                # "& original count offset: s1, s2, ..."
                after_colon = line.split(":", 1)
                if len(after_colon) == 2:
                    suggestions = [s.strip() for s in after_colon[1].split(",")]
                break

        if len(suggestions) != 1:
            continue  # multiple suggestions → defer to advanced pass

        suggestion = suggestions[0]
        max_dist = 1 if len(word) < 8 else 2
        if _edit_distance(word.lower(), suggestion.lower()) <= max_dist:
            corrections[word] = _preserve_case(word, suggestion)

    if not corrections:
        return text

    # Step 3: replace whole-word occurrences, longest words first (avoid partial matches)
    for wrong in sorted(corrections, key=len, reverse=True):
        right = corrections[wrong]
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text)

    return text


def _classify_image(geom, page_w: int) -> str:
    """Classify an image region as img-left, img-right, or img-full based on
    what fraction of its width lies in each page half."""
    x1, _, x2, _ = geom.bounds
    img_w = x2 - x1
    if img_w <= 0:
        return "img-full"
    mid = page_w / 2
    left_frac = (min(x2, mid) - x1) / img_w
    right_frac = (x2 - max(x1, mid)) / img_w
    if left_frac >= 0.8:
        return "img-left"
    if right_frac >= 0.8:
        return "img-right"
    return "img-full"


def process_page(
    img_bgr: np.ndarray,
    page_num: int,           # 1-based
    renderer: MathRenderer,
    checkpoint_dir: Path,
    spellcheck_lang: str | None = None,
) -> dict:
    """Process one page and return a result dict (also saved as checkpoint JSON).

    Result dict keys:
      page_num      : int
      html_fragments: list of XHTML fragment strings (in reading order)
      images        : list of {"name": str, "data_b64": str}  (PNG files)
      titles        : list of str  (detected title texts for TOC)
    """
    checkpoint_file = checkpoint_dir / f"page_{page_num:04d}.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)

    t0 = time.perf_counter()
    logger.info("  Page %d: layout analysis...", page_num)

    layout_from_array = _get_layout()
    _crop, _ocr_blocks_batch, _lightonocr_to_html, _split_plain_text_crop = _get_ocr_helpers()

    # Expected document language(s) as base ISO codes (e.g. {"ru"}) for the
    # per-block language hallucination gate.  Empty when language is unknown.
    expected_langs = {
        p.split('_')[0].split('-')[0].strip().lower()
        for p in (spellcheck_lang or "").split(',') if p.strip()
    }

    blocks = layout_from_array(img_bgr)
    blocks = sorted(blocks, key=lambda b: b[0].bounds[1])

    # First pass: crop all blocks
    # block_data: (geom, label, crop_or_None, masked_or_None)
    # masked is computed lazily on first _mask() call and cached for reuse.
    block_data: list[list] = []  # mutable lists for in-place masked cache
    for geom, label in blocks:
        if label in _SKIP_LABELS:
            block_data.append([geom, label, None, None])
            continue
        crop = _crop(img_bgr, geom)
        if crop.size == 0:
            block_data.append([geom, label, None, None])
            continue
        block_data.append([geom, label, crop, None])

    # Collect image bboxes for masking overlapping text crops.
    # Only figure-type labels are masked (tables contain text that should be OCR'd).
    _FIGURE_MASK_LABELS = {"figure", "figure_and_caption", "figure_caption"}
    image_bboxes = []
    for geom, label, _c, _m in block_data:
        if label in _FIGURE_MASK_LABELS:
            x1, y1, x2, y2 = geom.bounds
            image_bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    def _mask(entry) -> np.ndarray:
        """Return masked crop for a block_data entry, computing+caching on first call."""
        geom, label, crop, masked = entry
        if masked is not None:
            return masked
        if crop is None:
            return None
        bx1, by1, bx2, by2 = [int(v) for v in geom.bounds]
        bw, bh = bx2 - bx1, by2 - by1
        result = crop.copy()
        for ix1, iy1, ix2, iy2 in image_bboxes:
            ov_x1 = max(bx1, ix1)
            ov_y1 = max(by1, iy1)
            ov_x2 = min(bx2, ix2)
            ov_y2 = min(by2, iy2)
            if ov_x2 > ov_x1 and ov_y2 > ov_y1:
                ov_area = (ov_x2 - ov_x1) * (ov_y2 - ov_y1)
                block_area = bw * bh if bw > 0 and bh > 0 else 1
                if ov_area / block_area > 0.9:
                    continue
                ox1, oy1 = ov_x1 - bx1, ov_y1 - by1
                ox2, oy2 = ov_x2 - bx1, ov_y2 - by1
                result[oy1:oy2, ox1:ox2] = 255
        entry[3] = result
        return result

    # Collect OCR crops (with XY-cut splitting for tall text blocks).
    # Pre-OCR filtering skips tiny blocks (area < _MIN_OCR_AREA) and
    # deduplicates identical sub-crops, avoiding wasted VLM calls.
    # Output is identical: tiny blocks produce hallucinations that the
    # post-OCR gate drops anyway; identical crops yield the same text.
    _MIN_OCR_AREA = 5000
    ocr_indices = []   # original block_data indices
    ocr_crops = []     # flat list of sub-crops (non-skipped only)
    ocr_split_counts = []  # how many sub-crops per block (0 if pre-skipped)
    n_skipped_area = 0

    for i, entry in enumerate(block_data):
        geom, label, crop, _ = entry
        if label not in _OCR_LABELS or crop is None:
            continue
        if label in _TITLE_LABELS:
            continue  # title blocks OCR'd separately with Tesseract
        masked = _mask(entry)
        block_area = int(masked.shape[0]) * int(masked.shape[1])
        if block_area < _MIN_OCR_AREA:
            ocr_indices.append(i)
            ocr_split_counts.append(0)
            n_skipped_area += 1
            continue
        if label in _TEXT_LABELS:
            sub_crops = _split_plain_text_crop(masked)
        else:
            sub_crops = [masked]
        ocr_indices.append(i)
        ocr_crops.extend(sub_crops)
        ocr_split_counts.append(len(sub_crops))

    # Deduplicate sub-crops: hash each and reuse OCR result for identical crops.
    unique_crops = []
    crop_to_unique = []   # for each ocr_crops[i]: index into unique_crops
    _seen_hashes = {}
    n_skipped_dedup = 0

    for c in ocr_crops:
        h = hashlib.blake2b(c.tobytes(), digest_size=16).hexdigest()
        if h in _seen_hashes:
            crop_to_unique.append(_seen_hashes[h])
            n_skipped_dedup += 1
        else:
            _seen_hashes[h] = len(unique_crops)
            crop_to_unique.append(len(unique_crops))
            unique_crops.append(c)

    n_ocr = len(ocr_indices)
    n_img = sum(1 for _, label, crop, _ in block_data if label in _IMAGE_LABELS and crop is not None)
    logger.info("  Page %d: %d blocks — OCR:%d images:%d (skipped:%d area, %d dedup → %d VLM calls)",
                page_num, len(block_data), n_ocr, n_img,
                n_skipped_area, n_skipped_dedup, len(unique_crops))

    # Run batch OCR on unique non-tiny crops only
    unique_texts = _ocr_blocks_batch(unique_crops) if unique_crops else []

    # Reassemble flat_texts aligned with original ocr_crops
    flat_texts = [unique_texts[crop_to_unique[j]] for j in range(len(ocr_crops))]

    # Re-assemble per-block OCR text
    ocr_results: dict[int, str] = {}
    flat_idx = 0
    for k, i in enumerate(ocr_indices):
        count = ocr_split_counts[k]
        parts = flat_texts[flat_idx:flat_idx + count]
        ocr_results[i] = "\n\n".join(p for p in parts if p)
        flat_idx += count

    # OCR title blocks with Tesseract (deterministic, no hallucination).
    # VLM-based OCR fabricates content on small/empty header crops;
    # Tesseract faithfully transcribes whatever text is actually present.
    tess_lang = _to_tess_lang(spellcheck_lang) if spellcheck_lang else "eng"
    for i, entry in enumerate(block_data):
        geom, label, crop, _ = entry
        if label in _TITLE_LABELS and crop is not None:
            masked = _mask(entry)
            text = _tesseract_heading_ocr(masked, tess_lang)
            if text:
                ocr_results[i] = text

    # Compute which image blocks are covered by OCR'd text blocks.
    # When a table/figure overlaps >50% by area with text blocks that
    # produced non-empty OCR output, skip the image — the content is
    # already captured by OCR.
    covered_image_indices: set[int] = set()
    for i, entry in enumerate(block_data):
        geom, label, crop, _ = entry
        if label not in _IMAGE_LABELS or crop is None:
            continue
        ix1, iy1, ix2, iy2 = geom.bounds
        img_area = (ix2 - ix1) * (iy2 - iy1)
        if img_area <= 0:
            continue
        total_overlap = 0
        for j, (tgeom, tlabel, _tc, _tm) in enumerate(block_data):
            if tlabel not in _TEXT_LABELS:
                continue
            if not ocr_results.get(j, "").strip():
                continue
            tx1, ty1, tx2, ty2 = tgeom.bounds
            ov_x1 = max(ix1, tx1)
            ov_y1 = max(iy1, ty1)
            ov_x2 = min(ix2, tx2)
            ov_y2 = min(iy2, ty2)
            if ov_x2 > ov_x1 and ov_y2 > ov_y1:
                total_overlap += (ov_x2 - ov_x1) * (ov_y2 - ov_y1)
        if total_overlap / img_area > 0.5:
            covered_image_indices.add(i)

    # Build XHTML fragments and collect images
    html_fragments = []
    images = []
    titles = []
    img_counter = 0

    for i, entry in enumerate(block_data):
        geom, label, crop, _ = entry
        if crop is None:
            continue

        if label in _OCR_LABELS:
            ocr_text = ocr_results.get(i, "")
            raw_html = _lightonocr_to_html(ocr_text)

            # Skip blocks whose OCR output is effectively empty (VLM hallucination
            # on tiny/near-empty crops produces boilerplate that was stripped above).
            # Only skip if there is NO math content AND no real text.
            if label not in _TITLE_LABELS:
                plain_text = re.sub(r'<[^>]+>', '', raw_html).strip()
                has_display = '$$' in plain_text
                has_inline = '$' in plain_text
                plain_text = re.sub(r'\$\$.*?\$\$', '', plain_text, flags=re.DOTALL)
                plain_text = re.sub(r'\$[^$]*\$', '', plain_text)
                plain_text = plain_text.strip()
                if len(plain_text) <= 5 and not (has_display or has_inline):
                    continue

                # Robust anti-hallucination gates (thresholds calibrated against
                # known good/bad blocks). Repetition/macro-spam is already handled
                # upstream in _strip_vlm_hallucinations; here we add the signals
                # that need the crop image and the document language:
                #   * blank crop : a near-blank region (ink < 2%) that still
                #     produced verbose output — the classic VLM blank-doc
                #     hallucination (good sparse formulas have ink >= 5%).
                #   * tiny crop : a crop smaller than 10000px² (~100×100)
                #     producing > 30 chars — too small to contain real text
                #     (smallest legitimate block is ~28000px²). Catches single
                #     invented formulas from layout fragments.
                #   * dense crop : output far exceeds what the crop can hold
                #     (> 0.03 char/px, > 100 chars) — a small region returning
                #     a page of invented "filler" math (e.g. a 44x45px box ->
                #     a page of example matrices). Real blocks stay under 0.002.
                #   * wrong language : confident prose in a language other than
                #     the document's (e.g. a fabricated English medical essay in
                #     a Russian book). Guarded by conf >= 0.85 AND >= 10 alpha
                #     words so Cyrillic-cousin misdetections and formula blocks
                #     with stray latin letters are never dropped.
                try:
                    _m = _get_metrics_fn()(ocr_text, _mask(entry), expected_langs)
                    if _m["ink_ratio"] < 0.02 and _m["ocr_len"] > 50:
                        logger.info("  Page %d: dropped blank-crop hallucination "
                                    "(ink=%.3f len=%d)", page_num,
                                    _m["ink_ratio"], _m["ocr_len"])
                        continue
                    if _m["area"] < 10000 and _m["ocr_len"] > 30:
                        logger.info("  Page %d: dropped tiny-crop hallucination "
                                    "(area=%d len=%d)", page_num,
                                    _m["area"], _m["ocr_len"])
                        continue
                    if _m["ocr_len"] > 100 and _m["char_density"] > 0.03:
                        logger.info("  Page %d: dropped dense-crop hallucination "
                                    "(area=%d len=%d density=%.3f)", page_num,
                                    _m["area"], _m["ocr_len"], _m["char_density"])
                        continue
                    if (expected_langs and _m["lang"]
                            and _m["lang"] not in expected_langs
                            and _m["conf"] >= 0.85 and _m["n_alpha_words"] >= 10):
                        logger.info("  Page %d: dropped wrong-language hallucination "
                                    "(lang=%s conf=%.2f words=%d)", page_num,
                                    _m["lang"], _m["conf"], _m["n_alpha_words"])
                        continue
                except Exception as _e:
                    logger.warning("hallucination gate failed: %s", _e)

            # Pre-render math to SVG
            rendered = _render_math_in_html(raw_html, renderer)
            if label in _TITLE_LABELS:
                # Wrap in <h1> for semantic structure; replace any <p> wrapper
                inner = re.sub(r'^<p>(.*)</p>\s*$', r'\1', rendered.strip(), flags=re.DOTALL)
                rendered = f"<h1>{inner}</h1>\n"
                block_class = "title"
            else:
                block_class = "text" if label in _TEXT_LABELS else "formula"
            if block_class == "formula":
                # Put each top-level formula SVG on its own line. Match only
                # boundaries between two top-level rendered formulas (the second
                # <svg carries a data-latex attr); never insert inside a single
                # formula's internal/nested <svg> children.
                rendered = re.sub(
                    r'(</svg>)\s*(<svg\s[^>]*\bdata-latex\b)',
                    r'\1<br/>\2',
                    rendered,
                )
                fragment = f'<div class="block formula">\n<div class="formula-group">\n{rendered}</div>\n</div>\n'
            else:
                fragment = f'<div class="block {block_class}">\n{rendered}</div>\n'

            # Deduplicate adjacent text fragments: if this fragment's plain
            # text is wholly contained in the previous text fragment, skip it.
            # Layout overlaps often cause the last line to be OCR'd twice.
            if block_class == "text" and html_fragments:
                prev = html_fragments[-1]
                if 'class="block text"' in prev:
                    prev_text = re.sub(r'<[^>]+>', '', prev)
                    prev_text = re.sub(r'\s+', ' ', prev_text).strip()
                    this_text = re.sub(r'<[^>]+>', '', rendered)
                    this_text = re.sub(r'\s+', ' ', this_text).strip()
                    if this_text and this_text in prev_text:
                        continue

            html_fragments.append(fragment)

            # Collect title text for TOC
            if label in _TITLE_LABELS:
                text = ocr_results.get(i, "").strip()
                if text:
                    titles.append(html.escape(text[:120]))

        elif label in _IMAGE_LABELS:
            if i in covered_image_indices:
                continue
            img_name = f"p{page_num:04d}_b{img_counter:02d}.png"
            img_counter += 1
            png_bytes = _bgr_to_png_bytes(crop)
            images.append({
                "name": img_name,
                "data_b64": base64.b64encode(png_bytes).decode("ascii"),
            })
            alt_text = html.escape(label)
            img_cls = _classify_image(geom, img_bgr.shape[1])
            fragment = (
                f'<div class="block figure {img_cls}">\n'
                f'<img class="{img_cls}" src="images/{img_name}" alt="{alt_text}"/>\n'
                f'</div>\n'
            )
            html_fragments.append(fragment)

        else:
            # Unknown label — treat as image
            img_name = f"p{page_num:04d}_b{img_counter:02d}.png"
            img_counter += 1
            png_bytes = _bgr_to_png_bytes(crop)
            images.append({
                "name": img_name,
                "data_b64": base64.b64encode(png_bytes).decode("ascii"),
            })
            img_cls = _classify_image(geom, img_bgr.shape[1])
            fragment = (
                f'<div class="block figure {img_cls}">\n'
                f'<img class="{img_cls}" src="images/{img_name}" alt="{html.escape(label)}"/>\n'
                f'</div>\n'
            )
            html_fragments.append(fragment)

    result = {
        "page_num": page_num,
        "html_fragments": html_fragments,
        "images": images,
        "titles": titles,
    }

    # Save checkpoint in a background thread so it doesn't block the next page.
    # Write to a temp file first, then rename for atomicity.
    def _write_checkpoint():
        tmp = checkpoint_file.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(result, f)
            tmp.rename(checkpoint_file)
        except Exception as e:
            logger.warning("Checkpoint write failed for page %d: %s", page_num, e)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=_write_checkpoint, daemon=True).start()

    logger.info("  Page %d: done in %.1fs", page_num, time.perf_counter() - t0)
    return result


# ---------------------------------------------------------------------------
# EPUB assembly
# ---------------------------------------------------------------------------

_EPUB_CSS = """\
body {
  font-family: serif;
  font-size: 1em;
  line-height: 1.6;
  margin: 0;
  padding: 0.5em 1em;
}
.block {
  margin-bottom: 0.8em;
}
.block p {
  margin: 0.3em 0;
}
.block.figure {
  overflow: hidden;
}
.block.figure img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.5em auto;
}
.block.figure.img-left {
  float: left;
  width: 50%;
  margin: 0.5em 1em 0.5em 0;
}
.block.figure.img-left img {
  width: 100%;
}
.block.figure.img-right {
  float: right;
  width: 50%;
  margin: 0.5em 0 0.5em 1em;
}
.block.figure.img-right img {
  width: 100%;
}
.block.figure.img-full img {
  width: 100%;
}
.block.formula {
  text-align: center;
  overflow-x: auto;
}
.block.formula .formula-group {
  display: inline-block;
  text-align: left;
}
.block.title {
  margin: 1.2em 0 0.4em 0;
}
.block.title h1 {
  font-size: 1.4em;
  font-weight: bold;
  margin: 0;
}
.page-break {
  border: none;
  border-top: 1px solid #ccc;
  margin: 1.5em 0;
}
h1, h2, h3 {
  font-family: serif;
  margin: 0.6em 0 0.3em 0;
}
svg {
  max-width: 100%;
  height: auto;
}
img.img-left {
  float: left;
  width: 50%;
  margin: 0.5em 1em 0.5em 0;
}
img.img-right {
  float: right;
  width: 50%;
  margin: 0.5em 0 0.5em 1em;
}
img.img-full {
  display: block;
  width: 100%;
  margin: 0.5em auto;
}
"""

_CONTAINER_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _build_opf(title: str, author: str, uid: str, image_names: list[str], has_svg: bool = False) -> str:
    """Build the EPUB OPF (package document)."""
    manifest_images = "\n".join(
        f'    <item id="img_{i}" href="images/{name}" media-type="image/png"/>'
        for i, name in enumerate(image_names)
    )
    content_properties = ' properties="svg"' if has_svg else ""
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf"
         unique-identifier="uid" xml:lang="en">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{html.escape(title)}</dc:title>
    <dc:creator>{html.escape(author) if author else "Unknown"}</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="uid">{uid}</dc:identifier>
    <meta property="dcterms:modified">{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="css" href="style.css" media-type="text/css"/>
    <item id="content" href="content.xhtml" media-type="application/xhtml+xml"{content_properties}/>
{manifest_images}
  </manifest>
  <spine toc="ncx">
    <itemref idref="content"/>
  </spine>
</package>
"""


def _build_nav(title: str, toc_entries: list[dict]) -> str:
    """Build the EPUB navigation document (nav.xhtml)."""
    items = "\n".join(
        f'      <li><a href="content.xhtml#page-{e["page"]}">{html.escape(e["text"])}</a></li>'
        for e in toc_entries
    )
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en">
<head><meta charset="utf-8"/><title>{html.escape(title)}</title></head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>
{items}
    </ol>
  </nav>
</body>
</html>
"""


def _build_ncx(title: str, uid: str, toc_entries: list[dict]) -> str:
    """Build the EPUB NCX (table of contents)."""
    nav_points = "\n".join(
        f"""\
  <navPoint id="np-{e['page']}" playOrder="{idx+1}">
    <navLabel><text>{html.escape(e['text'])}</text></navLabel>
    <content src="content.xhtml#page-{e['page']}"/>
  </navPoint>"""
        for idx, e in enumerate(toc_entries)
    )
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN"
  "http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="{uid}"/>
    <meta name="dtb:depth" content="1"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle><text>{html.escape(title)}</text></docTitle>
  <navMap>
{nav_points}
  </navMap>
</ncx>
"""


def _build_content_xhtml(title: str, page_results: list[dict]) -> str:
    """Assemble the content.xhtml EPUB spine document from all page fragments."""
    sections = []
    for result in page_results:
        pn = result["page_num"]
        body = "".join(result["html_fragments"])
        sections.append(
            f'  <section id="page-{pn}" epub:type="bodymatter">\n'
            f'{body}'
            f'  <hr class="page-break"/>\n'
            f'  </section>\n'
        )
    body_content = "\n".join(sections)
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops"
      xml:lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="style.css"/>
</head>
<body>
{body_content}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# TOC building helpers
# ---------------------------------------------------------------------------

_RE_STRIP_TAGS = re.compile(r'<[^>]+>')
_RE_STRIP_MATH = re.compile(r'\$\$.*?\$\$|\$[^\$\n]+?\$', re.DOTALL)
_RE_DOTS = re.compile(r'[\s.·•]{6,}')  # long dot-leader runs to strip from snippets
_RE_DOT_LEADER = re.compile(r'(\. ){4,}|\.{4,}')  # dot-leader pattern in TOC heading lines

# Map user --lang codes (ISO 639-1 / hunspell) to Tesseract language codes.
_HUNSPELL_TO_TESS: dict[str, str] = {
    "ru": "rus", "ru_ru": "rus",
    "en": "eng", "en_us": "eng", "en_gb": "eng",
    "de": "deu", "de_de": "deu",
    "fr": "fra", "fr_fr": "fra",
    "es": "spa", "es_es": "spa",
}


def _to_tess_lang(lang: str) -> str:
    """Convert a hunspell/ISO 639-1 lang code to a Tesseract lang code."""
    return _HUNSPELL_TO_TESS.get(lang.lower().replace("-", "_"), lang)


def _strip_html(text: str) -> str:
    """Remove HTML tags and math delimiters, collapse whitespace."""
    text = _RE_STRIP_TAGS.sub(' ', text)
    text = _RE_STRIP_MATH.sub(' ', text)
    return ' '.join(text.split())


def _normalize_heading(text: str) -> str:
    """Normalize a heading string extracted from OCR.

    Tesseract sometimes splits decorative/spaced-out letters into separate
    whitespace-delimited tokens, e.g. "Первая" may come out as "Перва я,".
    This function merges such fragments back together.

    Algorithm:
      1. Replace newlines with spaces, then strip.
      2. Tokenize on whitespace.
      3. Walk tokens left-to-right. For each token, if its alpha-only content
         has length <= 2 AND it does not end with '.' (not an abbreviation),
         concatenate it directly onto the previous token.
      4. Collapse any remaining multi-spaces and strip.

    Examples:
      "Перва я,"       -> "Первая,"
      "Четверт а я,"   -> "Четвертая,"
      "И. Н. Веселовский" -> "И. Н. Веселовский"  (initials kept)
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return text
    tokens = text.split()
    if not tokens:
        return text
    merged: list[str] = [tokens[0]]
    for tok in tokens[1:]:
        alpha_only = "".join(ch for ch in tok if ch.isalpha())
        punct_suffix = tok[len(alpha_only):]
        is_initial = tok.endswith(".")
        if len(alpha_only) <= 2 and not is_initial and merged:
            prev = merged[-1]
            prev_alpha_end = len(prev)
            while prev_alpha_end > 0 and not prev[prev_alpha_end - 1].isalpha():
                prev_alpha_end -= 1
            prev_core = prev[:prev_alpha_end]
            merged[-1] = prev_core + alpha_only + punct_suffix
        else:
            merged.append(tok)
    return re.sub(r" +", " ", " ".join(merged)).strip()


def _is_valid_heading(text: str) -> bool:
    """Return True if *text* looks like a genuine chapter heading.

    Rejects:
    - Empty strings or strings with no word >= 3 alpha chars.
    - Strings where more than half of non-space chars are non-alphabetic
      (catches math garbage, number sequences, etc.).
    """
    if not text:
        return False
    words = text.split()
    if not any(sum(1 for ch in w if ch.isalpha()) >= 3 for w in words):
        return False
    non_space = [ch for ch in text if ch != " "]
    if non_space:
        non_alpha_ratio = sum(1 for ch in non_space if not ch.isalpha()) / len(non_space)
        if non_alpha_ratio > 0.5:
            return False
    return True


def _is_clean_word(w: str) -> bool:
    """Return True if *w* has uniform letter casing (not mixed-case OCR garbage).

    Accepts: all-lower, all-upper, Title-case, or short words (<= 2 alpha chars).
    Rejects: mixed internal case like 'оНЕС', 'ПНат' which are OCR noise.
    """
    alphas = [c for c in w if c.isalpha()]
    if len(alphas) <= 2:
        return True
    uppers = sum(1 for c in alphas if c.isupper())
    lowers = sum(1 for c in alphas if c.islower())
    if uppers == len(alphas) or lowers == len(alphas):
        return True  # all-caps or all-lower
    if alphas[0].isupper() and lowers == len(alphas) - 1:
        return True  # Title-case
    return False


def _merge_ocr_candidates(
    tess_names: "list[str]",
    lighton_names: "list[str]",
) -> "list[str]":
    """Union of Tesseract and LightOnOCR heading candidates.

    For each Tesseract candidate, search for a sufficiently similar LightOnOCR
    candidate using SequenceMatcher ratio >= 0.85 AND a shared anchor word
    (from the tail of the heading, skipping the first word which is typically
    a common prefix like 'Chapter' / 'Глава').  When a match is found, the
    LightOnOCR text is used (cleaner OCR quality).  Unmatched Tesseract
    candidates are kept as-is.  LightOnOCR candidates not matched to any
    Tesseract entry are appended at the end.

    Returns a deduplicated list (case-insensitive) preserving order.
    """
    import difflib

    def _share_anchor(a: str, b: str, min_alpha: int = 4) -> bool:
        # Skip first word — it is the common prefix ('Глава', 'Chapter', …)
        wa = {w.lower() for w in a.split()[1:] if sum(1 for c in w if c.isalpha()) >= min_alpha}
        wb = {w.lower() for w in b.split()[1:] if sum(1 for c in w if c.isalpha()) >= min_alpha}
        return bool(wa & wb)

    result: list[str] = []
    used_lighton: set[int] = set()

    for tname in tess_names:
        best_ratio, best_idx = 0.0, -1
        for i, lname in enumerate(lighton_names):
            if i in used_lighton:
                continue
            ratio = difflib.SequenceMatcher(None, tname.lower(), lname.lower()).ratio()
            if ratio > best_ratio and _share_anchor(tname, lname):
                best_ratio, best_idx = ratio, i
        if best_ratio >= 0.85 and best_idx >= 0:
            result.append(lighton_names[best_idx])
            used_lighton.add(best_idx)
        else:
            result.append(tname)

    for i, lname in enumerate(lighton_names):
        if i not in used_lighton:
            result.append(lname)

    # Deduplicate preserving order (case-insensitive)
    seen: set[str] = set()
    deduped: list[str] = []
    for name in result:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped


_tesseract_proc = None
_tesseract_lock = threading.Lock()


def _tesseract_heading_ocr(crop_bgr: "np.ndarray", tess_lang: str) -> str:
    """OCR a heading crop using Tesseract PSM 7 (single text line).

    Handles decorative letter-spaced fonts that confuse neural OCR models.
    Adds 20 px white border before OCR to avoid edge clipping.

    Uses a persistent Tesseract process (stdin/stdout pipe) to avoid
    ~0.3-1s model-load overhead per call.

    Returns stripped text, or "" if tesseract is not available or fails.
    """
    global _tesseract_proc
    import shutil

    tess_bin = shutil.which("tesseract")
    if tess_bin is None:
        return ""
    try:
        border = 20
        padded = cv2.copyMakeBorder(
            crop_bgr, border, border, border, border,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        # Encode to PNG in-memory
        success, buf = cv2.imencode(".png", padded)
        if not success:
            return ""

        with _tesseract_lock:
            # Start or reuse persistent tesseract process
            if _tesseract_proc is None or _tesseract_proc.poll() is not None:
                _tesseract_proc = subprocess.Popen(
                    [tess_bin, "stdin", "stdout", "-l", tess_lang, "--psm", "7", "--oem", "3"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
            proc = _tesseract_proc

        # Send image via stdin, read result from stdout.
        # Tesseract reads one image per stdin invocation when using "stdin".
        # For persistent use, we use a fresh process per call but avoid temp files.
        # Actually tesseract stdin mode reads one image and exits, so we need a
        # new process per call but skip the temp-file I/O.
        proc = subprocess.run(
            [tess_bin, "stdin", "stdout", "-l", tess_lang, "--psm", "7", "--oem", "3"],
            input=buf.tobytes(), capture_output=True, timeout=30,
        )
        return proc.stdout.decode("utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning(f"_tesseract_heading_ocr: {e}")
        return ""


def _split_at_blank_gaps(
    crop_bgr: np.ndarray,
    std_thresh: float = 5.0,
    min_seg_h: int = 10,
) -> list[np.ndarray]:
    """Split a crop into sub-crops at rows with near-zero variance (blank gaps).

    This isolates heading lines (which contain dot leaders that overflow the
    OCR token budget) from description paragraphs so each can be OCR'd
    independently.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blank = np.array([gray[y].std() < std_thresh for y in range(gray.shape[0])])
    segments: list[tuple[int, int]] = []
    in_blank = bool(blank[0])
    seg_start = 0
    for y in range(1, len(blank)):
        if bool(blank[y]) != in_blank:
            if not in_blank:
                segments.append((seg_start, y))
            seg_start = y
            in_blank = bool(blank[y])
    if not in_blank:
        segments.append((seg_start, len(blank)))
    return [crop_bgr[s:e] for s, e in segments if e - s >= min_seg_h]


def _ocr_toc_page(source_path: Path, page_num: int) -> str:
    """OCR a TOC page by splitting at blank gaps and concatenating results.

    Uses row-variance based splitting to isolate heading lines (which contain
    dot leaders) from description paragraphs, preventing token overflow.

    Args:
        source_path: Path to the source PDF/DjVu file.
        page_num: 1-based page number.

    Returns:
        Concatenated plain text from all segments.
    """
    from ocr_reflow.document_loader import load_page
    from ocr_reflow.ocr_export_layout import (
        _get_lightonocr, _resize_for_ocr, _bgr_to_pil,
    )
    import torch

    img = load_page(str(source_path), page_num - 1)  # 0-based
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (1024, int(h * 1024 / w)))

    # Split the full page at blank gaps first, then further split tall segments
    from ocr_reflow.ocr_export_layout import _split_plain_text_crop
    coarse_crops = _split_plain_text_crop(img_resized)

    # For each coarse crop, split again at blank gaps to isolate dot-leader lines
    all_segs: list[np.ndarray] = []
    for crop in coarse_crops:
        segs = _split_at_blank_gaps(crop)
        if segs:
            all_segs.extend(segs)
        else:
            all_segs.append(crop)

    processor, model = _get_lightonocr()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    MIN_H = 40  # pad segments shorter than this to avoid model errors

    texts: list[str] = []
    for seg in all_segs:
        if seg.shape[0] < MIN_H:
            pad = np.full((MIN_H - seg.shape[0], seg.shape[1], 3), 255, dtype=np.uint8)
            seg = np.vstack([seg, pad])
        pil = _resize_for_ocr(_bgr_to_pil(seg))
        conv = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": "Transcribe all text from this page verbatim. Line break only at paragraph boundaries. Preserve mathematical formulas in LaTeX notation."},
        ]}]
        text_prompt = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=[text_prompt], images=[pil], padding=True, return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            ids = model.generate(
                **inputs, max_new_tokens=512,
                do_sample=True, temperature=0.2, top_p=0.9, top_k=0, repetition_penalty=1.1,
            )
        input_len = inputs["input_ids"].shape[1]
        gen = ids[0, input_len:]
        text = processor.decode(gen, skip_special_tokens=True).strip()
        if text:
            texts.append(text)

    return "\n".join(texts)



def _live_heading_toc(
    page_results: list[dict],
    source_path: "Path | None" = None,
    tess_lang: str | None = None,
) -> list[dict]:
    """Build TOC from all title blocks detected across pages.

    Scans every page for <h1> fragments (title blocks), validates them,
    and filters out known non-chapter entries (Оглавление, Серия...).

    Used when --toc-pages is not specified.
    """
    _SKIP_TITLES = {"оглавление", "содержание"}
    entries: list[dict] = []
    seen: set[str] = set()
    for r in page_results:
        for frag in r.get("html_fragments", []):
            if "<h1>" not in frag:
                continue
            raw = _strip_html(frag).strip()
            if not raw:
                continue
            normalized = _normalize_heading(raw)
            if _is_valid_heading(normalized):
                if normalized.lower().strip().rstrip(".") in _SKIP_TITLES:
                    continue
                if normalized.startswith("Серия") or normalized.startswith("серия"):
                    continue
                key = normalized[:80].lower()
                if key not in seen:
                    seen.add(key)
                    entries.append({"page": r["page_num"], "text": normalized[:80]})
            elif source_path is not None and tess_lang:
                text = _reocr_title_crop(source_path, r["page_num"], tess_lang)
                if text:
                    normalized = _normalize_heading(text)
                    if _is_valid_heading(normalized):
                        key = normalized[:80].lower()
                        if key not in seen:
                            seen.add(key)
                            entries.append({"page": r["page_num"], "text": normalized[:80]})
    return entries


def _reocr_title_crop(source_path: "Path", page_num: int, tess_lang: str) -> str:
    """Load a page, find the title block via layout detection, OCR with Tesseract PSM 7."""
    try:
        from ocr_reflow.document_loader import load_page
        img_bgr = load_page(str(source_path), page_num - 1)
        layout_from_array = _get_layout()
        blocks = layout_from_array(img_bgr)
        _crop_fn, _, _, _ = _get_ocr_helpers()
        for geom, label in blocks:
            if label in _TITLE_LABELS:
                crop = _crop_fn(img_bgr, geom)
                if crop.size > 0:
                    return _tesseract_heading_ocr(crop, tess_lang)
    except Exception as e:
        logger.warning(f"_reocr_title_crop page {page_num}: {e}")
    return ""


def _extract_toc_from_pages(
    page_results: list[dict],
    toc_page_nums: list[int],
    source_path: "Path | None" = None,
    tess_lang: str | None = None,
) -> list[dict]:
    """Parse printed TOC pages and match chapter headings to content pages.

    Two-pass strategy:
    Pass 1 — html_fragments on TOC pages:
        Fragments that are short (< 40 chars), contain no em-dash, and pass
        _is_valid_heading are chapter names.  Following fragments are the
        description snippet.  This reliably extracts chapters whose names the
        layout model detected as separate blocks.

    Pass 2 — Tesseract re-OCR of heading segments:
        For chapters whose names were merged into description blocks by the
        layout model (e.g. chapters 4-8 on page 393 of archimedes1.djvu),
        re-OCR each TOC page using blank-gap splitting to isolate dot-leader
        heading lines, then run Tesseract PSM 7 on those short segments.
        Description snippets for these chapters come from Pass 1 fragments.

    Merge: union of Pass 1 and Pass 2 names (Pass 1 takes priority for
    chapters found in both).  For each chapter, use the best available
    snippet to search non-TOC body pages.

    Returns list of {page: int, text: str} sorted by page number.
    """
    toc_set = set(toc_page_nums)
    EM_DASH = "\u2014"

    # ------------------------------------------------------------------ #
    # Pass 1: parse html_fragments already extracted by the layout model  #
    # ------------------------------------------------------------------ #
    # Collect all fragments from TOC pages in order.
    # A fragment is a heading candidate if:
    #   - stripped text length < 40
    #   - no em-dash (—) in text  (descriptions always have em-dashes)
    #   - passes _is_valid_heading
    # Everything else following a heading candidate is its description.

    p1_chapters: list[tuple[str, str]] = []  # (name, snippet)
    pending_name: str | None = None
    pending_snippets: list[str] = []

    for pn in sorted(toc_page_nums):
        r_map = {r["page_num"]: r for r in page_results}
        r = r_map.get(pn)
        if r is None:
            continue
        for frag in r["html_fragments"]:
            text = _strip_html(frag).strip()
            if not text:
                continue
            short = len(text) < 40
            has_dash = EM_DASH in text
            if short and not has_dash and _is_valid_heading(text):
                if pending_name is not None:
                    p1_chapters.append((pending_name, " ".join(pending_snippets)))
                pending_name = text
                pending_snippets = []
            elif pending_name is not None:
                pending_snippets.append(text)

    if pending_name is not None:
        p1_chapters.append((pending_name, " ".join(pending_snippets)))

    # Track names that appeared as heading candidates but had no description
    # (section dividers like "ЧАСТЬ ПЕРВАЯ") — exclude from Pass 2 to prevent
    # them from stealing description snippets that belong to real chapters.
    p1_dividers: set[str] = {n.lower() for n, s in p1_chapters if not s.strip()}

    # Drop entries with no snippet (section dividers without a description)
    p1_chapters = [(n, s) for n, s in p1_chapters if s.strip()]
    logger.info(f"_extract_toc_from_pages Pass 1: {len(p1_chapters)} chapters from fragments")

    # ------------------------------------------------------------------ #
    # Pass 2: Tesseract PSM 6 full-page re-OCR + LightOnOCR block-title  #
    # ------------------------------------------------------------------ #
    # Collect heading candidates from two sources:
    #   A) Tesseract PSM 6 on each TOC page image — lines ending with a
    #      page number, leading alpha words extracted as heading name.
    #      Words with mixed internal case (OCR garbage) are rejected via
    #      _is_clean_word.
    #   B) LightOnOCR html_fragments with class="block title" (CSS class
    #      produced by _TITLE_LABELS) that have >= 2 words and pass
    #      _is_valid_heading.
    # The two lists are merged via _merge_ocr_candidates: fuzzy-match
    # (ratio >= 0.85 + shared anchor word) prefers LightOnOCR text;
    # unmatched Tesseract entries are kept as-is.
    # Page numbers from OCR are NOT used — snippet matching finds body pages.
    # Description snippets are NOT taken from re-OCR — we reuse Pass 1
    # fragment text which is cleaner.

    tess_names: list[str] = []   # heading names from Tesseract
    lighton_names: list[str] = []  # heading names from LightOnOCR block-title frags

    _RE_ENDS_WITH_NUM = re.compile(r"^(.*?)\s+(\d{1,4})\s*$")

    # --- 2A: LightOnOCR block-title candidates ---
    r_map_pass2 = {r["page_num"]: r for r in page_results}
    for pn in sorted(toc_page_nums):
        r = r_map_pass2.get(pn)
        if r is None:
            continue
        for frag in r["html_fragments"]:
            # Only fragments whose outer element has class="block title"
            if 'class="block title"' not in frag:
                continue
            text = _strip_html(frag).strip()
            if not text:
                continue
            if len(text.split()) >= 2 and len(text) < 40 and _is_valid_heading(text) \
                    and text.lower() not in p1_dividers:
                lighton_names.append(text)

    logger.info(
        f"_extract_toc_from_pages Pass 2 LightOnOCR: {len(lighton_names)} block-title candidates"
    )

    # --- 2B: Tesseract candidates ---
    if source_path is not None and tess_lang:
        from ocr_reflow.document_loader import load_page
        import subprocess, tempfile

        for pn in sorted(toc_page_nums):
            try:
                img = load_page(str(source_path), pn - 1)
                h, w = img.shape[:2]
                img_r = cv2.resize(img, (1024, int(h * 1024 / w)))
                border = 30
                padded = cv2.copyMakeBorder(
                    img_r, border, border, border, border,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255),
                )
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    tmp_path = tf.name
                cv2.imwrite(tmp_path, padded)
                try:
                    tess_bin = shutil.which("tesseract") or "tesseract"
                    result = subprocess.run(
                        [tess_bin, tmp_path, "stdout", "-l", tess_lang,
                         "--psm", "6", "--oem", "3"],
                        capture_output=True, text=True, timeout=60,
                    )
                    ocr_text = result.stdout
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                for line in ocr_text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    m = _RE_ENDS_WITH_NUM.match(line)
                    if not m:
                        continue
                    before_num = m.group(1)
                    # Extract heading prefix: leading words that are
                    # mostly alphabetic (no digits, alpha ratio >= 0.7)
                    words = before_num.split()
                    start = 0
                    while start < len(words) and not any(
                        ch.isalpha() for ch in words[start]
                    ):
                        start += 1
                    head_words: list[str] = []
                    for w in words[start:]:
                        alpha = sum(1 for ch in w if ch.isalpha())
                        has_digit = any(ch.isdigit() for ch in w)
                        if alpha == 0 or has_digit:
                            break
                        if alpha / len(w) < 0.7 and head_words:
                            break
                        if not _is_clean_word(w):
                            break  # mixed-case OCR garbage — stop here
                        head_words.append(w)
                    name = " ".join(head_words)
                    name = re.sub(r"[\s.]+$", "", name).strip()
                    name = _normalize_heading(name)
                    if (
                        len(name.split()) >= 2
                        and len(name) < 40
                        and _is_valid_heading(name)
                    ):
                        tess_names.append(name)

            except Exception as e:
                logger.warning(f"_extract_toc_from_pages Pass 2 page {pn}: {e}")

        logger.info(f"_extract_toc_from_pages Pass 2 Tesseract: {len(tess_names)} headings")

    p2_names = _merge_ocr_candidates(tess_names, lighton_names)
    logger.info(f"_extract_toc_from_pages Pass 2 merged: {len(p2_names)} headings")

    # Filter Pass 1: drop entries whose name is not confirmed by Pass 2.
    # This removes section dividers (e.g. "ЧАСТЬ ТРЕТЬЯ") that Pass 1
    # mistakenly paired with a description belonging to a real chapter.
    # Only apply the filter when Pass 2 found at least some headings —
    # if Pass 2 is empty (no Tesseract / no block-title frags) we keep
    # all Pass 1 entries to avoid losing everything.
    if p2_names:
        p2_name_set_norm = {_normalize_heading(n).lower() for n in p2_names}
        p1_chapters_filtered = [
            (n, s) for n, s in p1_chapters
            if _normalize_heading(n).lower() in p2_name_set_norm
        ]
        dropped = len(p1_chapters) - len(p1_chapters_filtered)
        if dropped:
            logger.info(
                f"_extract_toc_from_pages: dropped {dropped} Pass 1 entries "
                f"not confirmed by Pass 2 (likely section dividers)"
            )
        p1_chapters = p1_chapters_filtered

    # ------------------------------------------------------------------ #
    # Merge Pass 1 and Pass 2                                             #
    # ------------------------------------------------------------------ #
    # Pass 1 entries take priority (they have clean snippets).
    # Pass 2 names not already in Pass 1 are added; their snippets are
    # looked up from Pass 1 fragment text by searching for the description
    # that follows the chapter in TOC page order.

    # Build a snippet lookup from Pass 1 fragment scan (all descriptions
    # in order, keyed by position).  For Pass 2 names we need to find
    # which description block follows them in the TOC page fragments.
    # Simplest approach: collect all description texts from TOC pages in
    # order, then pair them with Pass 2 names by position.

    p1_name_set = {n.lower() for n, _ in p1_chapters}

    # Collect all description fragments from TOC pages in order
    all_desc_texts: list[str] = []
    r_map = {r["page_num"]: r for r in page_results}
    for pn in sorted(toc_page_nums):
        r = r_map.get(pn)
        if r is None:
            continue
        for frag in r["html_fragments"]:
            text = _strip_html(frag).strip()
            if not text:
                continue
            short = len(text) < 40
            has_dash = EM_DASH in text
            # Description: long or has em-dash
            if not short or has_dash:
                all_desc_texts.append(text)

    # For each Pass 2 name not in Pass 1, find its snippet by searching
    # all_desc_texts for text that appears on the corresponding body page.
    non_toc_results = [r for r in page_results if r["page_num"] not in toc_set]

    import re as _re

    def _edit1(a: str, b: str) -> bool:
        """True if a == b or they differ by exactly one edit (sub/ins/del)."""
        if a == b:
            return True
        if abs(len(a) - len(b)) > 1:
            return False
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) == 1
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        for i in range(len(long_)):
            if long_[:i] + long_[i+1:] == short:
                return True
        return False

    def _find_page_for_snippet(snippet: str) -> int | None:
        if not snippet:
            return None
        # Pass A: exact substring match with progressively shorter prefixes
        for length in (50, 35, 20, 15):
            key = snippet[:length].lower()
            if len(key) < 10:
                break
            for r in non_toc_results:
                for frag in r["html_fragments"]:
                    if key in _strip_html(frag).lower():
                        return r["page_num"]
        # Pass B: word-overlap fallback — tolerates single-char OCR errors.
        # Use only the FIRST 4 alpha words (>= 3 chars) from the snippet —
        # these are the chapter's opening sentence, which is distinctive.
        # Sliding windows over the full snippet would match generic historical
        # phrases (e.g. "веке нашей эры XVII") on unrelated pages.
        # Each word is matched either exactly or with edit distance <= 1
        # (handles single-char substitutions like "Терентий" vs "Теренций").

        def _find_word_fuzzy(word: str, text: str, start: int) -> int:
            """Find word in text starting at start, allowing edit distance 1.
            Returns end position of match, or -1."""
            # Try exact first (fast path)
            idx = text.find(word, start)
            if idx != -1:
                return idx + len(word)
            # Try each word-token in text from start position
            for m in _re.finditer(r"[а-яёА-ЯЁa-zA-Z]+", text[start:]):
                tok = m.group()
                if _edit1(word, tok):
                    return start + m.end()
            return -1

        snip_words = [
            w.lower()
            for w in _re.findall(r"[а-яёА-ЯЁa-zA-Z]+", snippet)
            if len(w) >= 3
        ]
        if len(snip_words) >= 4:
            window = snip_words[:4]
            for r in non_toc_results:
                for frag in r["html_fragments"]:
                    frag_text = _strip_html(frag).lower()
                    pos = 0
                    ok = True
                    for w in window:
                        end = _find_word_fuzzy(w, frag_text, pos)
                        if end == -1:
                            ok = False
                            break
                        pos = end
                    if ok:
                        return r["page_num"]
        return None

    def _find_page_by_title(name: str) -> int | None:
        """Find a body page whose content fuzzy-matches *name*.

        Two-pass strategy:
          Pass A — only <h1> title blocks (high precision).
          Pass B — if Pass A fails, search the first fragment on each page,
                   since chapter titles always appear at the page top.
        Used when snippet matching fails (e.g. TOC has dot-leader format
        with no separate description text per chapter)."""
        def _norm(s: str) -> str:
            return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", "", s.lower())).strip()
        name_words = [w for w in _norm(name).split() if len(w) >= 3]
        if not name_words:
            return None

        def _all_words_match(frag_text: str) -> bool:
            words = _norm(frag_text).split()
            return bool(words) and all(
                any(_edit1(nw, tw) for tw in words) for nw in name_words
            )

        # Pass A: high-precision <h1> title blocks
        for r in non_toc_results:
            for frag in r["html_fragments"]:
                if 'class="block title"' in frag and _all_words_match(_strip_html(frag)):
                    return r["page_num"]

        # Pass B: search the first fragment on each page
        for r in non_toc_results:
            frags = r["html_fragments"]
            if frags and _all_words_match(_strip_html(frags[0])):
                return r["page_num"]

        return None

    # Build snippet map for Pass 2 names: try each description text
    p2_with_snippets: list[tuple[str, str]] = []
    # Track pages already claimed by Pass 1 or earlier Pass 2 assignments
    claimed_pages: set[int] = set()
    for _, s in p1_chapters:
        pg = _find_page_for_snippet(s)
        if pg is not None:
            claimed_pages.add(pg)

    for name in p2_names:
        if name.lower() in p1_name_set:
            continue  # already in Pass 1
        # Find the first unclaimed description from all_desc_texts
        best_snip = ""
        for desc in all_desc_texts:
            pg = _find_page_for_snippet(desc)
            if pg is not None and pg not in claimed_pages:
                best_snip = desc
                claimed_pages.add(pg)
                break
        p2_with_snippets.append((name, best_snip))
        if best_snip:
            p1_name_set.add(name.lower())  # prevent duplicate names

    merged = p1_chapters + p2_with_snippets

    if not merged:
        logger.warning("_extract_toc_from_pages: no chapter headings found")
        return []

    logger.info(f"_extract_toc_from_pages: {len(merged)} chapters after merge")

    # ------------------------------------------------------------------ #
    # Match each chapter to a body page via snippet search                #
    # ------------------------------------------------------------------ #
    entries: list[dict] = []
    for name, snippet in merged:
        page = _find_page_for_snippet(snippet)
        if page is None:
            page = _find_page_by_title(name)
        if page is None:
            logger.warning(f"Chapter '{name[:40]}': snippet not matched, skipping")
            continue
        entries.append({"page": page, "text": name[:80]})

    # Sort by page number and deduplicate
    seen: set[int] = set()
    result = []
    for e in sorted(entries, key=lambda x: x["page"]):
        if e["page"] not in seen:
            seen.add(e["page"])
            result.append(e)

    # ------------------------------------------------------------------ #
    # Unclaimed-description pass: for TOC descriptions that matched a    #
    # body page but weren't paired with any chapter name, re-OCR that    #
    # body page to extract the missing chapter title.                    #
    # ------------------------------------------------------------------ #
    if source_path is not None and tess_lang:
        import subprocess, tempfile

        tess_bin = shutil.which("tesseract") or "tesseract"

        def _reocr_body_title(page_num: int) -> str | None:
            """Re-OCR a body page with PSM 6 to find a chapter title."""
            try:
                from ocr_reflow.document_loader import load_page
            except ImportError:
                from document_loader import load_page
            try:
                img = load_page(str(source_path), page_num - 1)
                h, w = img.shape[:2]
                img_r = cv2.resize(img, (1024, int(h * 1024 / w)))
                border = 20
                padded = cv2.copyMakeBorder(
                    img_r, border, border, border, border,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255),
                )
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    tmp_path = tf.name
                cv2.imwrite(tmp_path, padded)
                try:
                    r = subprocess.run(
                        [tess_bin, tmp_path, "stdout", "-l", tess_lang,
                         "--psm", "6", "--oem", "3"],
                        capture_output=True, text=True, timeout=60,
                    )
                    for line in r.stdout.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        words = line.split()
                        start = 0
                        while start < len(words) and not any(
                            ch.isalpha() for ch in words[start]
                        ):
                            start += 1
                        head_words = []
                        for w in words[start:]:
                            alpha = sum(1 for ch in w if ch.isalpha())
                            has_digit = any(ch.isdigit() for ch in w)
                            if alpha == 0 or has_digit:
                                break
                            if alpha / len(w) < 0.7 and head_words:
                                break
                            if not _is_clean_word(w):
                                break
                            head_words.append(w)
                        name = " ".join(head_words)
                        name = re.sub(r"[\s.]+$", "", name).strip()
                        name = _normalize_heading(name)
                        if (
                            len(name.split()) >= 2
                            and len(name) < 40
                            and _is_valid_heading(name)
                        ):
                            return name
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            except Exception as e:
                logger.debug(f"_reocr_body_title page {page_num}: {e}")
            return None

        extra_entries: list[dict] = []
        for desc in all_desc_texts:
            pg = _find_page_for_snippet(desc)
            if pg is None or pg in seen:
                continue
            # This description matched a body page not yet in TOC
            title = _reocr_body_title(pg)
            if title:
                logger.info(f"Unclaimed-desc: found chapter '{title}' at p{pg}")
                extra_entries.append({"page": pg, "text": title})
                seen.add(pg)

        if extra_entries:
            result = sorted(result + extra_entries, key=lambda x: x["page"])

    return result


def _apply_spellcheck(page_results: list[dict], lang: str) -> int:
    """Advanced hunspell correction pass using book vocab + trigram LM.

    Imports logic from fix_epub_spelling.py.  Returns number of corrections.
    """
    from ocr_reflow.fix_epub_spelling import (
        _TrigramLM, _corrections_for_text, _plain_text, fix_spelling_in_html,
    )
    from collections import Counter
    import re

    # Collect all plain text from HTML fragments
    all_plain = ""
    for r in page_results:
        for frag in r["html_fragments"]:
            all_plain += _plain_text(frag) + "\n"

    if not all_plain.strip():
        return 0

    # Build vocabulary from the book text
    vocab: Counter = Counter()
    for w in re.findall(r"[а-яёА-ЯЁa-zA-Z]+", all_plain):
        vocab[w.lower()] += 1

    logger.info("Advanced spellcheck: %d unique words in vocabulary", len(vocab))

    lm = _TrigramLM(all_plain)
    corrections = _corrections_for_text(all_plain, lang, vocab, lm=lm)

    if not corrections:
        logger.info("Advanced spellcheck: no corrections found")
        return 0

    logger.info("Advanced spellcheck: %d corrections", len(corrections))

    # Apply to HTML fragments
    for r in page_results:
        for i, frag in enumerate(r["html_fragments"]):
            fixed = fix_spelling_in_html(frag, corrections)
            if fixed != frag:
                r["html_fragments"][i] = fixed

    return len(corrections)


def build_epub(
    page_results: list[dict],
    output_path: Path,
    title: str,
    author: str,
    toc_page_nums: list[int] | None = None,
    source_path: Path | None = None,
    tess_lang: str | None = None,
):
    """Assemble all page results into an EPUB 3 file."""
    uid = str(uuid.uuid4())

    # Collect all image names across pages
    all_image_names = []
    for r in page_results:
        for img in r["images"]:
            all_image_names.append(img["name"])

    # Build TOC entries
    if toc_page_nums:
        toc_entries = _extract_toc_from_pages(
            page_results, toc_page_nums,
            source_path=source_path, tess_lang=tess_lang,
        )
        if not toc_entries:
            logger.warning("Printed TOC extraction failed, falling back to live heading detection")
            toc_entries = _live_heading_toc(page_results, source_path, tess_lang)
        else:
            live_entries = _live_heading_toc(page_results, source_path, tess_lang)
            if len(live_entries) > len(toc_entries) * 1.5:
                logger.warning(
                    f"Printed TOC ({len(toc_entries)} entries) significantly smaller "
                    f"than live detection ({len(live_entries)}); using live headings"
                )
                toc_entries = live_entries
    else:
        toc_entries = _live_heading_toc(page_results, source_path, tess_lang)

    if toc_entries:
        # Filter out entries that point to TOC pages themselves
        if toc_page_nums:
            toc_set = set(toc_page_nums)
            before = len(toc_entries)
            toc_entries = [e for e in toc_entries if e["page"] not in toc_set]
            if len(toc_entries) < before:
                logger.info(f"Removed {before - len(toc_entries)} TOC entries pointing to TOC pages")
        logger.info("TOC: %d entries", len(toc_entries))
        for e in toc_entries:
            logger.info("  p%4d: %s", e['page'], e['text'][:60])
    else:
        logger.info("TOC: no entries found")

    content_xhtml = _build_content_xhtml(title, page_results)
    has_svg = "<svg" in content_xhtml
    opf = _build_opf(title, author, uid, all_image_names, has_svg=has_svg)
    nav = _build_nav(title, toc_entries)
    ncx = _build_ncx(title, uid, toc_entries)

    with zipfile.ZipFile(output_path, "w") as zf:
        # mimetype must be first and uncompressed
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
        )
        # Text files: compress with DEFLATE
        zf.writestr("META-INF/container.xml", _CONTAINER_XML, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/content.opf", opf, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/nav.xhtml", nav, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/toc.ncx", ncx, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/style.css", _EPUB_CSS, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/content.xhtml", content_xhtml, compress_type=zipfile.ZIP_DEFLATED)

        # Images are already compressed (PNG/JPEG) — store without recompression.
        for r in page_results:
            for img in r["images"]:
                img_bytes = base64.b64decode(img["data_b64"])
                zf.writestr(f"OEBPS/images/{img['name']}", img_bytes, compress_type=zipfile.ZIP_STORED)

    logger.info("EPUB written: %s", output_path)


# ---------------------------------------------------------------------------
# Page count helpers
# ---------------------------------------------------------------------------

def _get_page_count(filepath: str) -> int:
    """Get total page count for a PDF or DjVu file."""
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        doc = fitz.open(str(path))
        n = len(doc)
        doc.close()
        return n
    elif suffix == ".djvu":
        import djvu.decode
        try:
            from ocr_reflow.document_loader import _DjVuContext
        except ImportError:
            from document_loader import _DjVuContext
        ctx = _DjVuContext.get()
        doc = ctx.new_document(djvu.decode.FileURI(str(path)))
        doc.decoding_job.wait()
        n = len(doc.pages)
        del doc
        return n
    else:
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: convert a PDF/DjVu to EPUB with OCR and math pre-rendering."""
    parser = argparse.ArgumentParser(
        description="Export a PDF or DjVu file to EPUB with OCR and pre-rendered math SVG."
    )
    parser.add_argument("input", help="Path to PDF or DjVu file")
    parser.add_argument("-o", "--output", help="Output EPUB path (default: <input_stem>.epub)")
    parser.add_argument(
        "--pages",
        help="Page range, 1-based inclusive, e.g. 1-20 (default: all)",
    )
    parser.add_argument("--title", help="Book title (default: filename stem)")
    parser.add_argument("--author", default="", help="Author name")
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory for resumable page checkpoints (default: /tmp/<stem>_epub_cache)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint (skip already-done pages)",
    )
    parser.add_argument(
        "--mathjax",
        help="Path to tex-svg-full.js (default: auto-detect from this file's location)",
    )
    parser.add_argument(
        "--toc-pages",
        help="Comma-separated page numbers of the printed TOC (e.g. '330,331'). "
             "Used to extract chapter names and match them to content pages.",
    )
    parser.add_argument(
        "--lang",
        help="Language(s) for spell check, e.g. 'ru', 'en', 'ru,en'. "
             "Short aliases (ru, en, de, fr, es) are resolved to hunspell dict names. "
             "If omitted, spell check is skipped.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Minimum DPI for DJVU page rendering. Higher values improve OCR quality "
             "for low-resolution sources (default: 300, range: 150-600).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (logs written to stderr by default).",
    )
    args = parser.parse_args()
    from ocr_reflow.log_setup import setup_logging
    setup_logging(log_path=args.log_file)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    stem = input_path.stem
    output_path = Path(args.output).resolve() if args.output else input_path.with_suffix(".epub")
    title = args.title or stem
    author = args.author

    # Parse --toc-pages
    toc_page_nums: list[int] | None = None
    if args.toc_pages:
        try:
            toc_page_nums = [int(p.strip()) for p in args.toc_pages.split(",") if p.strip()]
        except ValueError:
            logger.error("--toc-pages must be comma-separated integers, got '%s'", args.toc_pages)
            sys.exit(1)
        logger.info("Printed TOC pages: %s", toc_page_nums)

    # Parse and validate --lang
    spellcheck_lang: str | None = None
    if args.lang:
        spellcheck_lang = _resolve_langs(args.lang)
        logger.info("Spell check language(s): %s", spellcheck_lang)
        # Validate that hunspell can open the requested dictionaries
        import subprocess as _sp
        try:
            result = _sp.run(
                ["hunspell", "-d", spellcheck_lang, "-l"],
                input="",
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode not in (0, 1):  # hunspell returns 1 when words are misspelled
                raise RuntimeError(result.stderr.strip())
        except FileNotFoundError:
            logger.error("hunspell not found. Install it (e.g. pacman -S hunspell).")
            sys.exit(1)
        except Exception as e:
            logger.error("hunspell failed for lang '%s': %s", spellcheck_lang, e)
            logger.error("  Install the dictionary, e.g.: pacman -S hunspell-%s",
                         spellcheck_lang.split(',')[0].lower().replace('_', '-'))
            sys.exit(1)

    # Checkpoint dir
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(f"/tmp/{stem}_epub_cache")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # MathJax path
    if args.mathjax:
        mathjax_path = Path(args.mathjax)
    else:
        # Auto-detect: look relative to this file
        here = Path(__file__).parent
        mathjax_path = here / "static" / "mathjax" / "tex-svg-full.js"
        if not mathjax_path.exists():
            logger.warning("MathJax not found at %s. Math will not be pre-rendered.", mathjax_path)
            mathjax_path = None

    # Page range
    total_pages = _get_page_count(str(input_path))
    if args.pages:
        m = re.match(r'^(\d+)(?:-(\d+))?$', args.pages)
        if not m:
            logger.error("invalid --pages format '%s', expected N or N-M", args.pages)
            sys.exit(1)
        start_page = int(m.group(1))
        end_page = int(m.group(2)) if m.group(2) else start_page
    else:
        start_page = 1
        end_page = total_pages

    start_page = max(1, start_page)
    end_page = min(total_pages, end_page)
    page_nums = list(range(start_page, end_page + 1))

    # Auto-detect language if --lang was not given
    if not args.lang and total_pages >= 3:
        from ocr_reflow.language_detection import detect as _detect_lang
        detected = _detect_lang(str(input_path), total_pages)
        spellcheck_lang = _resolve_langs(detected)
        logger.info("Spell check language: %s (auto-detected)", spellcheck_lang)
        args.lang = detected
        import subprocess as _sp
        try:
            result = _sp.run(
                ["hunspell", "-d", spellcheck_lang, "-l"],
                input="", capture_output=True, text=True, timeout=10,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr.strip())
        except FileNotFoundError:
            logger.warning("hunspell not found, spell check disabled.")
            spellcheck_lang = None
        except Exception as e:
            logger.warning("hunspell failed for '%s': %s", spellcheck_lang, e)
            spellcheck_lang = None

    logger.info("Exporting '%s' pages %d-%d (%d pages) → %s",
                input_path.name, start_page, end_page, len(page_nums), output_path)
    logger.info("Checkpoint dir: %s", checkpoint_dir)

    # Validate or record the source file in the checkpoint dir
    source_marker = checkpoint_dir / "source.txt"
    input_abs = str(Path(args.input).resolve())
    if args.resume and source_marker.exists():
        recorded = source_marker.read_text().strip()
        if recorded != input_abs:
            logger.error(
                "checkpoint dir was created for a different file:\n"
                "  recorded: %s\n"
                "  current:  %s\n"
                "Use a different --checkpoint-dir or omit --resume to start fresh.",
                recorded, input_abs,
            )
            sys.exit(1)
    if not source_marker.exists():
        source_marker.write_text(input_abs)

    # Clear checkpoints if not resuming
    if not args.resume:
        for f in checkpoint_dir.glob("page_*.json"):
            f.unlink()
        source_marker.write_text(input_abs)

    try:
        from ocr_reflow.document_loader import load_page
    except ImportError:
        from document_loader import load_page

    with MathRenderer(mathjax_path) if mathjax_path else _NullRenderer() as renderer:
        page_results = []
        # Pipeline: prefetch page N+1's image+layout on a background thread
        # while page N's VLM generate runs.  The GPU YOLO layout model and
        # the VLM share the same CUDA context, so layout runs on CPU thread
        # while VLM runs on GPU — they overlap naturally.
        from concurrent.futures import ThreadPoolExecutor

        _prefetch_executor = ThreadPoolExecutor(max_workers=1)
        _prefetch_future = None

        def _prefetch_page(pn):
            """Load image and run layout for page pn (returns img_bgr)."""
            try:
                return load_page(str(input_path), pn - 1, min_dpi=args.dpi)
            except Exception as e:
                logger.warning("Could not load page %d: %s", pn, e)
                return None

        for idx, pn in enumerate(page_nums, 1):
            logger.info("[%d/%d] Page %d/%d", idx, len(page_nums), pn, total_pages)
            # Skip load_page entirely if checkpoint already exists
            checkpoint_file = checkpoint_dir / f"page_{pn:04d}.json"
            if checkpoint_file.exists():
                img_bgr = None
            else:
                # Use prefetched image if available, otherwise load directly
                if _prefetch_future is not None:
                    img_bgr = _prefetch_future.result()
                    _prefetch_future = None
                else:
                    try:
                        img_bgr = load_page(str(input_path), pn - 1, min_dpi=args.dpi)
                    except Exception as e:
                        logger.warning("Could not load page %d: %s", pn, e)
                        continue

            # Start prefetching the next page while we process this one
            if idx < len(page_nums):
                next_pn = page_nums[idx]
                next_ckpt = checkpoint_dir / f"page_{next_pn:04d}.json"
                if not next_ckpt.exists():
                    _prefetch_future = _prefetch_executor.submit(_prefetch_page, next_pn)

            try:
                result = process_page(img_bgr, pn, renderer, checkpoint_dir, spellcheck_lang=spellcheck_lang)
                page_results.append(result)
            except Exception as e:
                logger.error("Error processing page %d: %s", pn, e)
                import traceback
                traceback.print_exc()
                continue

        _prefetch_executor.shutdown(wait=True)

    if not page_results:
        logger.error("No pages processed. Aborting.")
        sys.exit(1)

    if spellcheck_lang:
        _apply_spellcheck(page_results, spellcheck_lang)

    logger.info("Assembling EPUB (%d pages)...", len(page_results))
    tess_lang = _to_tess_lang(args.lang) if args.lang else None
    build_epub(
        page_results, output_path, title, author,
        toc_page_nums=toc_page_nums, source_path=input_path, tess_lang=tess_lang,
    )
    logger.info("Done: %s", output_path)


class _NullRenderer:
    """Drop-in replacement when MathJax is unavailable — leaves LaTeX as-is."""

    def render(self, latex: str, display: bool) -> str:
        delim = "$$" if display else "$"
        safe = html.escape(latex)
        return f'<code class="math-fallback" data-latex="{safe}">{delim}{safe}{delim}</code>'

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


if __name__ == "__main__":
    main()
