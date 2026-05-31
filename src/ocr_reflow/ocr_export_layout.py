"""OCR export: run layout analysis on a single page and produce an HTML string.

Two output modes:
  ocr_page_to_html()        — layout-preserving (absolutely-positioned blocks)
  ocr_page_to_html_simple() — linear flow (plain <p> / <img> sequence)

For streaming, use ocr_page_block_generator() which yields (event, payload)
tuples as OCR completes per batch.

Usage (CLI):
    python ocr_export_layout.py INPUT [--page N] [--output-dir DIR]

Block handling:
    plain text / title / titled_block_* -> LightOnOCR -> <p> (with MathJax for inline math)
    isolate_formula / isolate_formula_and_caption -> LightOnOCR -> <p> (MathJax display math)
    figure / figure_and_caption / figure_caption -> inline base64 <img>
    table / table_caption / table_footnote -> inline base64 <img>
    abandon -> skip
"""

import argparse
import base64
import html
import io
import logging
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

_lightonocr_processor = None
_lightonocr_model = None


def _get_device():
    """Detect compute device: CUDA if available, else CPU."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_lightonocr():
    """Get or create the cached LightOnOCR-2-1B model (VLM for OCR)."""

    if _lightonocr_model is None:
        # Cross-import deduplication: the same physical file can be loaded
        # under two sys.modules keys (ocr_export_layout vs
        # ocr_reflow.ocr_export_layout) depending on invocation mode.
        # Share the singleton across both copies.
        import sys
        alt_name = (
            "ocr_reflow.ocr_export_layout"
            if __name__ == "ocr_export_layout"
            else "ocr_export_layout"
        )
        alt = sys.modules.get(alt_name)
        if alt is not None and alt is not sys.modules.get(__name__):
            alt_model = getattr(alt, "_lightonocr_model", None)
            if alt_model is not None:
                _lightonocr_processor = alt._lightonocr_processor
                _lightonocr_model = alt_model
                return _lightonocr_processor, _lightonocr_model

        print("Loading LightOnOCR-2-1B...", file=sys.stderr)
        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        model_id = "lightonai/LightOnOCR-2-1B"
        device = _get_device()
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        _lightonocr_processor = LightOnOcrProcessor.from_pretrained(model_id)
        _lightonocr_model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        _lightonocr_model.eval()
        _lightonocr_model = torch.compile(_lightonocr_model, mode="reduce-overhead")
        print("LightOnOCR-2-1B loaded.", file=sys.stderr)

        # Sync to the other module copy so it finds the model next time
        if alt is not None and alt is not sys.modules.get(__name__):
            alt._lightonocr_processor = _lightonocr_processor
            alt._lightonocr_model = _lightonocr_model

    return _lightonocr_processor, _lightonocr_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_OCR_SIDE = 1024


def _resize_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Scale down so the longest side <= _MAX_OCR_SIDE, preserving aspect ratio."""
    w, h = pil_img.size
    longest = max(w, h)
    if longest <= _MAX_OCR_SIDE:
        return pil_img
    scale = _MAX_OCR_SIDE / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array (OpenCV format) to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# Resized height threshold above which a plain text crop is split before OCR.
# At 1024px wide, 250px height keeps input tokens ~200 — well within the fast generation bucket.
_MAX_OCR_HEIGHT = 250
# Minimum whitespace run (in original pixels) to be considered a valid split point.
_MIN_GAP_PX = 5


def _find_split_y(crop_bgr: np.ndarray) -> int | None:
    """Find the best horizontal split point using XY-cut on row brightness.

    Uses relative normalisation so it works on both white and gray/yellowed pages.
    Returns the Y coordinate (in crop pixels) of the split, or None if no
    suitable whitespace gap is found.
    """
    t0 = time.perf_counter()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = gray.shape[0]

    # Per-row mean brightness
    row_means = gray.mean(axis=1)

    # Smooth to suppress noise from individual light pixels between lines
    kernel = np.ones(7) / 7
    smoothed = np.convolve(row_means, kernel, mode="same")

    # Normalise to [0, 1] — relative to this crop's own brightness range
    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo < 1.0:
        print(f"[timing] _find_split_y ({crop_bgr.shape[1]}x{crop_bgr.shape[0]}px): {time.perf_counter()-t0:.3f}s  result=None (uniform)", file=sys.stderr)
        return None  # uniform image (blank or solid), can't split meaningfully
    normalized = (smoothed - lo) / (hi - lo)

    # Whitespace rows: relatively bright (top 15% of brightness range)
    is_gap = normalized > 0.85

    # Find contiguous whitespace runs of at least _MIN_GAP_PX rows
    best_y = None
    best_dist = h  # distance from midpoint — prefer splits near centre
    i = 0
    midpoint = h // 2
    while i < h:
        if is_gap[i]:
            j = i
            while j < h and is_gap[j]:
                j += 1
            run_len = j - i
            if run_len >= _MIN_GAP_PX:
                gap_mid = (i + j) // 2
                dist = abs(gap_mid - midpoint)
                if dist < best_dist:
                    best_dist = dist
                    best_y = gap_mid
            i = j
        else:
            i += 1

    print(f"[timing] _find_split_y ({crop_bgr.shape[1]}x{crop_bgr.shape[0]}px): {time.perf_counter()-t0:.3f}s  split_y={best_y}", file=sys.stderr)
    return best_y


def _split_plain_text_crop(crop_bgr: np.ndarray, _depth: int = 0) -> list:
    """Recursively split a plain text crop at whitespace gaps (XY-cut).

    Returns a list of BGR sub-crops, each small enough for OCR within the
    token budget.  Only splits when the resized height would exceed
    _MAX_OCR_HEIGHT.  Falls back to the unsplit crop if no gap is found.
    """
    t0 = time.perf_counter()
    # Check resized height
    pil = _bgr_to_pil(crop_bgr)
    resized = _resize_for_ocr(pil)
    if resized.size[1] <= _MAX_OCR_HEIGHT:
        print(f"[timing] _split_plain_text_crop depth={_depth} ({crop_bgr.shape[1]}x{crop_bgr.shape[0]}px → resized {resized.size[0]}x{resized.size[1]}px): {time.perf_counter()-t0:.3f}s  no split needed", file=sys.stderr)
        return [crop_bgr]

    split_y = _find_split_y(crop_bgr)
    if split_y is None or split_y <= 0 or split_y >= crop_bgr.shape[0]:
        print(f"[timing] _split_plain_text_crop depth={_depth} ({crop_bgr.shape[1]}x{crop_bgr.shape[0]}px → resized {resized.size[0]}x{resized.size[1]}px): {time.perf_counter()-t0:.3f}s  no gap found, keeping as-is", file=sys.stderr)
        # No usable gap — return as-is rather than cutting mid-line
        return [crop_bgr]

    top = crop_bgr[:split_y, :]
    bot = crop_bgr[split_y:, :]
    print(f"[timing] _split_plain_text_crop depth={_depth} ({crop_bgr.shape[1]}x{crop_bgr.shape[0]}px → resized {resized.size[0]}x{resized.size[1]}px): {time.perf_counter()-t0:.3f}s  split at y={split_y} → top={top.shape[0]}px bot={bot.shape[0]}px", file=sys.stderr)
    return _split_plain_text_crop(top, _depth + 1) + _split_plain_text_crop(bot, _depth + 1)


def _ocr_single_crop(processor, model, device, dtype, crop):
    """OCR a single BGR crop. Used for OOM fallback and batch_size=1."""
    import torch
    pil_img = _resize_for_ocr(_bgr_to_pil(crop))
    conv = [{"role": "user", "content": [{"type": "image", "image": pil_img}]}]
    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
              for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512,
                                    do_sample=True, temperature=0.2, top_p=0.9, top_k=0)
    input_len = inputs["input_ids"].shape[1]
    gen = output_ids[0, input_len:]
    return processor.decode(gen, skip_special_tokens=True).strip(), pil_img, input_len, len(gen)


def _ocr_blocks_batch(crops: list, batch_size: int = 4) -> list:
    """Run LightOnOCR on a list of BGR crops using true batched generate().

    Crops are sorted by pixel area before batching to minimise padding waste.
    On CUDA OOM the failing batch falls back to sequential batch_size=1.
    Returns a list of plain-text strings in the same order as input crops.
    """
    import torch

    if not crops:
        return []

    processor, model = _get_lightonocr()
    t_inference_start = time.perf_counter()   # model loaded — pure inference from here
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Sort by pixel area ascending so similar-sized crops end up in the same batch
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1])
    sorted_crops = [crops[i] for i in order]
    results_sorted = [None] * len(sorted_crops)

    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    try:
        for batch_num, batch_start in enumerate(range(0, len(sorted_crops), batch_size), start=1):
            batch_crops = sorted_crops[batch_start:batch_start + batch_size]
            t_batch = time.perf_counter()

            if len(batch_crops) == 1:
                # Single crop — skip batching overhead
                text, pil_img, input_len, gen_len = _ocr_single_crop(
                    processor, model, device, dtype, batch_crops[0]
                )
                results_sorted[batch_start] = text
                print(
                    f"[timing] OCR batch {batch_num} (single): "
                    f"orig={batch_crops[0].shape[1]}x{batch_crops[0].shape[0]}px "
                    f"resized={pil_img.size[0]}x{pil_img.size[1]}px "
                    f"input_tokens={input_len} gen_tokens={gen_len} "
                    f"generate={time.perf_counter()-t_batch:.3f}s",
                    file=sys.stderr,
                )
                continue

            try:
                # Build prompt strings (no tokenize) + resize images
                pil_images = [_resize_for_ocr(_bgr_to_pil(c)) for c in batch_crops]
                texts = []
                for pil in pil_images:
                    conv = [{"role": "user", "content": [{"type": "image", "image": pil}]}]
                    texts.append(processor.apply_chat_template(
                        conv, add_generation_prompt=True, tokenize=False
                    ))

                # Batch tokenise + image-process in one call
                inputs = processor(
                    text=texts, images=pil_images,
                    padding=True, return_tensors="pt",
                )
                inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                          for k, v in inputs.items()}

                t_gen = time.perf_counter()
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=512,
                        do_sample=True, temperature=0.2, top_p=0.9, top_k=0,
                    )
                t_gen_done = time.perf_counter()

                # Left-padded: all sequences end at the same position
                input_len = inputs["input_ids"].shape[1]
                gen_lens = []
                hit_limit = []  # indices (within batch) that hit max_new_tokens
                for j in range(len(batch_crops)):
                    gen = output_ids[j, input_len:]
                    # strip trailing pad tokens
                    gen = gen[gen != processor.tokenizer.pad_token_id] if processor.tokenizer.pad_token_id is not None else gen
                    results_sorted[batch_start + j] = processor.decode(gen, skip_special_tokens=True).strip()
                    gen_lens.append(len(gen))
                    if len(gen) >= 512:
                        hit_limit.append(j)

                print(
                    f"[timing] OCR batch {batch_num} ({len(batch_crops)} crops): "
                    f"padded_input_tokens={input_len} "
                    f"gen_tokens={gen_lens} "
                    f"preprocess={t_gen-t_batch:.3f}s "
                    f"generate={t_gen_done-t_gen:.3f}s "
                    f"total={t_gen_done-t_batch:.3f}s",
                    file=sys.stderr,
                )

                # Re-run any crop that hit the token limit — likely a hallucination loop
                for j in hit_limit:
                    print(
                        f"[timing] OCR batch {batch_num} crop {j+1}: hit max_new_tokens ({gen_lens[j]}) — re-running individually",
                        file=sys.stderr,
                    )
                    t_rerun = time.perf_counter()
                    text, pil_img, rerun_input_len, rerun_gen_len = _ocr_single_crop(
                        processor, model, device, dtype, batch_crops[j]
                    )
                    results_sorted[batch_start + j] = text
                    print(
                        f"[timing] OCR batch {batch_num} crop {j+1} rerun: "
                        f"input_tokens={rerun_input_len} gen_tokens={rerun_gen_len} "
                        f"generate={time.perf_counter()-t_rerun:.3f}s",
                        file=sys.stderr,
                    )

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(
                    f"[timing] OCR batch {batch_num}: CUDA OOM — falling back to batch_size=1",
                    file=sys.stderr,
                )
                for j, crop in enumerate(batch_crops):
                    t_fb = time.perf_counter()
                    text, pil_img, input_len, gen_len = _ocr_single_crop(
                        processor, model, device, dtype, crop
                    )
                    results_sorted[batch_start + j] = text
                    print(
                        f"[timing] OCR batch {batch_num} fallback crop {j+1}/{len(batch_crops)}: "
                        f"input_tokens={input_len} gen_tokens={gen_len} "
                        f"generate={time.perf_counter()-t_fb:.3f}s",
                        file=sys.stderr,
                    )
    finally:
        processor.tokenizer.padding_side = orig_padding_side

    # Restore original crop order
    results = [None] * len(crops)
    for sorted_i, orig_i in enumerate(order):
        results[orig_i] = results_sorted[sorted_i]

    print(
        f"[timing] _ocr_blocks_batch inference only: {time.perf_counter()-t_inference_start:.3f}s  "
        f"({len(crops)} crops, batch_size={batch_size})",
        file=sys.stderr,
    )
    return results


def _wrap_gather(inner: str) -> str:
    """Wrap bare multi-line LaTeX (using \\\\ line breaks) in \\begin{gather}.

    If the content already has an environment (\\begin{...}) or has no line
    breaks, return it unchanged.
    """
    if "\\\\" in inner and "\\begin{" not in inner:
        return f"\\begin{{gather}}\n{inner}\n\\end{{gather}}"
    return inner


def _lightonocr_to_html(text: str) -> str:
    """Convert LightOnOCR raw output to an HTML paragraph fragment.

    LightOnOCR produces one of three formats:
      1. ```latex\\n...\\n```  — display math (fence style)
      2. $$\\n...\\n$$         — display math (delimiter style)
      3. mixed / plain text   — may contain inline $...$ math or plain text

    In all cases MathJax will handle the math delimiters; we only need to
    strip the ```latex fence and avoid html-escaping LaTeX content.
    """
    text = text.strip()

    # Remove end-of-line hyphens: "вычис-\nления" → "вычисления"
    # Only merge when the character after the newline is lowercase (real dashes
    # like "—" are mid-line and never followed by \n; proper names like
    # "Иванов-Петров" are also mid-line).
    text = re.sub(r'(\w)- *\n([а-яёa-z])', r'\1\2', text)

    # Format 1: ```latex\n...\n```  -> display math
    if text.startswith("```latex"):
        inner = text[len("```latex"):].strip()
        if inner.endswith("```"):
            inner = inner[:-3].strip()
        # Strip any stray $$ delimiters the model may have mixed in; each
        # resulting part becomes its own display block.
        parts = [p.strip() for p in inner.split("$$") if p.strip()]
        escaped = [p.replace("&", "&amp;").replace("<", "&lt;") for p in parts]
        return "".join(f"<p>$$\n{_wrap_gather(p)}\n$$</p>\n" for p in escaped)

    # Format 2: $$...$$ — strip all $$ delimiters; each segment becomes its
    # own display block so multi-line formulas render on separate lines.
    if text.startswith("$$"):
        parts = [p.strip() for p in text.split("$$") if p.strip()]
        return "".join(f"<p>$$\n{_wrap_gather(p)}\n$$</p>\n" for p in parts)

    # Format 3: contains inline math or plain text
    # If any $ present, pass raw (MathJax handles it); otherwise html-escape
    if "$" in text:
        # Escape bare & and < that are outside math delimiters — they are
        # LaTeX/text characters that must be valid XML in XHTML.
        text = text.replace("&", "&amp;").replace("<", "&lt;")
        return f"<p>{text}</p>\n"

    return f"<p>{html.escape(text)}</p>\n"


def _figure_to_base64(img_bgr: np.ndarray) -> str:
    """Encode a BGR crop as a base64 PNG data URI."""
    pil_img = _bgr_to_pil(img_bgr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _crop(img_bgr: np.ndarray, box) -> np.ndarray:
    """Crop img_bgr to a shapely box (or any object with .bounds)."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box.bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    return img_bgr[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _html_head(base_url: str) -> str:
    """Generate standard HTML head boilerplate for the visual debug page."""
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Export</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']], packages: {{'[+]': ['ams']}} }},
  startup: {{
    ready() {{
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => MathJax.typesetPromise());
    }}
  }}
}};
</script>
<script src="{base_url}/static/mathjax/tex-svg-full.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ width: 100%; background: #f5f0e8; font-family: serif; font-size: calc(4vw * var(--font-scale, 1)); line-height: 1.6; padding: 0 1em; }}
  .block {{ margin-bottom: 0.8em; }}
  .block p {{ margin: 0.4em 0; }}
  .block.formula {{ overflow-x: auto; }}
  .block.figure img {{ max-width: 100%; height: auto; display: block; margin: 0.5em 0; }}
  mjx-container[display="true"] {{ display: block; overflow-x: auto; }}
</style>
</head>
<body>
"""

_HTML_TAIL = """\
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

# Labels treated as text (run LightOnOCR)
_TEXT_LABELS = {"plain text", "title", "titled_block_title", "titled_block_body"}

# Labels treated as formulas (run LightOnOCR, render via MathJax)
_FORMULA_LABELS = {"isolate_formula", "isolate_formula_and_caption"}

# Labels treated as images (save crop)
_IMAGE_LABELS = {
    "figure", "figure_and_caption", "figure_caption",
    "table", "table_caption", "table_footnote",
}

# Labels to skip
_SKIP_LABELS = {"abandon"}

# All labels that go through LightOnOCR
_OCR_LABELS = _TEXT_LABELS | _FORMULA_LABELS

# Merging adjacent text blocks to reduce OCR calls
_MERGE_GAP_PX = 40       # max vertical gap between mergeable text blocks
_MERGE_X_OVERLAP = 0.5   # min horizontal overlap fraction (of narrower block)


def _group_mergeable_text_blocks(block_data, img_bgr):
    """Group consecutive plain-text blocks that are close and column-aligned.

    Returns a list of groups, each a dict:
      {"type": "single", "indices": [i]}
      {"type": "merged", "indices": [i, j, ...], "crop": merged_crop_bgr}

    Merge conditions (all must hold):
      1. Both blocks are _TEXT_LABELS with non-None crops
      2. Vertical gap between them <= _MERGE_GAP_PX
      3. Horizontal overlap >= _MERGE_X_OVERLAP of the narrower block's width
      4. No non-text block straddles the gap between them
    """
    def _x_overlap_frac(b1, b2):
        ax1, _, ax2, _ = b1.bounds
        bx1, _, bx2, _ = b2.bounds
        overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        narrower = min(ax2 - ax1, bx2 - bx1)
        return overlap / narrower if narrower > 0 else 0.0

    def _gap_is_clear(cur_geom, nxt_geom):
        """Return True if no non-text block straddles the vertical gap."""
        _, cur_y2 = cur_geom.bounds[0], cur_geom.bounds[3]
        _, nxt_y1 = nxt_geom.bounds[0], nxt_geom.bounds[1]
        for geom, label, crop in block_data:
            if label in _TEXT_LABELS or label in _SKIP_LABELS:
                continue
            _, gy1, _, gy2 = geom.bounds
            # block straddles the gap if it starts before nxt_y1 and ends after cur_y2
            if gy1 < nxt_y1 and gy2 > cur_y2:
                return False
        return True

    def _can_merge(i, j):
        geom_i, label_i, crop_i = block_data[i]
        geom_j, label_j, crop_j = block_data[j]
        if label_i not in _TEXT_LABELS or label_j not in _TEXT_LABELS:
            return False
        if crop_i is None or crop_j is None:
            return False
        _, _, _, iy2 = geom_i.bounds
        _, jy1, _, _ = geom_j.bounds
        if jy1 - iy2 > _MERGE_GAP_PX:
            return False
        if _x_overlap_frac(geom_i, geom_j) < _MERGE_X_OVERLAP:
            return False
        if not _gap_is_clear(geom_i, geom_j):
            return False
        return True

    def _make_merged_crop(indices):
        bounds = [block_data[i][0].bounds for i in indices]
        x1 = int(min(b[0] for b in bounds))
        y1 = int(min(b[1] for b in bounds))
        x2 = int(max(b[2] for b in bounds))
        y2 = int(max(b[3] for b in bounds))
        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return img_bgr[y1:y2, x1:x2].copy()

    groups = []
    current = [0]

    for j in range(1, len(block_data)):
        i = current[-1]
        # Try to merge j with the last block in current group
        if _can_merge(i, j):
            current.append(j)
        else:
            # Flush current group
            if len(current) == 1:
                groups.append({"type": "single", "indices": current})
            else:
                groups.append({"type": "merged", "indices": current,
                                "crop": _make_merged_crop(current)})
            current = [j]

    # Flush last group
    if len(current) == 1:
        groups.append({"type": "single", "indices": current})
    else:
        groups.append({"type": "merged", "indices": current,
                        "crop": _make_merged_crop(current)})

    return groups


def _split_ocr_text_to_blocks(text, block_data, indices):
    """Distribute OCR text across multiple blocks by paragraph breaks.

    Paragraphs are split on \\n\\n. Distributed proportionally by block height.
    Remaining paragraphs (if any) go to the last block.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return {i: "" for i in indices}

    heights = [block_data[i][0].bounds[3] - block_data[i][0].bounds[1] for i in indices]
    total_h = sum(heights)

    assigned = [[] for _ in indices]
    para_idx = 0
    for k, h in enumerate(heights):
        if total_h > 0:
            count = round(len(paragraphs) * h / total_h)
        else:
            count = 0
        # ensure we don't starve later blocks if rounding goes wrong
        remaining_blocks = len(indices) - k
        remaining_paras = len(paragraphs) - para_idx
        count = max(0, min(count, remaining_paras - (remaining_blocks - 1)))
        if k == len(indices) - 1:
            # last block gets everything remaining
            count = remaining_paras
        assigned[k] = paragraphs[para_idx:para_idx + count]
        para_idx += count

    return {i: "\n\n".join(assigned[k]) for k, i in enumerate(indices)}


def ocr_page_to_html(
    img_bgr: np.ndarray,
    no_pix2tex: bool = False,  # kept for backward compat, ignored
    base_url: str = "http://192.168.1.121:8000",
) -> str:
    """Run layout + OCR on img_bgr and return a self-contained HTML string.

    All figure/table crops are embedded as inline base64 PNG data URIs.
    Formula blocks are processed by LightOnOCR and rendered via MathJax.
    MathJax is served from base_url/static/mathjax/tex-chtml-full.js.
    """
    t_page_start = time.perf_counter()

    # --- Layout analysis ---
    try:
        from layout import layout_from_array
    except ImportError:
        from ocr_reflow.layout import layout_from_array

    print("Running layout analysis...", file=sys.stderr)
    t_layout = time.perf_counter()
    blocks = layout_from_array(img_bgr)  # list of (shapely_geom, label_str)
    t_layout_done = time.perf_counter()
    print(f"[timing] layout analysis: {t_layout_done-t_layout:.3f}s  blocks={len(blocks)}", file=sys.stderr)

    # Sort blocks top-to-bottom so HTML order matches page reading order
    blocks = sorted(blocks, key=lambda b: b[0].bounds[1])

    # --- Process blocks ---
    page_h, page_w = img_bgr.shape[:2]
    aspect_pct = page_h / page_w * 100

    html_parts = [_html_head(base_url)]

    # First pass: crop all blocks
    t_crop = time.perf_counter()
    block_data = []  # list of (geom, label, crop_or_None)
    for geom, label in blocks:
        if label in _SKIP_LABELS:
            block_data.append((geom, label, None))
            continue
        crop = _crop(img_bgr, geom)
        if crop.size == 0:
            block_data.append((geom, label, None))
            continue
        block_data.append((geom, label, crop))

    # Collect image bboxes (in page coordinates) to mask out of text crops.
    # This prevents figure content from bleeding into OCR input when a low-confidence
    # plain text block overlaps a figure region.
    # NOTE: formula blocks are intentionally excluded — they need to be OCR'd themselves.
    image_bboxes = []  # list of (x1, y1, x2, y2) in page coords
    for geom, label in blocks:
        if label in _IMAGE_LABELS:
            x1, y1, x2, y2 = geom.bounds
            image_bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    def _mask_image_regions(crop: np.ndarray, block_geom) -> np.ndarray:
        """White-fill any image/formula bbox that overlaps this block's crop."""
        bx1, by1, bx2, by2 = [int(v) for v in block_geom.bounds]
        masked = crop.copy()
        for ix1, iy1, ix2, iy2 in image_bboxes:
            # Intersection in page coords
            ox1 = max(bx1, ix1) - bx1
            oy1 = max(by1, iy1) - by1
            ox2 = min(bx2, ix2) - bx1
            oy2 = min(by2, iy2) - by1
            if ox2 > ox1 and oy2 > oy1:
                masked[oy1:oy2, ox1:ox2] = 255
        return masked

    def _merged_geom_bounds(indices):
        """Union bounding box of a list of block indices as (x1,y1,x2,y2)."""
        bounds = [block_data[i][0].bounds for i in indices]
        return (
            min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds),
        )

    # Group adjacent text blocks to reduce OCR calls
    t_merge = time.perf_counter()
    groups = _group_mergeable_text_blocks(block_data, img_bgr)
    n_text_blocks = sum(1 for _, label, crop in block_data if label in _TEXT_LABELS and crop is not None)
    n_merged_groups = sum(1 for g in groups if g["type"] == "merged")
    n_calls_saved = sum(len(g["indices"]) - 1 for g in groups if g["type"] == "merged")
    t_merge_done = time.perf_counter()
    print(f"[timing] merge: {t_merge_done-t_merge:.3f}s  "
          f"{n_text_blocks} text blocks → {n_merged_groups} merged groups, "
          f"{n_calls_saved} OCR calls saved", file=sys.stderr)

    # Build OCR work list from groups
    # ocr_groups      : groups that need OCR
    # ocr_group_crops : one masked crop per group (merged or single)
    t_crop_done_start = time.perf_counter()
    ocr_groups = []
    ocr_group_crops = []

    for g in groups:
        indices = g["indices"]
        first_label = block_data[indices[0]][1]

        if g["type"] == "merged":
            # Merged text group: use the pre-built merged crop
            raw_crop = g["crop"]
            # Build a fake geom bounds for masking (union bbox)
            ux1, uy1, ux2, uy2 = _merged_geom_bounds(indices)
            class _FakeGeom:
                bounds = (ux1, uy1, ux2, uy2)
            masked = _mask_image_regions(raw_crop, _FakeGeom())
            ocr_groups.append(g)
            ocr_group_crops.append(masked)

        else:
            # Single block
            i = indices[0]
            geom, label, crop = block_data[i]
            if label not in _OCR_LABELS or crop is None:
                continue
            masked = _mask_image_regions(crop, geom)
            ocr_groups.append(g)
            ocr_group_crops.append(masked)

    t_crop_done = time.perf_counter()
    print(f"[timing] crop+mask: {t_crop_done-t_crop_done_start:.3f}s  ({len(ocr_group_crops)} groups)", file=sys.stderr)

    # XY-cut split each group crop so tall crops don't exceed the token budget.
    t_split = time.perf_counter()
    ocr_flat_crops = []
    ocr_split_counts = []
    for idx, masked_crop in enumerate(ocr_group_crops):
        g = ocr_groups[idx]
        first_label = block_data[g["indices"][0]][1]
        if first_label in _TEXT_LABELS:
            sub_crops = _split_plain_text_crop(masked_crop)
        else:
            sub_crops = [masked_crop]
        ocr_flat_crops.extend(sub_crops)
        ocr_split_counts.append(len(sub_crops))
    t_split_done = time.perf_counter()
    n_splits = sum(ocr_split_counts) - len(ocr_split_counts)
    print(f"[timing] split: {t_split_done-t_split:.3f}s  "
          f"({len(ocr_groups)} groups → {len(ocr_flat_crops)} crops, {n_splits} extra splits)", file=sys.stderr)

    print(f"  Running batch OCR on {len(ocr_flat_crops)} crops "
          f"({len(ocr_groups)} groups, {n_splits} extra splits)...",
          file=sys.stderr)
    t_ocr = time.perf_counter()
    flat_texts = _ocr_blocks_batch(ocr_flat_crops)
    t_ocr_done = time.perf_counter()
    print(f"[timing] OCR total: {t_ocr_done-t_ocr:.3f}s", file=sys.stderr)

    # Re-assemble: join sub-crop texts per group, then distribute to blocks
    ocr_results = {}
    flat_idx = 0
    for idx, count in enumerate(ocr_split_counts):
        g = ocr_groups[idx]
        parts = flat_texts[flat_idx:flat_idx + count]
        group_text = "\n\n".join(p for p in parts if p)
        flat_idx += count

        if g["type"] == "single":
            ocr_results[g["indices"][0]] = group_text
        else:
            # Distribute paragraphs across original blocks proportionally
            split = _split_ocr_text_to_blocks(group_text, block_data, g["indices"])
            ocr_results.update(split)

    # Second pass: build HTML in reading order (layout-preserving)
    t_html = time.perf_counter()
    for i, (geom, label, crop) in enumerate(block_data):
        print(f"  Block {i+1}/{len(block_data)}: {label}", file=sys.stderr)

        if crop is None:
            continue

        x1, y1, x2, y2 = geom.bounds

        if label in _OCR_LABELS:
            text = ocr_results[i]
            block_class = "text" if label in _TEXT_LABELS else "formula"
            html_parts.append(f'<div class="block {block_class}">\n')
            html_parts.append(_lightonocr_to_html(text))
            html_parts.append('</div>\n')

        elif label in _IMAGE_LABELS:
            data_uri = _figure_to_base64(crop)
            html_parts.append('<div class="block figure">\n')
            html_parts.append(f'<img src="{data_uri}" alt="{html.escape(label)}">\n')
            html_parts.append('</div>\n')

        else:
            logger.warning("Unknown block label '%s', saving as image.", label)
            data_uri = _figure_to_base64(crop)
            html_parts.append('<div class="block figure">\n')
            html_parts.append(f'<img src="{data_uri}" alt="{html.escape(label)}">\n')
            html_parts.append('</div>\n')

    html_parts.append(_HTML_TAIL)
    t_html_done = time.perf_counter()
    print(f"[timing] html assembly: {t_html_done-t_html:.3f}s", file=sys.stderr)
    print(f"[timing] ocr_page_to_html TOTAL: {t_html_done-t_page_start:.3f}s", file=sys.stderr)
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Simple (flow-layout) export — linear <p>/<img> sequence, sequential OCR
# ---------------------------------------------------------------------------

def ocr_page_to_html_simple(
    img_bgr: np.ndarray,
    no_pix2tex: bool = False,  # kept for backward compat, ignored
    base_url: str = "http://192.168.1.121:8000",
) -> str:
    """Run layout + OCR on img_bgr and return a self-contained HTML string.

    Blocks are rendered as a linear flow (plain <p> / <img> tags), not
    absolutely positioned.  Uses sequential (non-batched) OCR.
    """
    t_page_start = time.perf_counter()

    try:
        from layout import layout_from_array
    except ImportError:
        from ocr_reflow.layout import layout_from_array

    print("Running layout analysis...", file=sys.stderr)
    t_layout = time.perf_counter()
    blocks = layout_from_array(img_bgr)
    t_layout_done = time.perf_counter()
    print(f"[timing] layout analysis: {t_layout_done-t_layout:.3f}s  blocks={len(blocks)}", file=sys.stderr)

    blocks = sorted(blocks, key=lambda b: b[0].bounds[1])

    # Build simple flow HTML head
    simple_head = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Export</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']], packages: {{'[+]': ['ams']}} }},
  startup: {{
    ready() {{
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => MathJax.typesetPromise());
    }}
  }}
}};
</script>
<script src="{base_url}/static/mathjax/tex-svg-full.js"></script>
<style>
  body {{ font-family: serif; max-width: 100%; margin: 0; padding: 1em 1.4em; box-sizing: border-box; line-height: 1.6; }}
  img {{ max-width: 100%; display: block; margin: 1em 0; }}
  p {{ margin: 0.8em 0; }}
  mjx-container[display="true"] {{ display: block; max-width: 100%; overflow-x: auto; }}
</style>
</head>
<body>
"""

    html_parts = [simple_head]

    t_crop = time.perf_counter()
    block_data = []
    for geom, label in blocks:
        if label in _SKIP_LABELS:
            block_data.append((geom, label, None))
            continue
        crop = _crop(img_bgr, geom)
        if crop.size == 0:
            block_data.append((geom, label, None))
            continue
        block_data.append((geom, label, crop))

    image_bboxes = []
    for geom, label in blocks:
        if label in _IMAGE_LABELS:
            x1, y1, x2, y2 = geom.bounds
            image_bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    ocr_indices = [i for i, (_, label, crop) in enumerate(block_data)
                   if label in _OCR_LABELS and crop is not None]

    def _mask_image_regions_simple(crop: np.ndarray, block_geom) -> np.ndarray:
        bx1, by1, bx2, by2 = [int(v) for v in block_geom.bounds]
        masked = crop.copy()
        for ix1, iy1, ix2, iy2 in image_bboxes:
            ox1 = max(bx1, ix1) - bx1
            oy1 = max(by1, iy1) - by1
            ox2 = min(bx2, ix2) - bx1
            oy2 = min(by2, iy2) - by1
            if ox2 > ox1 and oy2 > oy1:
                masked[oy1:oy2, ox1:ox2] = 255
        return masked

    ocr_crops = [
        _mask_image_regions_simple(block_data[i][2], block_data[i][0])
        for i in ocr_indices
    ]
    t_crop_done = time.perf_counter()
    print(f"[timing] crop+mask: {t_crop_done-t_crop:.3f}s  ({len(ocr_indices)} blocks)", file=sys.stderr)

    t_split = time.perf_counter()
    ocr_flat_crops = []
    ocr_split_counts = []
    for idx, masked_crop in enumerate(ocr_crops):
        block_idx = ocr_indices[idx]
        _, label, _ = block_data[block_idx]
        if label in _TEXT_LABELS:
            sub_crops = _split_plain_text_crop(masked_crop)
        else:
            sub_crops = [masked_crop]
        ocr_flat_crops.extend(sub_crops)
        ocr_split_counts.append(len(sub_crops))
    t_split_done = time.perf_counter()
    n_splits = sum(ocr_split_counts) - len(ocr_split_counts)
    print(f"[timing] split: {t_split_done-t_split:.3f}s  ({len(ocr_indices)} blocks → {len(ocr_flat_crops)} crops, {n_splits} extra splits)", file=sys.stderr)

    print(f"  Running sequential OCR on {len(ocr_flat_crops)} crops...", file=sys.stderr)
    t_ocr = time.perf_counter()
    flat_texts = _ocr_blocks_batch(ocr_flat_crops, batch_size=1)
    t_ocr_done = time.perf_counter()
    print(f"[timing] OCR total: {t_ocr_done-t_ocr:.3f}s", file=sys.stderr)

    ocr_results = {}
    flat_idx = 0
    for idx, count in enumerate(ocr_split_counts):
        block_idx = ocr_indices[idx]
        parts = flat_texts[flat_idx:flat_idx + count]
        ocr_results[block_idx] = "\n\n".join(p for p in parts if p)
        flat_idx += count

    for i, (geom, label, crop) in enumerate(block_data):
        print(f"  Block {i+1}/{len(block_data)}: {label}", file=sys.stderr)
        if crop is None:
            continue
        if label in _OCR_LABELS:
            text = ocr_results[i]
            html_parts.append(_lightonocr_to_html(text))
        elif label in _IMAGE_LABELS:
            data_uri = _figure_to_base64(crop)
            html_parts.append(f'<img src="{data_uri}" alt="{html.escape(label)}">\n')
        else:
            logger.warning("Unknown block label '%s', saving as image.", label)
            data_uri = _figure_to_base64(crop)
            html_parts.append(f'<img src="{data_uri}" alt="{html.escape(label)}">\n')

    html_parts.append("</body>\n</html>\n")
    t_done = time.perf_counter()
    print(f"[timing] ocr_page_to_html_simple TOTAL: {t_done-t_page_start:.3f}s", file=sys.stderr)
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Streaming generator — yields (event, payload) per batch for SSE streaming
# ---------------------------------------------------------------------------

def ocr_page_block_generator(img_bgr: np.ndarray, base_url: str = "http://192.168.1.121:8000"):
    """Generate OCR results block by block for streaming.

    Yields tuples:
      ("layout_done", {"n_blocks": N, "aspect_pct": float})
      ("block", {"index": i, "html": "<div ...>...</div>"})   — one per block, after each batch
      ("done", {})

    Designed to be consumed by an async wrapper that bridges to SSE.
    The generator runs synchronously (call from a thread pool executor).
    """
    t_page_start = time.perf_counter()

    try:
        from layout import layout_from_array
    except ImportError:
        from ocr_reflow.layout import layout_from_array

    print("Running layout analysis...", file=sys.stderr)
    t_layout = time.perf_counter()
    blocks = layout_from_array(img_bgr)
    t_layout_done = time.perf_counter()
    print(f"[timing] layout analysis: {t_layout_done-t_layout:.3f}s  blocks={len(blocks)}", file=sys.stderr)

    blocks = sorted(blocks, key=lambda b: b[0].bounds[1])
    page_h, page_w = img_bgr.shape[:2]
    aspect_pct = page_h / page_w * 100

    yield ("layout_done", {"n_blocks": len(blocks), "aspect_pct": aspect_pct})

    # --- Crop and mask ---
    block_data = []
    for geom, label in blocks:
        if label in _SKIP_LABELS:
            block_data.append((geom, label, None))
            continue
        crop = _crop(img_bgr, geom)
        if crop.size == 0:
            block_data.append((geom, label, None))
            continue
        block_data.append((geom, label, crop))

    image_bboxes = []
    for geom, label in blocks:
        if label in _IMAGE_LABELS:
            x1, y1, x2, y2 = geom.bounds
            image_bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    def _mask(crop, block_geom):
        bx1, by1, bx2, by2 = [int(v) for v in block_geom.bounds]
        masked = crop.copy()
        for ix1, iy1, ix2, iy2 in image_bboxes:
            ox1 = max(bx1, ix1) - bx1
            oy1 = max(by1, iy1) - by1
            ox2 = min(bx2, ix2) - bx1
            oy2 = min(by2, iy2) - by1
            if ox2 > ox1 and oy2 > oy1:
                masked[oy1:oy2, ox1:ox2] = 255
        return masked

    # --- Build OCR work list and yield figures immediately in reading order ---
    # Figures need no OCR — yield them now so they appear at the correct position
    # in the linear flow (block_data is sorted by y1).
    # Text/formula blocks go into the OCR queue and are yielded after each batch.
    ocr_groups = []
    ocr_group_crops = []
    for i, (geom, label, crop) in enumerate(block_data):
        if crop is None:
            continue
        if label in _IMAGE_LABELS:
            data_uri = _figure_to_base64(crop)
            frag = (
                f'<div id="block-{i}" class="block figure">\n'
                f'<img src="{data_uri}" alt="{html.escape(label)}">\n'
                f'</div>\n'
            )
            yield ("block", {"index": i, "html": frag})
        elif label in _OCR_LABELS:
            masked = _mask(crop, geom)
            ocr_groups.append({"type": "single", "indices": [i]})
            ocr_group_crops.append(masked)

    # XY-cut split
    ocr_flat_crops = []
    ocr_split_counts = []
    for idx, masked_crop in enumerate(ocr_group_crops):
        g = ocr_groups[idx]
        first_label = block_data[g["indices"][0]][1]
        if first_label in _TEXT_LABELS:
            sub_crops = _split_plain_text_crop(masked_crop)
        else:
            sub_crops = [masked_crop]
        ocr_flat_crops.extend(sub_crops)
        ocr_split_counts.append(len(sub_crops))

    # --- Process batches and yield results ---
    # We need to process one batch at a time and yield block HTML after each batch.
    # We replicate the batch loop from _ocr_blocks_batch but yield between batches.
    import torch

    processor, model = _get_lightonocr()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    batch_size = 4

    flat_texts_sorted = [None] * len(ocr_flat_crops)

    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    try:
        for batch_num, batch_start in enumerate(range(0, len(ocr_flat_crops), batch_size), start=1):
            batch_crops = ocr_flat_crops[batch_start:batch_start + batch_size]
            t_batch = time.perf_counter()

            if len(batch_crops) == 1:
                text, pil_img, input_len, gen_len = _ocr_single_crop(
                    processor, model, device, dtype, batch_crops[0]
                )
                flat_texts_sorted[batch_start] = text
                print(
                    f"[timing] OCR batch {batch_num} (single): "
                    f"input_tokens={input_len} gen_tokens={gen_len} "
                    f"generate={time.perf_counter()-t_batch:.3f}s",
                    file=sys.stderr,
                )
            else:
                try:
                    pil_images = [_resize_for_ocr(_bgr_to_pil(c)) for c in batch_crops]
                    texts = []
                    for pil in pil_images:
                        conv = [{"role": "user", "content": [{"type": "image", "image": pil}]}]
                        texts.append(processor.apply_chat_template(
                            conv, add_generation_prompt=True, tokenize=False
                        ))
                    inputs = processor(
                        text=texts, images=pil_images,
                        padding=True, return_tensors="pt",
                    )
                    inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                              for k, v in inputs.items()}
                    t_gen = time.perf_counter()
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs, max_new_tokens=512,
                            do_sample=True, temperature=0.2, top_p=0.9, top_k=0,
                        )
                    t_gen_done = time.perf_counter()
                    input_len = inputs["input_ids"].shape[1]
                    gen_lens = []
                    hit_limit = []
                    for j in range(len(batch_crops)):
                        gen = output_ids[j, input_len:]
                        gen = gen[gen != processor.tokenizer.pad_token_id] if processor.tokenizer.pad_token_id is not None else gen
                        flat_texts_sorted[batch_start + j] = processor.decode(gen, skip_special_tokens=True).strip()
                        gen_lens.append(len(gen))
                        if len(gen) >= 512:
                            hit_limit.append(j)
                    print(
                        f"[timing] OCR batch {batch_num} ({len(batch_crops)} crops): "
                        f"gen_tokens={gen_lens} "
                        f"preprocess={t_gen-t_batch:.3f}s "
                        f"generate={t_gen_done-t_gen:.3f}s "
                        f"total={t_gen_done-t_batch:.3f}s",
                        file=sys.stderr,
                    )
                    for j in hit_limit:
                        print(f"[timing] OCR batch {batch_num} crop {j+1}: hit max_new_tokens — re-running", file=sys.stderr)
                        t_rerun = time.perf_counter()
                        text, pil_img, rerun_input_len, rerun_gen_len = _ocr_single_crop(
                            processor, model, device, dtype, batch_crops[j]
                        )
                        flat_texts_sorted[batch_start + j] = text
                        print(
                            f"[timing] OCR batch {batch_num} crop {j+1} rerun: "
                            f"input_tokens={rerun_input_len} gen_tokens={rerun_gen_len} "
                            f"generate={time.perf_counter()-t_rerun:.3f}s",
                            file=sys.stderr,
                        )
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    print(f"[timing] OCR batch {batch_num}: CUDA OOM — falling back to batch_size=1", file=sys.stderr)
                    for j, crop in enumerate(batch_crops):
                        text, pil_img, input_len, gen_len = _ocr_single_crop(
                            processor, model, device, dtype, crop
                        )
                        flat_texts_sorted[batch_start + j] = text

            # After each batch: figure out which original block indices are now complete
            # and yield their HTML fragments.
            # We need to check which groups have all their flat crops resolved.
            # Build a mapping: flat_crop_index -> group_index
            # (computed once outside the loop would be cleaner, but we do it lazily here)
            # Determine which groups are fully resolved after this batch
            flat_idx_cursor = 0
            group_flat_ranges = []  # (start, end) in flat_texts_sorted order
            for idx in range(len(ocr_groups)):
                count = ocr_split_counts[idx]
                group_flat_ranges.append((flat_idx_cursor, flat_idx_cursor + count))
                flat_idx_cursor += count

            for group_idx, (g_start, g_end) in enumerate(group_flat_ranges):
                # Check if all flat crops for this group are resolved
                if all(flat_texts_sorted[i] is not None for i in range(g_start, g_end)):
                    g = ocr_groups[group_idx]
                    # Check if we haven't yielded this group yet
                    # We use a sentinel: replace None with a marker after yielding
                    # Instead, track via a set — but we can't mutate flat_texts_sorted easily.
                    # Simple approach: only yield groups whose last flat crop was in this batch.
                    last_flat_in_group = g_end - 1
                    # The sorted index of the last crop in this group
                    # We need to check if this group's last crop was resolved in the current batch
                    batch_end = batch_start + len(batch_crops)
                    if g_start <= last_flat_in_group < batch_end:
                        # This group just completed — assemble and yield its block
                        parts = flat_texts_sorted[g_start:g_end]
                        text = "\n\n".join(p for p in parts if p)
                        block_idx = g["indices"][0]
                        geom, label, crop = block_data[block_idx]
                        if crop is not None:
                            block_class = "text" if label in _TEXT_LABELS else "formula"
                            frag = (
                                f'<div id="block-{block_idx}" class="block {block_class}">\n'
                                + _lightonocr_to_html(text)
                                + '</div>\n'
                            )
                            yield ("block", {"index": block_idx, "html": frag})

    finally:
        processor.tokenizer.padding_side = orig_padding_side

    print(f"[timing] ocr_page_block_generator TOTAL: {time.perf_counter()-t_page_start:.3f}s", file=sys.stderr)
    yield ("done", {})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the VLM-based OCR export pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Export a single document page to HTML with OCR text and MathJax formulas."
    )
    parser.add_argument("input", help="Input file: image, PDF, or DjVu")
    parser.add_argument(
        "--page", type=int, default=1, metavar="N",
        help="1-based page number for PDF/DjVu (default: 1)"
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Output directory for index.html (default: <input_stem>_ocr_layout_page<N>/)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    page_0 = args.page - 1  # convert to 0-based for internal use

    # Default output dir next to the input file
    if args.output_dir is None:
        out_dir = input_path.parent / f"{input_path.stem}_ocr_layout_page{args.page}"
    else:
        out_dir = Path(args.output_dir)

    # Load page
    try:
        from document_loader import load_page
    except ImportError:
        try:
            from ocr_reflow.document_loader import load_page
        except ImportError:
            print("ERROR: document_loader not available.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading page {args.page} from {input_path}...", file=sys.stderr)
    try:
        img_bgr = load_page(str(input_path), page_0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Page size: {img_bgr.shape[1]}x{img_bgr.shape[0]} px", file=sys.stderr)

    # Copy MathJax locally so the HTML works via file:// without a running server
    mathjax_src = Path(__file__).parent / "static" / "mathjax" / "tex-svg-full.js"
    out_dir.mkdir(parents=True, exist_ok=True)
    if mathjax_src.exists():
        import shutil
        mathjax_dst = out_dir / "static" / "mathjax"
        mathjax_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mathjax_src, mathjax_dst / "tex-svg-full.js")
        local_base_url = "."
    else:
        local_base_url = "http://192.168.1.121:8000"

    # Run OCR export
    html_str = ocr_page_to_html(img_bgr, base_url=local_base_url)

    # Write HTML to output dir
    index_path = out_dir / "index.html"
    index_path.write_text(html_str, encoding="utf-8")
    print(f"HTML written to: {index_path}", file=sys.stderr)
    print(str(index_path))  # stdout: path to result
    sys.exit(0)


if __name__ == "__main__":
    # Allow running as a script from the src/ directory
    _here = Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    main()
