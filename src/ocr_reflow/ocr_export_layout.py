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
import os
import re
import sys
import time
from collections import Counter
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
    global _lightonocr_processor, _lightonocr_model

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

        logger.info("Loading LightOnOCR-2-1B...")
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
        # Enable cuDNN autotuner — picks fastest conv kernels for fixed input shapes.
        if device != "cpu":
            torch.backends.cudnn.benchmark = True
        _lightonocr_model = torch.compile(_lightonocr_model, mode="reduce-overhead")
        logger.info("LightOnOCR-2-1B loaded.")

        # Warmup torch.compile — first generate() call triggers lazy
        # compilation (30-60s cold start). A dummy forward pass pre-compiles
        # the CUDA graphs so the first real page doesn't pay this cost.
        if device != "cpu":
            logger.info("Warming up torch.compile (this may take a moment)...")
            dummy_img = Image.new("RGB", (28, 28), (255, 255, 255))
            conv = [{"role": "user", "content": [{"type": "image", "image": dummy_img}]}]
            dummy_text = _lightonocr_processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            dummy_inputs = _lightonocr_processor(
                text=[dummy_text], images=[dummy_img], padding=True, return_tensors="pt"
            )
            dummy_inputs = {
                k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                for k, v in dummy_inputs.items()
            }
            with torch.no_grad():
                _ = _lightonocr_model.generate(
                    **dummy_inputs, max_new_tokens=1,
                    do_sample=False,
                )
            logger.info("Warmup complete.")

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
    return pil_img.resize((new_w, new_h), Image.BILINEAR)


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array (OpenCV format) to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# Resized height threshold above which a plain text crop is split before OCR.
# At 1024px wide, 250px height keeps input tokens ~200 — well within the fast generation bucket.
_MAX_OCR_HEIGHT = 200
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
        logger.debug("_find_split_y (%dx%dpx): %.3fs  result=None (uniform)", crop_bgr.shape[1], crop_bgr.shape[0], time.perf_counter() - t0)
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

    logger.debug("_find_split_y (%dx%dpx): %.3fs  split_y=%s", crop_bgr.shape[1], crop_bgr.shape[0], time.perf_counter() - t0, best_y)
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
        logger.debug("_split_plain_text_crop depth=%d (%dx%dpx -> resized %dx%dpx): %.3fs  no split needed", _depth, crop_bgr.shape[1], crop_bgr.shape[0], resized.size[0], resized.size[1], time.perf_counter() - t0)
        return [crop_bgr]

    split_y = _find_split_y(crop_bgr)
    if split_y is None or split_y <= 0 or split_y >= crop_bgr.shape[0]:
        if resized.size[1] > _MAX_OCR_HEIGHT * 2:
            split_y = crop_bgr.shape[0] // 2
            logger.debug("_split_plain_text_crop depth=%d (%dx%dpx -> resized %dx%dpx): %.3fs  no gap found, force-split at midpoint y=%d", _depth, crop_bgr.shape[1], crop_bgr.shape[0], resized.size[0], resized.size[1], time.perf_counter() - t0, split_y)
        else:
            logger.debug("_split_plain_text_crop depth=%d (%dx%dpx -> resized %dx%dpx): %.3fs  no gap found, keeping as-is", _depth, crop_bgr.shape[1], crop_bgr.shape[0], resized.size[0], resized.size[1], time.perf_counter() - t0)
            return [crop_bgr]

    top = crop_bgr[:split_y, :]
    bot = crop_bgr[split_y:, :]
    logger.debug("_split_plain_text_crop depth=%d (%dx%dpx -> resized %dx%dpx): %.3fs  split at y=%d -> top=%dpx bot=%dpx", _depth, crop_bgr.shape[1], crop_bgr.shape[0], resized.size[0], resized.size[1], time.perf_counter() - t0, split_y, top.shape[0], bot.shape[0])
    return _split_plain_text_crop(top, _depth + 1) + _split_plain_text_crop(bot, _depth + 1)


def _ocr_single_crop(processor, model, device, dtype, crop, max_new_tokens: int = 512):
    """OCR a single BGR crop. Used for OOM fallback and batch_size=1."""
    import torch
    crop = _remove_dot_leaders(crop)
    pil_img = _resize_for_ocr(_bgr_to_pil(crop))
    if min(pil_img.size) < 28:
        logger.debug("OCR skip: crop too small after resize (%dx%dpx)", pil_img.size[0], pil_img.size[1])
        return "", pil_img, 0, 0
    conv = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": "Transcribe all text from this page verbatim. Line break only at paragraph boundaries. Preserve mathematical formulas in LaTeX notation."},
    ]}]
    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
              for k, v in inputs.items()}
    _gs = os.environ.get("OCR_GEN_SAMPLE", "false").lower() == "true"
    _gp = float(os.environ.get("OCR_GEN_REP_PENALTY", "1.1"))
    _gstop = os.environ.get("OCR_GEN_STOP_STRINGS", "true").lower() == "true"
    _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=_gs)
    if _gs:
        _gen_kwargs.update(temperature=0.2, top_p=0.9, top_k=0)
    if _gp > 0:
        _gen_kwargs["repetition_penalty"] = _gp
    if _gstop:
        _gen_kwargs["stop_strings"] = _HALLUCINATION_TRUNCATION_MARKERS
        _gen_kwargs["tokenizer"] = processor.tokenizer
    with torch.no_grad():
        output_ids = model.generate(**inputs, **_gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    gen = output_ids[0, input_len:]
    return processor.decode(gen, skip_special_tokens=True).strip(), pil_img, input_len, len(gen)


def _remove_dot_leaders(crop_bgr: np.ndarray) -> np.ndarray:
    """Remove TOC dot leaders (repeated dots between text and page numbers).

    Detects small compact components in horizontal rows of 5+ dots within
    200px span. Inpaints them so LightOnOCR doesn't hallucinate \\dots.
    Returns the (possibly modified) crop — no-op if no dot leaders found.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh, None, None, None, 8, cv2.CV_32S
    )

    # Group small compact components by Y row (4 px tolerance)
    rows: dict[int, list[int]] = {}
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:
            continue
        _, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        if h > 8 or w > 15:
            continue
        ar = w / max(h, 1)
        if not (0.3 <= ar <= 5.0):
            continue
        matched = False
        for ey in list(rows.keys()):
            if abs(ey - y) <= 4:
                rows[ey].append(i)
                matched = True
                break
        if not matched:
            rows[y] = [i]

    # Build mask: only keep rows with >= 5 dots in a 200 px horizontal span
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    for _y_ref, comps in rows.items():
        if len(comps) < 5:
            continue
        xs = sorted(
            stats[ci, cv2.CC_STAT_LEFT] + stats[ci, cv2.CC_STAT_WIDTH] // 2
            for ci in comps
        )
        longest = 0
        left = 0
        for right in range(len(xs)):
            while xs[right] - xs[left] > 200:
                left += 1
            longest = max(longest, right - left + 1)
        if longest >= 5:
            for ci in comps:
                mask[labels == ci] = 255

    if cv2.countNonZero(mask) == 0:
        return crop_bgr

    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(crop_bgr, dilated, 3, cv2.INPAINT_TELEA)


def _ocr_blocks_batch(crops: list, batch_size: int = 4) -> list:
    """Run LightOnOCR on a list of BGR crops using true batched generate().

    Crops are sorted by pixel area before batching to minimise padding waste.
    On CUDA OOM the failing batch falls back to sequential batch_size=1.
    Returns a list of plain-text strings in the same order as input crops.
    """
    import torch

    if not crops:
        return []

    # Remove TOC dot leaders before any OCR processing
    crops = [_remove_dot_leaders(c) for c in crops]

    processor, model = _get_lightonocr()
    t_inference_start = time.perf_counter()   # model loaded — pure inference from here
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Sort by pixel area ascending so similar-sized crops end up in the same batch
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1])
    sorted_crops = [crops[i] for i in order]
    results_sorted = [None] * len(sorted_crops)

    # Pre-scan: replace crops too small for the vision encoder with a
    # minimum-sized blank image. Pixtral's 14×14 patches + 2×2 patch merger
    # require ≥28px in both dimensions; smaller images crash unfold().
    # Cache the PIL images for reuse in the batch loop below.
    sorted_pils: list[Image.Image | None] = [None] * len(sorted_crops)
    for pos in range(len(sorted_crops)):
        c = sorted_crops[pos]
        pil = _resize_for_ocr(_bgr_to_pil(c))
        if min(pil.size) < 28:
            logger.debug("OCR skip: crop %d too small after resize (%dx%dpx)",
                         pos, pil.size[0], pil.size[1])
            sorted_crops[pos] = np.full((28, 28, 3), 255, dtype=np.uint8)
            sorted_pils[pos] = _resize_for_ocr(_bgr_to_pil(sorted_crops[pos]))
        else:
            sorted_pils[pos] = pil

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
                logger.debug(
                    "OCR batch %d (single): %dx%dpx resized=%dx%dpx "
                    "input_tokens=%d gen_tokens=%d generate=%.3fs",
                    batch_num, batch_crops[0].shape[1], batch_crops[0].shape[0],
                    pil_img.size[0], pil_img.size[1],
                    input_len, gen_len, time.perf_counter() - t_batch,
                )
                continue

            try:
                # Reuse cached PIL images from pre-scan (avoids double BGR→PIL+resize)
                pil_images = [sorted_pils[batch_start + j] for j in range(len(batch_crops))]
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
                # Configurable generate params via env vars for benchmarking.
                # Defaults match the current production settings.
                _gs = os.environ.get("OCR_GEN_SAMPLE", "true").lower() == "true"
                _gp = float(os.environ.get("OCR_GEN_REP_PENALTY", "1.1"))
                _gstop = os.environ.get("OCR_GEN_STOP_STRINGS", "false").lower() == "true"
                _gen_kwargs = dict(max_new_tokens=512, do_sample=_gs)
                if _gs:
                    _gen_kwargs.update(temperature=0.2, top_p=0.9, top_k=0)
                if _gp > 0:
                    _gen_kwargs["repetition_penalty"] = _gp
                if _gstop:
                    _gen_kwargs["stop_strings"] = _HALLUCINATION_TRUNCATION_MARKERS
                    _gen_kwargs["tokenizer"] = processor.tokenizer
                with torch.no_grad():
                    output_ids = model.generate(**inputs, **_gen_kwargs)
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

                logger.debug(
                    "OCR batch %d (%d crops): padded_input_tokens=%d gen_tokens=%s "
                    "preprocess=%.3fs generate=%.3fs total=%.3fs",
                    batch_num, len(batch_crops), input_len, gen_lens,
                    t_gen - t_batch, t_gen_done - t_gen, t_gen_done - t_batch,
                )

                # Crops that hit max_new_tokens are likely hallucinating (model
                # invents content beyond the actual text). Keep the truncated
                # output — re-running with more tokens only gives more runway.
                if hit_limit:
                    logger.warning(
                        "OCR batch %d: %d crop(s) hit max_new_tokens (512) — keeping truncated output",
                        batch_num, len(hit_limit),
                    )

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                logger.warning("OCR batch %d: CUDA OOM - falling back to batch_size=1", batch_num)
                for j, crop in enumerate(batch_crops):
                    t_fb = time.perf_counter()
                    text, pil_img, input_len, gen_len = _ocr_single_crop(
                        processor, model, device, dtype, crop
                    )
                    results_sorted[batch_start + j] = text
                    logger.debug(
                        "OCR batch %d fallback crop %d/%d: input_tokens=%d gen_tokens=%d generate=%.3fs",
                        batch_num, j + 1, len(batch_crops),
                        input_len, gen_len, time.perf_counter() - t_fb,
                    )
    finally:
        processor.tokenizer.padding_side = orig_padding_side

    # Restore original crop order
    results = [None] * len(crops)
    for sorted_i, orig_i in enumerate(order):
        results[orig_i] = results_sorted[sorted_i]

    logger.debug(
        "_ocr_blocks_batch inference only: %.3fs  (%d crops, batch_size=%d)",
        time.perf_counter() - t_inference_start, len(crops), batch_size,
    )
    return results


def _strip_hallucinated_equals(latex: str) -> str:
    r"""Strip trailing `= \d+` from \& -aligned lines (digit-count hallucination).

    The OCR model sometimes reads a digit-count column in a table as a formula
    result, producing e.g.:

        \begin{aligned}
        12.\ & 2^{126}(2^{127} - 1) = 77 \\
        ...
        \end{aligned}

    where 77 is the *number of decimal digits*, not the formula value.
    Remove `= N` when it appears at the end of a \& -aligned line whose left
    side starts with a line-number pattern (\d+.\ &) and the RHS is a small
    integer (≤ 9999).
    """
    return re.sub(
        r'(^|\n)(\d+\.\s*\\?\s*&\s*[^=\n]*?)=\s*\d{1,4}\s*((?:\\\\?)?)(?=\n|$)',
        r'\1\2\3',
        latex,
    )


def _compress_long_latex(latex: str, max_len: int = 1500) -> str:
    """Compress extremely long LaTeX (e.g. deeply nested power towers) so
    that MathJax can still render it and the text fallback stays readable."""
    if len(latex) <= max_len:
        return latex

    # Simple truncation with brace balancing
    truncated = latex[:max_len]
    opens = truncated.count('{') - truncated.count('}')
    if opens > 0:
        truncated += '}' * opens
    return truncated.strip() + '\\cdots'


def _wrap_gather(inner: str) -> str:
    """Wrap bare multi-line LaTeX (using \\\\ line breaks) in \\begin{gather}.

    If the content already has an environment (\\begin{...}) or has no line
    breaks, return it unchanged.
    """
    if "\\\\" in inner and "\\begin{" not in inner:
        return f"\\begin{{gather}}\n{inner}\n\\end{{gather}}"
    return inner


def _deduplicate_formula_numbers(text: str) -> str:
    """Remove duplicate numbered formula entries across consecutive $$...$$ blocks.

    The VLM sometimes restates the last formula number from one block as
    the first entry in the next block.  For example, block 1 ends with
    formula 8 and block 2 starts with formula 8 again (different value).
    This function keeps only the first occurrence of each formula number.
    """
    num_re = re.compile(r'(?<!\d)(\d+)\.(?:\s|\\)')
    segments = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    seen: set[int] = set()

    for i, seg in enumerate(segments):
        if seg.startswith('$$') and seg.endswith('$$'):
            inner = seg[2:-2].strip()
            lines = inner.split('\n')
            deduped = []
            for line in lines:
                m = num_re.search(line)
                if m:
                    n = int(m.group(1))
                    if n in seen:
                        continue
                    seen.add(n)
                deduped.append(line)
            if any(num_re.search(l) for l in deduped):
                segments[i] = '$$\n' + '\n'.join(deduped) + '\n$$'
            # else: leave segment unchanged (it's a formula block without
            #        equation numbers, not a hallucination)
        elif seg.strip():
            for m in num_re.finditer(seg):
                seen.add(int(m.group(1)))

    return ''.join(segments)


# Phrases that signal the VLM has stopped transcribing and started inventing
# English boilerplate. Used both as stop_strings during generation and as
# truncation markers in post-processing.
_HALLUCINATION_TRUNCATION_MARKERS = [
    "Document Title:",
    "Author(s):",
    "Abstract:",
    "Key Findings:",
    "Recommendations:",
    "Implications:",
    "In summary,",
    "This document provides",
    "This transcription preserves",
    "No LaTeX content exists outside",
    "finite difference method",
    "finite element method",
    "discontinuous equation method",
    "discontinuity equation method",
    "boundary value problem",
    "### Conclusion",
    "\n---\n",
    # Prompt echo: the model repeats the OCR instruction
    "Transcribe all text from this page verbatim",
    "Line break only at paragraph boundaries",
    "Preserve mathematical formulas in LaTeX notation",
    "The image contains no tables",
    "The rest of the page is blank",
    # Fabricated math exposition (English prose about formulas)
    "is a closed, non-empty domain",
    "the integral of a function over a region",
    "can be computed using the following formula",
    "for example, consider the function",
    "is a scalar function",
    "The boundary value problem for",
    "This shows that the integral",
    "the heat flux problem involves",
    "Heat Transfer in Geothermal",
]


# Markdown section headers characteristic of LightOnOCR's "fake academic
# document" hallucination.  These English headers never appear in the scanned
# Russian math source (which uses no Markdown at all), so two or more distinct
# ones are a high-precision signal that the whole block was invented.
_FAKE_ACADEMIC_HEADER_RE = re.compile(
    r'(?:#{1,4}|\*\*)\s*'
    r'(References|Bibliography|Footnotes?|Conclusions?|Introduction|'
    r'Literature\s+Review|Index|Examples?|Applications?|Properties|Abstract|'
    r'Acknowledge?ments?|Appendices|Appendix|Citations?|Definitions?|Overview|'
    r'Methodology|Results?|Discussions?)\b',
    re.IGNORECASE,
)


def _looks_like_fake_academic_doc(text: str) -> bool:
    """Return True if `text` is a LightOnOCR runaway 'fake academic document'.

    Two independent high-precision signals:
      * two or more distinct fake English Markdown section headers, or
      * a single LaTeX macro repeated to the point of spam (>= 10 times) while
        dominating the (short) block.
    """
    headers = {m.group(1).lower() for m in _FAKE_ACADEMIC_HEADER_RE.finditer(text)}
    if len(headers) >= 2:
        return True

    # A block whose entire content is a single bare LaTeX macro (e.g.
    # \mathcal{M}, \mathbb{R}, \alpha) is figure / decoration noise: real
    # formulas in this source always carry operators, subscripts or digits.
    bare = text.strip().strip('$').strip()
    if re.fullmatch(r'\\[a-zA-Z]+(?:\{[A-Za-z]\})?', bare):
        return True

    macros = re.findall(r'\\[a-zA-Z]+(?:\{[^{}]*\})?', text)
    if macros:
        most_common, count = Counter(macros).most_common(1)[0]
        words = max(len(text.split()), 1)
        if count >= 10 and count >= 0.20 * words:
            return True

    return False


# ---------------------------------------------------------------------------
# Robust anti-hallucination signals (content-agnostic).
#
# These return RAW scores only — no thresholds baked in.  Thresholds are
# applied by the per-block drop gate in epub_export.process_page, calibrated
# empirically against known good / bad blocks.
# ---------------------------------------------------------------------------

def _ink_ratio(crop_bgr) -> float:
    """Fraction of foreground (dark) pixels in a crop.

    A near-blank crop (figure/decoration/empty region) is the #1 VLM
    hallucination trigger.  Returns 0.0 for empty/None crops.
    """
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return 0.0
    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr
    return float((gray < 200).mean())


class _RepeatDetector:
    """Trailing character n-gram repeat counter (ported from olmOCR's
    ``repeatdetect.py``).  Detects loop hallucinations like ``ababab`` or
    ``\\mathbb \\mathbb \\mathbb`` that run to the token limit."""

    def __init__(self, s: str, max_ngram_size: int = 10):
        self.s = re.sub(r"\s+", " ", s)
        self.max_ngram_size = max_ngram_size

    def _ngram_repeats(self, n: int) -> int:
        s = self.s
        if len(s) < n * 2:
            return 0
        tail = s[-n:]
        count = 0
        i = len(s) - n
        while i >= 0 and s[i:i + n] == tail:
            count += 1
            i -= n
        return count

    def max_repeats(self) -> int:
        return max(
            (self._ngram_repeats(n) for n in range(1, self.max_ngram_size + 1)),
            default=0,
        )


def _repetition_metrics(text: str) -> dict:
    """Raw repetition / low-diversity scores for a block of OCR text.

    Keys:
      char_repeat     trailing-ngram max consecutive repeats (olmOCR style)
      n_tok           whitespace token count
      uniq_ratio      unique / total tokens (1.0 when empty)
      n_macro         count of LaTeX macro occurrences (anywhere in text)
      macro_top_count absolute count of the single most common macro NAME
                      (ignores args, so ``\\mathbb{R} \\mathbb{C} \\mathbb{N}``
                      all count as ``mathbb`` -> 3)
      macro_top_frac  most-common macro name count / total tokens
      macro_frac      fraction of all tokens that contain a macro
    """
    rd = _RepeatDetector(text)
    char_repeat = rd.max_repeats()

    toks = text.split()
    n_tok = len(toks)
    uniq_ratio = (len(set(toks)) / n_tok) if n_tok else 1.0

    # Count macros by NAME anywhere in the text (not just token-start), so that
    # ``$\mathbb{R}^2$ $\mathbb{C}$ ...`` spam with distinct args is detected.
    macro_names = re.findall(r"\\([a-zA-Z]+)", text)
    n_macro = len(macro_names)
    macro_tokens = sum(1 for t in toks if "\\" in t)
    if macro_names and n_tok:
        _, macro_top_count = Counter(macro_names).most_common(1)[0]
        macro_top_frac = macro_top_count / n_tok
        macro_frac = macro_tokens / n_tok
    else:
        macro_top_count = 0
        macro_top_frac = 0.0
        macro_frac = 0.0

    return {
        "char_repeat": char_repeat,
        "n_tok": n_tok,
        "uniq_ratio": uniq_ratio,
        "n_macro": n_macro,
        "macro_top_count": macro_top_count,
        "macro_top_frac": macro_top_frac,
        "macro_frac": macro_frac,
    }


def _strip_math_and_markup(text: str) -> str:
    """Strip LaTeX math, macros and markdown markup, leaving prose words."""
    t = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)  # bare LaTeX macros
    t = re.sub(r"[*_#`>{}\[\]\\^]", " ", t)  # markdown / latex punctuation
    return t


def _language_metrics(text: str) -> dict:
    """Detect the dominant prose language of a block.

    Keys: n_alpha_words, lang (ISO or None), conf (0..1).  Math/macros are
    stripped first so formula-only blocks report 0 alpha words.
    """
    prose = _strip_math_and_markup(text)
    words = [w for w in re.split(r"\s+", prose) if w.isalpha()]
    n_alpha_words = len(words)
    lang, conf = None, 0.0
    if n_alpha_words >= 1:
        try:
            try:
                from language_detection import classify_text  # type: ignore[no-redef]
            except ImportError:
                from ocr_reflow.language_detection import classify_text
            lang, conf = classify_text(" ".join(words))
        except Exception as e:  # pragma: no cover - detector optional
            logger.debug("language classify failed: %s", e)
    return {"n_alpha_words": n_alpha_words, "lang": lang, "conf": conf}


def block_hallucination_metrics(ocr_text: str, crop_bgr=None, expected_langs: set | None = None) -> dict:
    """Combine all raw anti-hallucination signals for one OCR block.

    Pure measurement, no decisions — used by both the calibration dump and
    the per-block drop gate.  When ``expected_langs`` is empty/None, the
    expensive fastText language classification is skipped (the result is
    only used by the wrong-language gate).
    """
    metrics = {"ink_ratio": _ink_ratio(crop_bgr), "ocr_len": len(ocr_text or "")}
    area = 0
    if crop_bgr is not None and getattr(crop_bgr, "size", 0):
        area = int(crop_bgr.shape[0]) * int(crop_bgr.shape[1])
    metrics["area"] = area
    # Output characters per pixel.  A tiny crop that emits a long string cannot
    # be a faithful transcription — it is "filler" hallucination (e.g. a 44x45px
    # region returning a page of invented example matrices).  Real blocks stay
    # under ~0.002 char/px; hallucinations run 0.2-0.6.
    metrics["char_density"] = (len(ocr_text or "") / area) if area else 0.0
    metrics.update(_repetition_metrics(ocr_text or ""))
    if expected_langs:
        metrics.update(_language_metrics(ocr_text or ""))
    else:
        metrics.update({"n_alpha_words": 0, "lang": None, "conf": 0.0})
    return metrics


def _strip_vlm_hallucinations(text: str) -> str:
    """Remove hallucinated boilerplate that LightOnOCR generates.

    Two-stage:
      1. Truncate at the first occurrence of any hallucination start marker
         (these are English boilerplate phrases that never appear in real book text).
      2. Remove known hallucinated commentary lines.
    """
    # Stage 0: detect whole-block "fake academic document" runaway.  On sparse
    # or figure-like regions LightOnOCR sometimes invents an entire English
    # Markdown paper (## References / ## Footnotes / fake citations) or spams a
    # single LaTeX macro (e.g. \mathcal{M}).  Such a block has no salvageable
    # content, so drop it entirely.
    if _looks_like_fake_academic_doc(text):
        return ""

    # Stage 1: truncate at first hallucination marker
    best_idx = len(text)
    for marker in _HALLUCINATION_TRUNCATION_MARKERS:
        idx = text.find(marker)
        if 0 < idx < best_idx:
            # Walk back to the nearest line/paragraph boundary
            trunc_at = idx
            while trunc_at > 0 and text[trunc_at - 1] in ' \t':
                trunc_at -= 1
            if trunc_at < best_idx:
                best_idx = trunc_at
    if best_idx < len(text):
        text = text[:best_idx].rstrip()

    # Stage 2: remove known hallucinated commentary lines
    _hallucinated_lines = [
        r'^Note\s*:\s*(?:(?:The|this)\s+)?image\b.*$',
        r'^Therefore\s*,?\s*(?:the\s+)?(?:Markdown|text)\s+(?:representation|output)\b.*$',
        r'^\s*with\s+no\s+additional\s+content\b.*$',
        r'^\s*the\s+letter\s+set\b.*$',
        r'^\s*No\s+additional\s+content\b.*$',
    ]
    for pattern in _hallucinated_lines:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Collapse runs of >3 consecutive identical inline math elements
    text = re.sub(
        r'(\$([^$]+)\$)(\s*\n\s*\1)+',
        lambda m: m.group(1),
        text,
    )

    # Strip truncated/incomplete inline math fragments (missing closing $).
    if text.count('$') % 2 == 1:
        text = re.sub(r'\s*\$[^$\n]*$', '', text, flags=re.MULTILINE)

    # Remove extra blank lines from stripping
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    # Detect content-agnostic repetition hallucinations (thresholds calibrated
    # against known good/bad blocks — see process_page calibration).  Three
    # independent signals, each with a wide safety margin from real content:
    #   * macro spam : one LaTeX macro name dominates a sea of distinct args
    #                  (e.g. "\mathbb{R} \mathbb{C} \mathbb{N} ..." or a run of
    #                  distinct "\frac{n}{m}") — defeats per-macro counting.
    #   * char loop  : a trailing character n-gram repeats many times.
    #   * low unique : long output with <35% unique whitespace tokens.
    rep = _repetition_metrics(text)
    if rep["macro_top_count"] >= 8 and rep["macro_top_frac"] >= 0.5:
        return ""
    if rep["char_repeat"] >= 10:
        return ""
    if rep["n_tok"] >= 50 and rep["uniq_ratio"] < 0.35:
        return ""

    return text


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

    # Strip VLM hallucinated boilerplate before any LaTeX processing
    text = _strip_vlm_hallucinations(text)

    # Remove end-of-line hyphens: "вычис-\nления" → "вычисления"
    # Only merge when the character after the newline is lowercase (real dashes
    # like "—" are mid-line and never followed by \n; proper names like
    # "Иванов-Петров" are also mid-line).
    text = re.sub(r'(\w)- *\n([а-яёa-z])', r'\1\2', text)

    # Strip entire HTML table blocks (OCR hallucination)
    text = re.sub(r'<table[^>]*>.*?</table>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # --- Phase 1: wrap bare LaTeX environments that lack ```latex fences ---
    # Temporarily mask fenced blocks so the bare-LaTeX regex doesn't match
    # content that already has proper fence markers (avoid double-$$ wrapping).
    _fence_placeholders: list[str] = []
    def _save_fence(m):
        _fence_placeholders.append(m.group(0))
        return f"\x00FENCE_{len(_fence_placeholders) - 1}\x00"
    text = re.sub(r'```latex\s*\n.*?\n\s*```', _save_fence, text, flags=re.DOTALL)

    # Fallback: a ```latex opener with no matching closing fence (malformed VLM
    # output that mixes ```latex with $$ delimiters, e.g. an aligned derivation).
    # Mask from the dangling opener to end-of-text so Phase 1.5 below does not
    # corrupt its bare commands; the Format-1 handler restores and splits it.
    text = re.sub(r'```latex\s*\n.*\Z', _save_fence, text, flags=re.DOTALL)

    def _wrap_bare_latex(m):
        inner = m.group(0).strip()
        inner = _strip_hallucinated_equals(inner)
        inner = _compress_long_latex(inner)
        return f"$$\n{inner}\n$$"
    text = re.sub(
        r'(?:^|\n)\\begin\{(aligned|array|gather|align|split|cases|equation|eqnarray)\}.*?\\end\{\1\}',
        _wrap_bare_latex,
        text,
        flags=re.DOTALL,
    )

    # NOTE: fenced blocks stay masked through Phase 1.5 below.  A ```latex fence
    # has no $ delimiters, so if it were restored here its bare commands (\ldots,
    # \frac, ...) would be treated as non-math by the Phase 1.5 segment splitter
    # and wrapped in $...$, producing malformed `$$ ... $\ldots$ ... $$` once the
    # fence is converted to display math.  Restoring after Phase 1.5 avoids this.

    # --- Phase 1.5: wrap bare LaTeX constructs that the VLM dropped $ / $$ from ---
    # When the VLM fails to wrap math in $...$ or $$...$$ delimiters, commands
    # like \left[...\right], \frac{}{}, \ldots appear as plain text.
    # These must be wrapped in $$...$$ for MathJax rendering, otherwise they
    # display as raw LaTeX source (\triangleright etc.).
    #
    # IMPORTANT: only wrap commands that are OUTSIDE existing $...$ / $$...$$
    # delimiters.  Wrapping \leq inside $0\leq n\leq9$ breaks the math.
    # Split into math / non-math segments and only wrap bare cmds in non-math.
    segments = re.split(r'(\$\$.*?\$\$|\$[^$\n]+\$)', text, flags=re.DOTALL)

    # Commands that need display math $$...$$ wrapping
    _display_bare = r'\\left\b.*?\\right\b.*?(?:\)|\]|\||\.|\\rangle|\})'
    # Commands that need inline $...$ wrapping
    _inline_bare = r'\\(?:ldots|cdots|ddots|dots|triangleright|triangleleft|infty|approx|neq|leq|geq|pm|mp|times|div|cdot|ast|star|circ|bullet|equiv|sim|simeq|propto|parallel|perp|angle|nabla|partial|forall|exists|nexists|emptyset|varnothing|in|notin|ni|subset|supset|subseteq|supseteq|cup|cap|setminus|wedge|vee|oplus|ominus|otimes|oslash|odot|bigcirc|bigcup|bigcap|bigvee|bigwedge|sum|prod|int|oint|lim|log|ln|sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|min|max|sup|inf|det|gcd|dim|hom|ker|Pr|to|mapsto|implies|iff|Rightarrow|Leftarrow|Leftrightarrow|longrightarrow|longmapsto|uparrow|downarrow)'

    for i, seg in enumerate(segments):
        if seg.startswith('$'):
            continue  # math segment — leave untouched
        # Wrap \left...\right in $$...$$
        segments[i] = re.sub(
            rf'({_display_bare})',
            r'$$\n\1\n$$', segments[i], flags=re.DOTALL,
        )
        # Wrap bare commands in $...$
        segments[i] = re.sub(
            rf'(\s|^)({_inline_bare})(\s|$|[^a-zA-Z])',
            r'\1$\2$\3', segments[i],
        )
    text = "".join(segments)

    # Restore fenced blocks (kept masked through Phase 1.5 above)
    for i, fb in enumerate(_fence_placeholders):
        text = text.replace(f"\x00FENCE_{i}\x00", fb)

    # --- Phase 2: convert ```latex ... ``` fences to $$...$$ ---
    def _fence_to_display(m):
        inner = m.group(1).strip()
        inner = _strip_hallucinated_equals(inner)
        inner = _compress_long_latex(inner)
        parts = [p.strip() for p in inner.split("$$") if p.strip()]
        return "".join(f"$$\n{_wrap_gather(p)}\n$$\n" for p in parts)
    text = re.sub(r'```latex\s*\n(.*?)\n\s*```', _fence_to_display, text, flags=re.DOTALL)

    # --- Phase 3: deduplicate formula numbers across adjacent $$...$$ blocks ---
    # The VLM sometimes restates the last formula number from one aligned block
    # as the first entry in the next block.  Keep only the first occurrence of
    # each formula number.
    text = _deduplicate_formula_numbers(text)

    # Format 1: pure ```latex\n...\n```  -> display math
    if text.startswith("```latex"):
        inner = text[len("```latex"):].strip()
        if inner.endswith("```"):
            inner = inner[:-3].strip()
        inner = _strip_hallucinated_equals(inner)
        inner = _compress_long_latex(inner)
        # Strip any stray $$ delimiters the model may have mixed in; each
        # resulting part becomes its own display block.
        parts = [p.strip() for p in inner.split("$$") if p.strip()]
        return "".join(f'<div class="formula">\n$$\n{_wrap_gather(p)}\n$$\n</div>\n' for p in parts)

    # Format 2: starts with $$...$$ — extract display math blocks from the
    # full text, treating $$-delimited segments as formulas and everything
    # else as regular text (same as Format 3 with $$ handling).
    if text.startswith("$$"):
        parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
        out_parts = []
        for p in parts:
            if p.startswith("$$") and p.endswith("$$"):
                inner = p[2:-2].strip()
                inner = _strip_hallucinated_equals(inner)
                inner = _compress_long_latex(inner)
                out_parts.append(f'<div class="formula">\n$$\n{_wrap_gather(inner)}\n$$\n</div>\n')
            else:
                stripped = p.strip()
                if stripped:
                    escaped = stripped.replace("&", "&amp;").replace("<", "&lt;")
                    escaped = escaped.replace("\n", " ")
                    out_parts.append(f"<p>{escaped}</p>\n")
        return "".join(out_parts)

    # Format 3: contains inline math or plain text
    if "$" in text:
        # Check for display math ($$) blocks — wrap each in its own
        # <div class="formula"> for proper line breaks / spacing.
        if "$$" in text:
            parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
            out_parts = []
            for p in parts:
                if p.startswith("$$") and p.endswith("$$"):
                    inner = p[2:-2].strip()
                    inner = _strip_hallucinated_equals(inner)
                    inner = _compress_long_latex(inner)
                    out_parts.append(f'<div class="formula">\n$$\n{_wrap_gather(inner)}\n$$\n</div>\n')
                else:
                    stripped = p.strip()
                    if stripped:
                        escaped = stripped.replace("&", "&amp;").replace("<", "&lt;")
                        escaped = escaped.replace("\n", " ")
                        out_parts.append(f"<p>{escaped}</p>\n")
            return "".join(out_parts)
        else:
            # Inline math only — escape bare & and <
            text = text.replace("&", "&amp;").replace("<", "&lt;")
            text = text.replace("\n", " ")
            return f"<p>{text}</p>\n"

    return f"<p>{html.escape(text).replace(chr(10), ' ')}</p>\n"


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

    logger.info("Running layout analysis...")
    t_layout = time.perf_counter()
    blocks = layout_from_array(img_bgr)  # list of (shapely_geom, label_str)
    t_layout_done = time.perf_counter()
    logger.debug("layout analysis: %.3fs  blocks=%d", t_layout_done - t_layout, len(blocks))

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
    # NOTE: table labels are also excluded — tables contain text that should be OCR'd.
    _FIGURE_MASK_LABELS = {"figure", "figure_and_caption", "figure_caption"}
    image_bboxes = []  # list of (x1, y1, x2, y2) in page coords
    for geom, label in blocks:
        if label in _FIGURE_MASK_LABELS:
            x1, y1, x2, y2 = geom.bounds
            image_bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    def _mask_image_regions(crop: np.ndarray, block_geom) -> np.ndarray:
        """White-fill any image/formula bbox that overlaps this block's crop."""
        bx1, by1, bx2, by2 = [int(v) for v in block_geom.bounds]
        bw, bh = bx2 - bx1, by2 - by1
        masked = crop.copy()
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
    logger.debug("merge: %.3fs  %d text blocks -> %d merged groups, %d OCR calls saved",
                 t_merge_done - t_merge, n_text_blocks, n_merged_groups, n_calls_saved)

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
    logger.debug("crop+mask: %.3fs  (%d groups)", t_crop_done - t_crop_done_start, len(ocr_group_crops))

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
    logger.debug("split: %.3fs  (%d groups -> %d crops, %d extra splits)",
                 t_split_done - t_split, len(ocr_groups), len(ocr_flat_crops), n_splits)

    logger.info("Running batch OCR on %d crops (%d groups, %d extra splits)...",
                len(ocr_flat_crops), len(ocr_groups), n_splits)
    t_ocr = time.perf_counter()
    flat_texts = _ocr_blocks_batch(ocr_flat_crops)
    t_ocr_done = time.perf_counter()
    logger.debug("OCR total: %.3fs", t_ocr_done - t_ocr)

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
        logger.info("Block %d/%d: %s", i + 1, len(block_data), label)

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
    logger.debug("html assembly: %.3fs", t_html_done - t_html)
    logger.debug("ocr_page_to_html TOTAL: %.3fs", t_html_done - t_page_start)
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

    logger.info("Running layout analysis...")
    t_layout = time.perf_counter()
    blocks = layout_from_array(img_bgr)
    t_layout_done = time.perf_counter()
    logger.debug("layout analysis: %.3fs  blocks=%d", t_layout_done - t_layout, len(blocks))

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
    logger.debug("crop+mask: %.3fs  (%d blocks)", t_crop_done - t_crop, len(ocr_indices))

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
    logger.debug("split: %.3fs  (%d blocks -> %d crops, %d extra splits)",
                 t_split_done - t_split, len(ocr_indices), len(ocr_flat_crops), n_splits)

    logger.info("Running sequential OCR on %d crops...", len(ocr_flat_crops))
    t_ocr = time.perf_counter()
    flat_texts = _ocr_blocks_batch(ocr_flat_crops, batch_size=1)
    t_ocr_done = time.perf_counter()
    logger.debug("OCR total: %.3fs", t_ocr_done - t_ocr)

    ocr_results = {}
    flat_idx = 0
    for idx, count in enumerate(ocr_split_counts):
        block_idx = ocr_indices[idx]
        parts = flat_texts[flat_idx:flat_idx + count]
        ocr_results[block_idx] = "\n\n".join(p for p in parts if p)
        flat_idx += count

    for i, (geom, label, crop) in enumerate(block_data):
        logger.info("Block %d/%d: %s", i + 1, len(block_data), label)
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
    logger.debug("ocr_page_to_html_simple TOTAL: %.3fs", t_done - t_page_start)
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

    logger.info("Running layout analysis...")
    t_layout = time.perf_counter()
    blocks = layout_from_array(img_bgr)
    t_layout_done = time.perf_counter()
    logger.debug("layout analysis: %.3fs  blocks=%d", t_layout_done - t_layout, len(blocks))

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
                logger.debug(
                    "OCR batch %d (single): input_tokens=%d gen_tokens=%d generate=%.3fs",
                    batch_num, input_len, gen_len, time.perf_counter() - t_batch,
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
                    _gs = os.environ.get("OCR_GEN_SAMPLE", "false").lower() == "true"
                    _gp = float(os.environ.get("OCR_GEN_REP_PENALTY", "1.1"))
                    _gstop = os.environ.get("OCR_GEN_STOP_STRINGS", "true").lower() == "true"
                    _gen_kwargs = dict(max_new_tokens=512, do_sample=_gs)
                    if _gs:
                        _gen_kwargs.update(temperature=0.2, top_p=0.9, top_k=0)
                    if _gp > 0:
                        _gen_kwargs["repetition_penalty"] = _gp
                    if _gstop:
                        _gen_kwargs["stop_strings"] = _HALLUCINATION_TRUNCATION_MARKERS
                        _gen_kwargs["tokenizer"] = processor.tokenizer
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **_gen_kwargs)
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
                    logger.debug(
                        "OCR batch %d (%d crops): gen_tokens=%s "
                        "preprocess=%.3fs generate=%.3fs total=%.3fs",
                        batch_num, len(batch_crops), gen_lens,
                        t_gen - t_batch, t_gen_done - t_gen, t_gen_done - t_batch,
                    )
                    if hit_limit:
                        logger.warning(
                            "OCR batch %d: %d crop(s) hit max_new_tokens (512) — keeping truncated output",
                            batch_num, len(hit_limit),
                        )
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    logger.warning("OCR batch %d: CUDA OOM - falling back to batch_size=1", batch_num)
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

    logger.debug("ocr_page_block_generator TOTAL: %.3fs", time.perf_counter() - t_page_start)
    yield ("done", {})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the VLM-based OCR export pipeline."""
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
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to log file (logs written to stderr by default).",
    )
    args = parser.parse_args()
    from ocr_reflow.log_setup import setup_logging
    setup_logging(log_path=args.log_file)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
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
            logger.error("document_loader not available.")
            sys.exit(1)

    logger.info("Loading page %d from %s...", args.page, input_path)
    try:
        img_bgr = load_page(str(input_path), page_0)
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)

    logger.info("Page size: %dx%d px", img_bgr.shape[1], img_bgr.shape[0])

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
    logger.info("HTML written to: %s", index_path)
    print(str(index_path))  # stdout: path to result (machine-readable)
    sys.exit(0)


if __name__ == "__main__":
    # Allow running as a script from the src/ directory
    _here = Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    main()
