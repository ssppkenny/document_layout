"""OCR export: run layout analysis on a single page and produce an HTML string.

Usage (CLI):
    python ocr_export.py INPUT [--page N] [--output-dir DIR]

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
import sys
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
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_lightonocr():
    global _lightonocr_processor, _lightonocr_model
    if _lightonocr_model is None:
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
        print("LightOnOCR-2-1B loaded.", file=sys.stderr)
    return _lightonocr_processor, _lightonocr_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_OCR_SIDE = 1024
_MAX_OCR_HEIGHT = 400  # px after resize; taller crops are split into halves


def _split_tall_crop(crop_bgr: np.ndarray) -> list:
    """Split a BGR crop that is too tall for LightOnOCR into vertical halves.

    Recursively splits until all parts are within _MAX_OCR_HEIGHT after resize.
    Returns a list of one or more BGR arrays.
    """
    pil_check = _resize_for_ocr(_bgr_to_pil(crop_bgr))
    if pil_check.height <= _MAX_OCR_HEIGHT:
        return [crop_bgr]
    h = crop_bgr.shape[0]
    mid = h // 2
    top = crop_bgr[:mid, :]
    bot = crop_bgr[mid:, :]
    print(f"  [split] crop {crop_bgr.shape} → top {top.shape} + bot {bot.shape}", file=sys.stderr)
    return _split_tall_crop(top) + _split_tall_crop(bot)


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
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)



def _ocr_blocks_batch(crops: list, batch_size: int = 8) -> list:
    """Run LightOnOCR on a list of BGR crops sequentially with per-block timing.

    Returns a list of plain-text strings in the same order as crops.
    """
    import torch
    import time

    if not crops:
        return []

    processor, model = _get_lightonocr()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_total = time.perf_counter()
    results = []
    for i, crop in enumerate(crops):
        t0 = time.perf_counter()
        pil_img = _resize_for_ocr(_bgr_to_pil(crop))
        t_resize = time.perf_counter()
        conv = [{"role": "user", "content": [{"type": "image", "image": pil_img}]}]
        inputs = processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        t_proc = time.perf_counter()
        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        t_to_dev = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)
        t_gen = time.perf_counter()
        input_len = inputs["input_ids"].shape[1]
        gen = output_ids[0, input_len:]
        text = processor.decode(gen, skip_special_tokens=True).strip()
        n_tokens = len(gen)
        print(
            f"  [ocr] block {i+1}/{len(crops)}: total={time.perf_counter()-t0:.2f}s"
            f"  resize={t_resize-t0:.2f}s  proc={t_proc-t_resize:.2f}s"
            f"  to_dev={t_to_dev-t_proc:.2f}s  gen={t_gen-t_to_dev:.2f}s"
            f"  tokens={n_tokens}  size={pil_img.size}",
            file=sys.stderr
        )
        results.append(text)

    print(f"  [ocr] all blocks total: {time.perf_counter()-t_total:.2f}s", file=sys.stderr)
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

    # Format 1: ```latex\n...\n```  -> display math
    if text.startswith("```latex"):
        inner = text[len("```latex"):].strip()
        if inner.endswith("```"):
            inner = inner[:-3].strip()
        # Strip any stray $$ delimiters the model may have mixed in; each
        # resulting part becomes its own display block.
        parts = [p.strip() for p in inner.split("$$") if p.strip()]
        return "".join(f"<p>$$\n{_wrap_gather(p)}\n$$</p>\n" for p in parts)

    # Format 2: $$...$$ — strip all $$ delimiters; each segment becomes its
    # own display block so multi-line formulas render on separate lines.
    if text.startswith("$$"):
        parts = [p.strip() for p in text.split("$$") if p.strip()]
        return "".join(f"<p>$$\n{_wrap_gather(p)}\n$$</p>\n" for p in parts)

    # Format 3: contains inline math or plain text
    # If any $ present, pass raw (MathJax handles it); otherwise html-escape
    if "$" in text:
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
  body {{ font-family: serif; max-width: 100%; margin: 0; padding: 1em 1.4em; box-sizing: border-box; line-height: 1.6; }}
  img {{ max-width: 100%; display: block; margin: 1em 0; }}
  p {{ margin: 0.8em 0; }}
  mjx-container[display="true"] {{ display: block; max-width: 100%; overflow-x: auto; }}
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
_FORMULA_LABELS = {"isolate_formula_and_caption"}

# Labels treated as images (save crop as embedded image)
_IMAGE_LABELS = {
    "figure", "figure_and_caption", "figure_caption",
    "table", "table_caption", "table_footnote",
    "isolate_formula",
}

# Labels to skip
_SKIP_LABELS = {"abandon"}

# All labels that go through LightOnOCR
_OCR_LABELS = _TEXT_LABELS | _FORMULA_LABELS


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
    # --- Layout analysis ---
    try:
        from layout import layout_from_array
    except ImportError:
        from ocr_reflow.layout import layout_from_array

    print("Running layout analysis...", file=sys.stderr)
    blocks = layout_from_array(img_bgr)  # list of (shapely_geom, label_str)
    print(f"  {len(blocks)} blocks detected.", file=sys.stderr)

    # Sort blocks top-to-bottom so HTML order matches page reading order
    blocks = sorted(blocks, key=lambda b: b[0].bounds[1])

    # --- Process blocks ---
    html_parts = [_html_head(base_url)]

    # First pass: crop all blocks
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

    # Collect image/formula bboxes (in page coordinates) to mask out of text crops.
    # This prevents figure content from bleeding into OCR input when a low-confidence
    # plain text block overlaps a figure region.
    image_bboxes = []  # list of (x1, y1, x2, y2) in page coords
    for geom, label in blocks:
        if label in _IMAGE_LABELS | _FORMULA_LABELS:
            x1, y1, x2, y2 = geom.bounds
            image_bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    # Batch-OCR all text and formula blocks in one pass
    ocr_indices = [i for i, (_, label, crop) in enumerate(block_data)
                   if label in _OCR_LABELS and crop is not None]

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

    ocr_crops_raw = [
        _mask_image_regions(block_data[i][2], block_data[i][0])
        for i in ocr_indices
    ]

    # Split tall crops into halves; track mapping back to original block index
    ocr_crops = []       # flat list of crops to OCR
    crop_map = []        # maps flat index → (block_idx_in_ocr_indices, part_index)
    for raw_i, crop in enumerate(ocr_crops_raw):
        parts = _split_tall_crop(crop)
        for part_i, part in enumerate(parts):
            ocr_crops.append(part)
            crop_map.append((raw_i, part_i, len(parts)))

    print(f"  Running batch OCR on {len(ocr_crops)} blocks...", file=sys.stderr)
    ocr_texts_flat = _ocr_blocks_batch(ocr_crops)

    # Reassemble split results: join halves with a space
    ocr_texts = []
    i = 0
    while i < len(crop_map):
        raw_i, part_i, n_parts = crop_map[i]
        if n_parts == 1:
            ocr_texts.append(ocr_texts_flat[i])
            i += 1
        else:
            parts_text = [ocr_texts_flat[i + j] for j in range(n_parts)]
            ocr_texts.append(" ".join(t for t in parts_text if t))
            i += n_parts

    ocr_results = dict(zip(ocr_indices, ocr_texts))

    # Second pass: build HTML in reading order
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

    html_parts.append(_HTML_TAIL)
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
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
        help="Output directory for index.html (default: <input_stem>_ocr_page<N>/)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    page_0 = args.page - 1  # convert to 0-based for internal use

    # Default output dir next to the input file
    if args.output_dir is None:
        out_dir = input_path.parent / f"{input_path.stem}_ocr_page{args.page}"
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

    # Run OCR export
    html_str = ocr_page_to_html(img_bgr)

    # Write HTML to output dir
    out_dir.mkdir(parents=True, exist_ok=True)
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
