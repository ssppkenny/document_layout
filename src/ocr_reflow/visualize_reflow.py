"""
Visualization of word detection and letter segmentation for a single page.

Left panel:  source page with layout block boundaries, word boxes (cyan),
             and letter boxes (green) overlaid.
Right panel: reflowed output page with placed letter boxes (green),
             zero-region skips (red), zero-size skips (orange),
             and clipped letters (yellow) overlaid.

Usage:
    pixi run python -m ocr_reflow.visualize_reflow
    pixi run python -m ocr_reflow.visualize_reflow --pdf books/algorithms.pdf --page 16 --out /tmp/opencode/viz.png
"""

import argparse
import os
import sys
import tempfile
from math import ceil
from operator import itemgetter

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
BLOCK_COLORS = {
    "plain text":         (200,  80,   0),   # blue
    "title":              (  0, 140, 255),   # orange
    "titled_block_title": (200,   0, 200),   # purple
    "titled_block_body":  (180,   0, 180),   # magenta
}
BLOCK_COLOR_DEFAULT = (120, 120, 120)        # gray for figures/tables/formulas

WORD_COLOR   = (200, 200,   0)   # cyan
LETTER_COLOR = (  0, 200,   0)   # green

PLACED_COLOR     = (  0, 200,   0)   # green
ZERO_REG_COLOR   = (  0,   0, 220)   # red
ZERO_SIZE_COLOR  = (  0, 140, 255)   # orange
CLIPPED_COLOR    = (  0, 220, 220)   # yellow


def _draw_rect(img, x, y, w, h, color, thickness=1):
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def build_left_panel(img, layout_boxes_sorted, model, DocumentFile,
                     find_rects, margins, merge_close_lines, Letter,
                     apply_binarization=False):
    """
    Draw layout blocks, word boxes, and letter boxes onto a copy of `img`.
    Returns the annotated image.
    """
    from shapely.geometry import box as shapely_box, LineString

    vis = img.copy()
    img_h, img_w = img.shape[:2]

    text_types = {"plain text", "title", "titled_block_title", "titled_block_body"}

    for box_geom, box_type in layout_boxes_sorted:
        bounds = box_geom.bounds
        xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])

        block_color = BLOCK_COLORS.get(box_type, BLOCK_COLOR_DEFAULT)

        # Draw block boundary (thick)
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), block_color, 3)

        # Label block type
        label = box_type[:18]
        cv2.putText(vis, label, (xmin + 4, ymin + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, block_color, 1, cv2.LINE_AA)

        if box_type not in text_types:
            continue

        # Crop box
        box_img = img[ymin:ymax, xmin:xmax].copy()
        box_h, box_w = box_img.shape[:2]
        if box_h == 0 or box_w == 0:
            continue

        # Run OCR on box
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, box_img)
        try:
            docs = DocumentFile.from_images([tmp_path])
            result = model(docs)
        finally:
            os.unlink(tmp_path)

        words_raw = result[0]["words"]
        if len(words_raw) == 0:
            continue

        padding = 35 if box_type == "title" else 5
        words = words_raw.copy()
        words[:, 0] = np.clip((words[:, 0] * box_w).astype(np.int32) - padding, 0, box_w)
        words[:, 1] = np.clip((words[:, 1] * box_h).astype(np.int32) - padding, 0, box_h)
        words[:, 2] = np.clip((words[:, 2] * box_w).astype(np.int32) + padding, 0, box_w)
        words[:, 3] = np.clip((words[:, 3] * box_h).astype(np.int32) + padding, 0, box_h)
        words = words.astype(np.int32)

        # For titles: merge into one box
        if box_type == "title" and len(words) > 1:
            extra = 20
            mx0 = max(0, int(np.min(words[:, 0])) - extra)
            my0 = max(0, int(np.min(words[:, 1])) - extra)
            mx1 = min(box_w, int(np.max(words[:, 2])) + extra)
            my1 = min(box_h, int(np.max(words[:, 3])) + extra)
            words = np.array([[mx0, my0, mx1, my1]], dtype=np.int32)

        # Draw word boxes (translated to full-image coords)
        for wx0, wy0, wx1, wy1, *_ in words:
            _draw_rect(vis, xmin + wx0, ymin + wy0, wx1 - wx0, wy1 - wy0, WORD_COLOR, 1)

        # Get letter boxes via find_rects
        word_list = [(int(w[0]), int(w[1]), int(w[2]), int(w[3])) for w in words]
        letters_raw = find_rects(box_img, word_list, use_binarization=apply_binarization)

        for lx0, ly0, lx1, ly1 in letters_raw:
            _draw_rect(vis, xmin + lx0, ymin + ly0, lx1 - lx0, ly1 - ly0, LETTER_COLOR, 1)

    return vis


def build_right_panel(img, layout_boxes_sorted, model, DocumentFile,
                      find_rects, margins, merge_close_lines, Letter,
                      create_page_with_word_wrapping,
                      new_page_width, zoom_factor, background_color,
                      apply_binarization=False):
    """
    Run reflow with debug=True and overlay placement records on the output.
    Returns the annotated reflowed image.
    """
    from shapely.geometry import box as shapely_box, LineString

    img_h, img_w = img.shape[:2]

    # Compute margins the same way as process_document_with_layout
    text_geoms = [g for g, t in layout_boxes_sorted if t in ("plain text", "title")]
    if text_geoms:
        min_xmin = min(g.bounds[0] for g in text_geoms)
        max_xmax = max(g.bounds[2] for g in text_geoms)
        left_margin  = max(1, int((min_xmin / img_w) * new_page_width))
        right_margin = max(1, int(((img_w - max_xmax) / img_w) * new_page_width))
    else:
        left_margin  = int(new_page_width * 0.025)
        right_margin = int(new_page_width * 0.025)

    # Start with a large canvas; we'll crop later
    canvas_h = 8000
    canvas = np.ones((canvas_h, new_page_width, 3), dtype=np.uint8)
    canvas[:] = background_color

    current_y = 0
    min_gap = max(1, int(img_h * zoom_factor * 0.003))

    all_debug_records = []

    text_block_types = {"plain text", "title"}

    for idx, (box_geom, box_type) in enumerate(layout_boxes_sorted):
        if box_type not in text_block_types:
            continue

        bounds = box_geom.bounds
        xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])

        prev_ymax = layout_boxes_sorted[idx - 1][0].bounds[3] if idx > 0 else 0
        next_ymin = layout_boxes_sorted[idx + 1][0].bounds[1] if idx < len(layout_boxes_sorted) - 1 else img_h
        gap_before = max(min_gap, int((bounds[1] - prev_ymax) * zoom_factor))
        gap_after  = max(min_gap, int((next_ymin - bounds[3]) * zoom_factor))

        box_img = img[ymin:ymax, xmin:xmax].copy()
        box_h, box_w = box_img.shape[:2]
        if box_h == 0 or box_w == 0:
            continue

        # OCR
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, box_img)
        try:
            docs = DocumentFile.from_images([tmp_path])
            result = model(docs)
        finally:
            os.unlink(tmp_path)

        words_raw = result[0]["words"]
        if len(words_raw) == 0:
            continue

        padding = 35 if box_type == "title" else 5
        words = words_raw.copy()
        words[:, 0] = np.clip((words[:, 0] * box_w).astype(np.int32) - padding, 0, box_w)
        words[:, 1] = np.clip((words[:, 1] * box_h).astype(np.int32) - padding, 0, box_h)
        words[:, 2] = np.clip((words[:, 2] * box_w).astype(np.int32) + padding, 0, box_w)
        words[:, 3] = np.clip((words[:, 3] * box_h).astype(np.int32) + padding, 0, box_h)
        words = words.astype(np.int32)

        if box_type == "title" and len(words) > 1:
            extra = 20
            mx0 = max(0, int(np.min(words[:, 0])) - extra)
            my0 = max(0, int(np.min(words[:, 1])) - extra)
            mx1 = min(box_w, int(np.max(words[:, 2])) + extra)
            my1 = min(box_h, int(np.max(words[:, 3])) + extra)
            words = np.array([[mx0, my0, mx1, my1]], dtype=np.int32)

        # Build lines + letters
        lm_list, rm_list = margins(words)
        lm_list, rm_list = merge_close_lines(lm_list, rm_list, words, y_threshold=30)
        if not lm_list:
            continue

        rectangles = {
            shapely_box(w[0], w[1], w[2], w[3]): (int(w[0]), int(w[1]), int(w[2]), int(w[3]))
            for w in words
        }
        lines = []
        for l, r in zip(lm_list, rm_list):
            line_geom = LineString([(l[0], l[1]), (r[0], r[1])])
            line_words = [rectangles[b] for b in rectangles if line_geom.intersects(b)]
            if line_words:
                lines.append(sorted(line_words))

        all_lines = []
        for line in lines:
            line_letters = find_rects(box_img, line, use_binarization=apply_binarization)
            line_letters = sorted(line_letters, key=itemgetter(0))
            if not line_letters:
                continue
            heights = [ly2 - ly1 for lx1, ly1, lx2, ly2 in line_letters]
            m_h = np.median(heights)
            sd = np.std(heights) if len(heights) > 1 else 0
            normal = [(lx1, ly1, lx2, ly2) for lx1, ly1, lx2, ly2 in line_letters
                      if abs((ly2 - ly1) - m_h) < sd]
            if len(normal) > 1:
                try:
                    xs = [(lx1 + lx2) / 2 for lx1, ly1, lx2, ly2 in normal]
                    ys = [ly2 for lx1, ly1, lx2, ly2 in normal]
                    m, c = np.polyfit(xs, ys, 1)
                except Exception:
                    m, c = 0, 0
            else:
                m, c = 0, 0
            letters = [
                Letter(lx1, ly1, lx2, ly2, ly2 - ceil(m * ((lx1 + lx2) / 2) + c))
                for lx1, ly1, lx2, ly2 in line_letters
            ]
            all_lines.append(letters)

        if not all_lines:
            continue

        # Reflow with debug=True
        result_page, records = create_page_with_word_wrapping(
            all_lines, box_img, zoom_factor, new_page_width,
            left_margin=left_margin, right_margin=right_margin,
            top_margin=0, bottom_margin=0,
            background_color=tuple(background_color),
            is_title=(box_type == "title"),
            debug=True,
        )

        # Find actual content height
        temp_h = result_page.shape[0]
        content_height = temp_h
        for row in range(temp_h - 1, -1, -1):
            if not np.all(result_page[row] == np.array(background_color)):
                content_height = row + 1
                break

        # Place on canvas
        current_y += gap_before
        required = current_y + content_height + 50
        if required > canvas.shape[0]:
            extra_canvas = np.ones((required + 2000, new_page_width, 3), dtype=np.uint8)
            extra_canvas[:] = background_color
            extra_canvas[:canvas.shape[0]] = canvas
            canvas = extra_canvas

        canvas[current_y:current_y + content_height, :] = result_page[:content_height, :]

        # Shift records by current_y and collect
        for rec in records:
            all_debug_records.append({**rec, 'y': rec['y'] + current_y})

        current_y += content_height + gap_after

    # Crop canvas
    if layout_boxes_sorted:
        last_ymax = layout_boxes_sorted[-1][0].bounds[3]
        bottom_margin_px = max(min_gap, int((img_h - last_ymax) * zoom_factor))
    else:
        bottom_margin_px = int(new_page_width * 0.025)
    final_h = min(current_y + bottom_margin_px, canvas.shape[0])
    output = canvas[:final_h].copy()

    # Draw placement records
    for rec in all_debug_records:
        x, y, w, h, status = rec['x'], rec['y'], rec['w'], rec['h'], rec['status']
        if status == 'placed':
            color = PLACED_COLOR
        elif status == 'zero_region':
            color = ZERO_REG_COLOR
        elif status == 'zero_size':
            color = ZERO_SIZE_COLOR
        else:  # clipped
            color = CLIPPED_COLOR
        _draw_rect(output, x, y, w, h, color, 1)

    # Print summary
    from collections import Counter
    counts = Counter(r['status'] for r in all_debug_records)
    print(f"\n[Right panel] Letter placement summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status:15s}: {count}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Visualize word/letter segmentation and reflow")
    parser.add_argument('--pdf',  default='books/algorithms.pdf')
    parser.add_argument('--page', type=int, default=16, help='0-based page index')
    parser.add_argument('--out',  default='/tmp/opencode/viz_reflow_p16.png')
    parser.add_argument('--page-width', type=int, default=2000)
    parser.add_argument('--zoom',       type=float, default=2.5)
    args = parser.parse_args()

    # Imports from the package
    try:
        from .document_loader import load_page
        from .layout import layout as analyze_layout
        from .main import find_rects, margins, merge_close_lines, Letter, get_doctr_model
        from .reflow import create_page_with_word_wrapping
    except ImportError:
        from document_loader import load_page
        from layout import layout as analyze_layout
        from main import find_rects, margins, merge_close_lines, Letter, get_doctr_model
        from reflow import create_page_with_word_wrapping

    from doctr.io import DocumentFile

    model, device = get_doctr_model()
    background_color = np.array([220, 220, 220], dtype=np.uint8)

    print(f"Loading page {args.page} from {args.pdf} ...")
    img = load_page(args.pdf, args.page)

    print("Running layout detection ...")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)
    try:
        layout_boxes = analyze_layout(tmp_path)
    finally:
        os.unlink(tmp_path)

    layout_boxes_sorted = sorted(layout_boxes, key=lambda item: (item[0].bounds[1], item[0].bounds[0]))
    print(f"Found {len(layout_boxes_sorted)} layout blocks:")
    for g, t in layout_boxes_sorted:
        b = g.bounds
        print(f"  {t:25s}  y={int(b[1]):4d}→{int(b[3]):4d}  x={int(b[0]):4d}→{int(b[2]):4d}")

    print("\nBuilding left panel (source + segmentation) ...")
    left = build_left_panel(
        img, layout_boxes_sorted, model, DocumentFile,
        find_rects, margins, merge_close_lines, Letter,
        apply_binarization=False,
    )

    print("Building right panel (reflowed + placement overlay) ...")
    right = build_right_panel(
        img, layout_boxes_sorted, model, DocumentFile,
        find_rects, margins, merge_close_lines, Letter,
        create_page_with_word_wrapping,
        new_page_width=args.page_width,
        zoom_factor=args.zoom,
        background_color=background_color,
        apply_binarization=False,
    )

    # Scale both panels to the same height for side-by-side display
    h_left  = left.shape[0]
    h_right = right.shape[0]
    target_h = max(h_left, h_right)

    def pad_height(panel, target):
        if panel.shape[0] >= target:
            return panel
        pad = np.ones((target - panel.shape[0], panel.shape[1], 3), dtype=np.uint8) * 220
        return np.vstack([panel, pad])

    left  = pad_height(left,  target_h)
    right = pad_height(right, target_h)

    # Add a thin separator
    sep = np.ones((target_h, 4, 3), dtype=np.uint8) * 80

    combined = np.hstack([left, sep, right])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, combined)
    print(f"\nSaved visualization to {args.out}  (size: {combined.shape[1]}x{combined.shape[0]})")

    # Legend
    print("\nLegend:")
    print("  Left panel:")
    print("    Thick colored rect = layout block boundary (blue=plain text, orange=title, purple/magenta=titled_block)")
    print("    Thin CYAN rect     = word box (DocTR OCR)")
    print("    Thin GREEN rect    = letter box (find_rects segmentation)")
    print("  Right panel:")
    print("    GREEN  rect = letter placed successfully")
    print("    RED    rect = letter skipped (zero-size region in source)")
    print("    ORANGE rect = letter skipped (zero scaled width/height)")
    print("    YELLOW rect = letter clipped past right margin")


if __name__ == '__main__':
    main()
