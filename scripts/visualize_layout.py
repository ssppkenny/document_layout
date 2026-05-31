"""Visualize raw YOLO detections vs. grouped layout blocks side by side.

Usage:
    python visualize_layout.py <file> [--page N] [--output OUTPUT]

Examples:
    python visualize_layout.py books/algorithms.pdf --page 16
    python visualize_layout.py books/dvurog.djvu --page 5 --output debug.png
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ocr_reflow.document_loader import load_page
from ocr_reflow.layout import get_yolo_model, find_grouped_bounding_boxes, _CACHED_YOLO_DEVICE
from shapely.geometry import box as shapely_box

# ---------------------------------------------------------------------------
# Color palette — BGR, one per block type
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    "plain text":                    (34,  139,  34),   # forest green
    "title":                         (148,   0, 211),   # purple
    "figure":                        (0,    0,  220),   # red
    "figure_and_caption":            (0,    0,  180),   # dark red
    "figure_caption":                (0,   80,  200),   # red-orange
    "isolate_formula":               (220,  0,    0),   # blue
    "isolate_formula_and_caption":   (180,  0,    0),   # dark blue
    "formula_caption":               (200, 80,    0),   # blue-ish
    "table":                         (0,  165,  255),   # orange
    "table_and_caption":             (0,  140,  220),   # dark orange
    "table_caption":                 (0,  200,  255),   # light orange
    "table_footnote":                (0,  220,  200),   # yellow-orange
    "abandon":                       (120, 120, 120),   # gray
}
DEFAULT_COLOR = (200, 200, 0)   # cyan for unknown types


def color_for(label):
    return TYPE_COLORS.get(label, DEFAULT_COLOR)


def draw_boxes(img, boxes_and_types, alpha=0.25, font_scale=0.55, thickness=2):
    """Draw semi-transparent filled boxes with labels onto a copy of img.

    Args:
        img: BGR numpy array
        boxes_and_types: list of (shapely_geometry, label_str)
        alpha: opacity of the filled overlay (0=transparent, 1=opaque)
        font_scale: cv2 font scale for labels
        thickness: border line thickness

    Returns:
        Annotated BGR numpy array
    """
    out = img.copy()
    overlay = img.copy()

    for i, (geom, label) in enumerate(boxes_and_types):
        x1, y1, x2, y2 = (int(round(v)) for v in geom.bounds)
        color = color_for(label)

        # Filled rectangle on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # Border on output
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # Label background + text
        text = f"{i}: {label}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        lx, ly = x1 + 3, y1 + th + 3
        cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + baseline), color, -1)
        cv2.putText(
            out, text, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Blend overlay
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def legend_strip(types_present, cell_h=28, cell_w=220):
    """Create a horizontal legend image for the types present."""
    unique = sorted(set(types_present))
    n = len(unique)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    strip = np.full((rows * cell_h, cols * cell_w, 3), 240, dtype=np.uint8)

    for idx, label in enumerate(unique):
        row, col = divmod(idx, cols)
        x = col * cell_w
        y = row * cell_h
        color = color_for(label)
        cv2.rectangle(strip, (x + 4, y + 4), (x + 22, y + cell_h - 4), color, -1)
        cv2.putText(
            strip, label, (x + 28, y + cell_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA
        )
    return strip


def add_title_bar(img, text, bar_h=36):
    bar = np.full((bar_h, img.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(
        bar, text, (10, bar_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA
    )
    return np.vstack([bar, img])


def main():
    parser = argparse.ArgumentParser(description="Visualize layout detection blocks side by side")
    parser.add_argument("file", help="Input file (image, PDF, or DjVu)")
    parser.add_argument("--page", type=int, default=0, metavar="N",
                        help="0-based page number (default: 0)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: layout_debug_p<N>.png)")
    args = parser.parse_args()

    out_path = args.output or f"layout_debug_p{args.page}.png"

    # -----------------------------------------------------------------------
    # 1. Load the page
    # -----------------------------------------------------------------------
    print(f"Loading page {args.page} from '{args.file}' ...")
    img_bgr = load_page(args.file, args.page)
    print(f"  Page size: {img_bgr.shape[1]}x{img_bgr.shape[0]} px")

    # Write to temp file — YOLO and layout() need a file path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        # -------------------------------------------------------------------
        # 2. Run YOLO and get RAW detections
        # -------------------------------------------------------------------
        print("Running YOLO detection ...")
        model = get_yolo_model()
        if model is None:
            print("ERROR: YOLO model not available.")
            sys.exit(1)

        from ocr_reflow.layout import _CACHED_YOLO_DEVICE
        from ocr_reflow.device_utils import get_device_for_yolo
        device = _CACHED_YOLO_DEVICE or get_device_for_yolo(model)

        det_res = model.predict(tmp_path, imgsz=1024, conf=0.2, device=device)

        names = det_res[0].names
        raw_labels = [names[int(n)] for n in det_res[0].boxes.cls]
        raw_confs  = [float(c) for c in det_res[0].boxes.conf]
        raw_xyxy   = [a.tolist() for a in det_res[0].boxes.xyxy]

        img_h, img_w = img_bgr.shape[:2]

        # Build raw shapely boxes (same expansion logic as layout.py)
        raw_boxes = []
        for i, (x1, y1, x2, y2) in enumerate(raw_xyxy):
            if raw_labels[i] == "plain text":
                minx = max(0, min(x1, x2) - 5)
                maxx = min(img_w, max(x1, x2) + 5)
                miny = max(0, min(y1, y2) - 5)
                maxy = min(img_h, max(y1, y2) + 5)
            else:
                minx, maxx = min(x1, x2), max(x1, x2)
                miny, maxy = min(y1, y2), max(y1, y2)
            raw_boxes.append(shapely_box(minx, miny, maxx, maxy))

        # Annotate raw detections (include confidence in label)
        raw_boxes_and_types = [
            (raw_boxes[i], f"{raw_labels[i]} {raw_confs[i]:.2f}")
            for i in range(len(raw_boxes))
        ]
        print(f"  Raw detections: {len(raw_boxes_and_types)}")

        # -------------------------------------------------------------------
        # 3. Run grouping to get PROCESSED layout blocks
        # -------------------------------------------------------------------
        grouped = find_grouped_bounding_boxes(raw_boxes, raw_labels)
        print(f"  Grouped blocks: {len(grouped)}")
        for geom, label in grouped:
            x1, y1, x2, y2 = (int(round(v)) for v in geom.bounds)
            print(f"    [{label}]  ({x1},{y1}) -> ({x2},{y2})")

        # -------------------------------------------------------------------
        # 4. Draw both panels
        # -------------------------------------------------------------------
        panel_raw     = draw_boxes(img_bgr, raw_boxes_and_types)
        panel_grouped = draw_boxes(img_bgr, grouped)

        panel_raw     = add_title_bar(panel_raw,     f"RAW YOLO detections  ({len(raw_boxes_and_types)} boxes)  —  page {args.page}")
        panel_grouped = add_title_bar(panel_grouped, f"GROUPED layout blocks  ({len(grouped)} boxes)  —  page {args.page}")

        # Resize both panels to the same height before stacking side by side
        h1, h2 = panel_raw.shape[0], panel_grouped.shape[0]
        target_h = max(h1, h2)
        if h1 < target_h:
            pad = np.full((target_h - h1, panel_raw.shape[1], 3), 200, dtype=np.uint8)
            panel_raw = np.vstack([panel_raw, pad])
        if h2 < target_h:
            pad = np.full((target_h - h2, panel_grouped.shape[1], 3), 200, dtype=np.uint8)
            panel_grouped = np.vstack([panel_grouped, pad])

        # Separator
        sep = np.full((target_h, 6, 3), 80, dtype=np.uint8)
        side_by_side = np.hstack([panel_raw, sep, panel_grouped])

        # -------------------------------------------------------------------
        # 5. Legend at the bottom
        # -------------------------------------------------------------------
        all_labels = raw_labels + [t for _, t in grouped]
        legend = legend_strip(all_labels)
        # Pad legend to full width
        if legend.shape[1] < side_by_side.shape[1]:
            pad = np.full((legend.shape[0], side_by_side.shape[1] - legend.shape[1], 3), 240, dtype=np.uint8)
            legend = np.hstack([legend, pad])
        else:
            legend = legend[:, :side_by_side.shape[1]]

        legend_title = np.full((24, side_by_side.shape[1], 3), 200, dtype=np.uint8)
        cv2.putText(legend_title, "LEGEND", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

        final = np.vstack([side_by_side, legend_title, legend])

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 6. Save
    # -----------------------------------------------------------------------
    cv2.imwrite(out_path, final)
    print(f"\nSaved: {out_path}  ({final.shape[1]}x{final.shape[0]} px)")


if __name__ == "__main__":
    main()
