"""Document layout analysis using YOLO-based detection.

Runs doclayout-yolo and YOLOv26-nano in ensemble: doclayout-yolo does
the primary detection (two-pass at conf 0.51/0.25), then YOLOv26-nano
adds regions that doclayout-yolo missed, deduplicated via IoU.
Returns list of (Polygon, label_str) pairs in reading order."""

import logging
import time
from shapely.geometry import box
from shapely.ops import unary_union
import networkx as nx
from collections import defaultdict
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE OPTIMIZATION: Model Caching
# YOLO model loading is expensive. Cache it as a module-level singleton.
# ============================================================================
_CACHED_YOLO_MODEL = None
_CACHED_YOLO_DEVICE = None
_CACHED_YOLO26_MODEL = None

# YOLOv26 model path on HuggingFace
_YOLO26_REPO = "Armaggheddon/yolo26-document-layout"
_YOLO26_FILENAME = "yolo26n_doc_layout.pt"  # nano by default, small/medium also available

# Mapping from YOLOv26 class names to doclayout-yolo class names
_YOLO26_TO_DOCLAYOUT = {
    "Text": "plain text",
    "Title": "title",
    "Section-header": "title",
    "Picture": "figure",
    "Formula": "isolate_formula",
    "Table": "table",
    "Caption": "figure_caption",
    "List-item": "plain text",
    "Page-header": "abandon",
    "Page-footer": "abandon",
    "Footnote": "abandon",
}

# Import device_utils with conditional logic for script/module execution
try:
    from device_utils import get_device_for_yolo
except ImportError:
    try:
        from .device_utils import get_device_for_yolo
    except ImportError:
        # Fallback if device_utils not available
        def get_device_for_yolo(model):
            return "cpu"
        logger.warning("device_utils not available in layout.py, defaulting to CPU")

# Try to import doclayout_yolo - it's optional
try:
    from doclayout_yolo import YOLOv10
    DOCLAYOUT_AVAILABLE = True
except ImportError:
    YOLOv10 = None
    DOCLAYOUT_AVAILABLE = False
    logger.warning("doclayout_yolo is not installed. Layout analysis will not be available. Install it with: pip install doclayout-yolo")

# Try to import ultralytics (YOLOv26) - optional
try:
    from ultralytics import YOLO as YOLO_ULTRALYTICS
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO_ULTRALYTICS = None
    ULTRALYTICS_AVAILABLE = False

# Get the path to the model file in the project
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "doclayout_yolo_docstructbench_imgsz1024.pt"


def get_yolo_model():
    """
    Get or create the cached YOLO model.

    PERFORMANCE OPTIMIZATION: Models take seconds to load from disk.
    Cache the model as a module-level singleton so it's only loaded once
    per Python session.

    Returns:
        YOLOv10: The cached YOLO model, or None if unavailable
    """
    global _CACHED_YOLO_MODEL, _CACHED_YOLO_DEVICE

    if _CACHED_YOLO_MODEL is not None:
        logger.debug(f"Using cached YOLO model on device: {_CACHED_YOLO_DEVICE}")
        return _CACHED_YOLO_MODEL

    if not DOCLAYOUT_AVAILABLE:
        logger.error("doclayout_yolo is not available")
        return None

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None

    try:
        logger.info("Loading YOLO model (first time - will be cached)...")
        model = YOLOv10(str(MODEL_PATH))
        device = get_device_for_yolo(model)
        logger.debug(f"Device for YOLOv10 determined: {device}")
        model = model.to(device)
        logger.info(f"YOLO model loaded on device: {device}")

        # Cache for future use
        _CACHED_YOLO_MODEL = model
        _CACHED_YOLO_DEVICE = device

        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv10 model from {MODEL_PATH}: {e}")
        return None


def get_yolo26_model():
    """Get or create the cached YOLOv26 model (ultralytics YOLO)."""
    global _CACHED_YOLO26_MODEL
    if _CACHED_YOLO26_MODEL is not None:
        return _CACHED_YOLO26_MODEL
    if not ULTRALYTICS_AVAILABLE:
        logger.error("ultralytics is not available, install with: pip install ultralytics")
        return None
    try:
        from model_manager import get_yolo26_path
        model_path = get_yolo26_path()
        _CACHED_YOLO26_MODEL = YOLO_ULTRALYTICS(model_path)
        logger.info("YOLOv26 model loaded and cached")
        return _CACHED_YOLO26_MODEL
    except Exception as e:
        logger.error(f"Failed to load YOLOv26 model: {e}")
        return None

def find_grouped_bounding_boxes(boxes, types):
    """Group raw YOLO detections into merged layout regions.

    Pairs captions with figures/tables/formulas, merges overlapping
    plain-text regions via a graph algorithm, slices plain-text boxes
    around embedded non-text regions, and groups adjacent title+body pairs.
    Returns list of ``(shapely.Polygon, label_str)``."""
    # Step 1: Index boxes by type
    type_to_indices = defaultdict(list)
    for idx, t in enumerate(types):
        type_to_indices[t].append(idx)

    # Step 1a: Dedup — if the same box coordinates are predicted as both
    # "abandon" and "plain text" with near-identical coordinates (>90% IoU),
    # prefer "plain text". This prevents ambiguous edge detections from
    # splitting legitimate text blocks in Step 6.
    plaintext_indices = set(type_to_indices.get("plain text", []))
    for t in list(type_to_indices.keys()):
        if t == "plain text":
            continue
        for idx in list(type_to_indices[t]):
            box_i = boxes[idx]
            for pt_idx in plaintext_indices:
                box_pt = boxes[pt_idx]
                if not box_i.intersects(box_pt):
                    continue
                inter = box_i.intersection(box_pt)
                if inter.is_empty:
                    continue
                area_i = box_i.area
                if area_i == 0:
                    continue
                # Check if the abandon-like box is >90% contained within the
                # plain text box. Use containment ratio (inter/box_i.area)
                # instead of IoU because _layout_from_det_res expands plain text
                # boxes by 5px but not other types, reducing IoU.
                containment = inter.area / area_i
                if containment >= 0.9:
                    type_to_indices[t].remove(idx)
                    break

    result = []

    # Step 2: Handle "figure" and "figure_caption"
    figures = set(type_to_indices.get("figure", []))
    captions = set(type_to_indices.get("figure_caption", []))
    used_figures = set()
    used_captions = set()

    # Pair each caption to the nearest unpaired figure
    for cap_idx in captions:
        cap_box = boxes[cap_idx]
        min_dist = float("inf")
        nearest_fig = None
        for fig_idx in figures - used_figures:
            fig_box = boxes[fig_idx]
            dist = cap_box.centroid.distance(fig_box.centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_fig = fig_idx
        if nearest_fig is not None:
            union = unary_union([boxes[nearest_fig], boxes[cap_idx]])
            result.append((box(*union.bounds), "figure_and_caption"))
            used_figures.add(nearest_fig)
            used_captions.add(cap_idx)

    # Add unpaired figures
    for fig_idx in figures - used_figures:
        result.append((boxes[fig_idx], "figure"))

    # Add unpaired captions
    for cap_idx in captions - used_captions:
        result.append((boxes[cap_idx], "figure_caption"))

    # Step 3: Handle "isolate_formula" and "formula_caption"
    formulas = set(type_to_indices.get("isolate_formula", []))
    formula_captions = set(type_to_indices.get("formula_caption", []))
    used_formulas = set()
    used_formula_captions = set()

    # Pair each formula caption to the nearest unpaired formula by y-axis proximity.
    # Captions are always placed to the right of their formula at the same vertical
    # band (equation number), so we match on vertical overlap / y-centre distance
    # rather than Euclidean centroid distance, which can mis-pair across rows.
    for cap_idx in formula_captions:
        cap_box = boxes[cap_idx]
        cap_y_mid = (cap_box.bounds[1] + cap_box.bounds[3]) / 2
        min_dist = float("inf")
        nearest_formula = None
        for formula_idx in formulas - used_formulas:
            formula_box = boxes[formula_idx]
            formula_y_mid = (formula_box.bounds[1] + formula_box.bounds[3]) / 2
            # Use vertical distance between y-midpoints as the primary metric
            y_dist = abs(cap_y_mid - formula_y_mid)
            if y_dist < min_dist:
                min_dist = y_dist
                nearest_formula = formula_idx
        if nearest_formula is not None:
            union = unary_union([boxes[nearest_formula], boxes[cap_idx]])
            result.append((box(*union.bounds), "isolate_formula_and_caption"))
            used_formulas.add(nearest_formula)
            used_formula_captions.add(cap_idx)

    # Add unpaired formulas
    for formula_idx in formulas - used_formulas:
        result.append((boxes[formula_idx], "isolate_formula"))

    # Add unpaired formula captions
    for cap_idx in formula_captions - used_formula_captions:
        result.append((boxes[cap_idx], "formula_caption"))

    # Step 4: Handle "table", "table_caption", and "table_footnote"
    tables = set(type_to_indices.get("table", []))
    table_captions = set(type_to_indices.get("table_caption", []))
    table_footnotes = set(type_to_indices.get("table_footnote", []))
    used_tables = set()
    used_table_captions = set()
    used_table_footnotes = set()

    # First, pair tables with their captions
    for cap_idx in table_captions:
        cap_box = boxes[cap_idx]
        min_dist = float("inf")
        nearest_table = None
        for table_idx in tables - used_tables:
            table_box = boxes[table_idx]
            dist = cap_box.centroid.distance(table_box.centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_table = table_idx
        if nearest_table is not None:
            used_tables.add(nearest_table)
            used_table_captions.add(cap_idx)

    # Then, pair tables (possibly already paired with caption) with footnotes
    table_groups = {}  # table_idx -> list of associated box indices
    for table_idx in tables:
        table_groups[table_idx] = [table_idx]
        # Add caption if paired
        for cap_idx in table_captions:
            if cap_idx in used_table_captions:
                cap_box = boxes[cap_idx]
                table_box = boxes[table_idx]
                # Check if this caption belongs to this table
                if table_idx in used_tables:
                    # Find which caption is paired with this table
                    min_dist = float("inf")
                    nearest_cap = None
                    for c_idx in table_captions:
                        if c_idx in used_table_captions:
                            dist = boxes[c_idx].centroid.distance(table_box.centroid)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_cap = c_idx
                    if nearest_cap == cap_idx:
                        table_groups[table_idx].append(cap_idx)

    # Pair footnotes with nearest table group
    for footnote_idx in table_footnotes:
        footnote_box = boxes[footnote_idx]
        min_dist = float("inf")
        nearest_table = None
        for table_idx in tables:
            table_box = boxes[table_idx]
            dist = footnote_box.centroid.distance(table_box.centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_table = table_idx
        if nearest_table is not None:
            table_groups[nearest_table].append(footnote_idx)
            used_table_footnotes.add(footnote_idx)

    # Create combined boxes for table groups
    for table_idx, group_indices in table_groups.items():
        if len(group_indices) > 1:
            # Combine all boxes in the group
            group_boxes = [boxes[idx] for idx in group_indices]
            union = unary_union(group_boxes)
            result.append((box(*union.bounds), "table_and_caption"))
        else:
            # Just the table
            result.append((boxes[table_idx], "table"))

    # Add unpaired table captions
    for cap_idx in table_captions - used_table_captions:
        result.append((boxes[cap_idx], "table_caption"))

    # Add unpaired table footnotes
    for footnote_idx in table_footnotes - used_table_footnotes:
        result.append((boxes[footnote_idx], "table_footnote"))

    # Step 5: Group "plain text" boxes by intersection
    plaintext_indices = type_to_indices.get("plain text", [])
    plain_text_boxes = []
    if plaintext_indices:
        G = nx.Graph()
        for i in plaintext_indices:
            G.add_node(i)
        for i_idx, i in enumerate(plaintext_indices):
            for j in plaintext_indices[i_idx + 1:]:
                if not boxes[i].intersects(boxes[j]):
                    continue
                inter = boxes[i].intersection(boxes[j])
                if inter.is_empty:
                    continue
                # Only merge if the overlap is substantial relative to the
                # smaller box — prevents a full-width box from absorbing a
                # narrow right-column box that merely touches its boundary.
                smaller_area = min(boxes[i].area, boxes[j].area)
                if smaller_area > 0 and inter.area / smaller_area >= 0.1:
                    G.add_edge(i, j)
        for component in nx.connected_components(G):
            subset = [boxes[i] for i in component]
            union = unary_union(subset)
            plain_text_boxes.append(union)  # keep actual union shape, not bounding hull

    # Step 6: Split plain text boxes around overlapping formula/figure/table regions.
    #
    # DocLayout-YOLO often draws a plain text region that spans the full column
    # height, including rows that contain a displayed formula.  The formula box
    # sits *inside* the text box.  We slice each plain text box horizontally at
    # every overlapping non-text region so that the resulting pieces contain only
    # actual prose — the formula, figure or table regions remain untouched as
    # their own boxes.
    #
    # Only the y-axis is used for splitting because displayed math and figures
    # occupy horizontal bands that run across the full column width.

    # Collect all non-text regions to use as cut boundaries.
    # This includes:
    #   - regions already finalised in `result` (figures, formulas, tables …)
    #   - ALL raw boxes whose type is not "plain text" (titles, abandon, etc.)
    #     These are added in Step 7 and therefore not yet in `result`, but they
    #     must still be cut out of merged plain text blocks so that a giant
    #     union box does not swallow correctly-detected titles or other elements.
    already_in_result = [geom for geom, _ in result]
    raw_non_text = [boxes[i] for i, t in enumerate(types) if t != "plain text"]
    non_text_regions = already_in_result + raw_non_text

    MIN_PIECE_HEIGHT = 20  # pixels — discard slivers smaller than this

    for pt_box in plain_text_boxes:
        px1, py1, px2, py2 = pt_box.bounds

        # Collect y-intervals of every non-text region that overlaps this text box.
        cut_bands = []
        for nt_geom in non_text_regions:
            if not pt_box.intersects(nt_geom):
                continue
            inter = pt_box.intersection(nt_geom)
            if inter.is_empty:
                continue
            ib = inter.bounds          # (x1, y1, x2, y2)
            overlap_height = ib[3] - ib[1]
            # Only cut if the overlap is tall enough to matter (>= 10 px).
            if overlap_height < 10:
                continue
            cut_bands.append((ib[1], ib[3]))  # (band_top, band_bottom)

        if not cut_bands:
            # No overlapping non-text regions — keep the box as-is.
            result.append((pt_box, "plain text"))
            continue

        # Merge overlapping/adjacent cut bands so we don't create duplicates.
        cut_bands.sort()
        merged_bands = [cut_bands[0]]
        for band_top, band_bot in cut_bands[1:]:
            if band_top <= merged_bands[-1][1]:
                merged_bands[-1] = (merged_bands[-1][0], max(merged_bands[-1][1], band_bot))
            else:
                merged_bands.append((band_top, band_bot))

        # Build the y-slices that represent text-only pieces.
        # Pieces are the gaps between (and outside) the cut bands.
        slice_tops = [py1] + [b[1] for b in merged_bands]
        slice_bots = [b[0] for b in merged_bands] + [py2]

        for top, bot in zip(slice_tops, slice_bots):
            if bot - top >= MIN_PIECE_HEIGHT:
                result.append((box(px1, top, px2, bot), "plain text"))

    # Step 7: Add all other types as individual boxes
    for t, indices in type_to_indices.items():
        if t in {"figure", "figure_caption", "isolate_formula", "formula_caption",
                 "table", "table_caption", "table_footnote", "plain text"}:
            continue
        for idx in indices:
            result.append((boxes[idx], t))

    # Step 8: Group adjacent title + plain text pairs into titled_block_title /
    # titled_block_body.  A title and a plain text box are grouped when:
    #   - the plain text top is within 50px below the title bottom
    #   - they share > 50% x-overlap relative to the narrower box
    # The two entries replace the originals in result, keeping title first so
    # main.py always sees titled_block_title immediately before titled_block_body.
    title_entries = [(i, geom, t) for i, (geom, t) in enumerate(result) if t == "title"]
    pt_entries    = [(i, geom, t) for i, (geom, t) in enumerate(result) if t == "plain text"]

    paired_result_indices = set()   # indices in result that have been consumed
    new_pairs = []                  # (title_idx, pt_idx, geom_t, geom_p)

    for ti, geom_t, _ in title_entries:
        tx1, ty1, tx2, ty2 = geom_t.bounds
        best = None
        best_gap = float('inf')
        for pi, geom_p, _ in pt_entries:
            if pi in paired_result_indices:
                continue
            px1, py1, px2, py2 = geom_p.bounds
            gap = py1 - ty2
            if gap < 0 or gap > 50:
                continue
            # x-overlap check
            overlap = min(tx2, px2) - max(tx1, px1)
            narrower = min(tx2 - tx1, px2 - px1)
            if narrower <= 0 or overlap / narrower < 0.5:
                continue
            if gap < best_gap:
                best_gap = gap
                best = (pi, geom_p)
        if best is not None:
            pi, geom_p = best
            paired_result_indices.add(ti)
            paired_result_indices.add(pi)
            new_pairs.append((ti, pi, geom_t, geom_p))

    if new_pairs:
        # Rebuild result: keep unpaired entries, insert paired ones in title's position
        new_result = []
        pair_by_title_idx = {ti: (geom_t, geom_p) for ti, pi, geom_t, geom_p in new_pairs}
        for i, (geom, t) in enumerate(result):
            if i in paired_result_indices:
                if i in pair_by_title_idx:
                    geom_t, geom_p = pair_by_title_idx[i]
                    new_result.append((geom_t, "titled_block_title"))
                    new_result.append((geom_p, "titled_block_body"))
                # plain text partner is skipped (already added above)
            else:
                new_result.append((geom, t))
        result = new_result

    # --- DEDUPLICATE plain text boxes, preserving overall order ---
    # Build a set of plain text indices to drop (those fully covered by another).
    plain_boxes = [(i, geom) for i, (geom, t) in enumerate(result) if t == "plain text"]
    drop_indices = set()
    for ii, (i, geom_i) in enumerate(plain_boxes):
        yi1, yi2 = geom_i.bounds[1], geom_i.bounds[3]
        xi1, xi2 = geom_i.bounds[0], geom_i.bounds[2]
        for jj, (j, geom_j) in enumerate(plain_boxes):
            if ii == jj:
                continue
            yj1, yj2 = geom_j.bounds[1], geom_j.bounds[3]
            if yi1 >= yj1 - 2 and yi2 <= yj2 + 2:
                xj1, xj2 = geom_j.bounds[0], geom_j.bounds[2]
                if (min(xi2, xj2) - max(xi1, xj1)) > 0.7 * (xi2 - xi1):
                    drop_indices.add(i)
                    break
    final_result = [(geom, t) for i, (geom, t) in enumerate(result) if i not in drop_indices]
    return final_result


def _layout_from_det_res(det_res, img_height: int, img_width: int):
    """Shared post-processing for YOLO detection results."""
    names = det_res[0].names
    blocknames = [names[int(n)] for n in det_res[0].boxes.cls]
    xyxy = [a.tolist() for a in det_res[0].boxes.xyxy]
    rect_list = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        if blocknames[i] == "plain text":
            minx = max(0, min(x1, x2) - 5)
            maxx = min(img_width, max(x1, x2) + 5)
            miny = max(0, min(y1, y2) - 5)
            maxy = min(img_height, max(y1, y2) + 5)
        else:
            minx = min(x1, x2)
            maxx = max(x1, x2)
            miny = min(y1, y2)
            maxy = max(y1, y2)
        rect_list.append(box(minx, miny, maxx, maxy))
    return find_grouped_bounding_boxes(rect_list, blocknames)


def layout(image_path):
    """
    Perform layout analysis on an image.

    PERFORMANCE OPTIMIZATION: Uses cached model to avoid reloading.

    Args:
        image_path: Path to the image file

    Returns:
        List of (geometry, type) tuples for detected layout elements
    """
    model = get_yolo_model()

    if model is None:
        raise RuntimeError(
            "doclayout_yolo is not available. Install it with: pip install doclayout-yolo. "
            "If installed, make sure the model file exists at: " + str(MODEL_PATH)
        )

    device = _CACHED_YOLO_DEVICE if _CACHED_YOLO_DEVICE else get_device_for_yolo(model)

    det_res = model.predict(
        image_path,
        imgsz=1024,
        conf=0.51,
        device=device,
    )

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img_height, img_width = img.shape[:2]

    return _layout_from_det_res(det_res, img_height, img_width)


def _layout_from_array_doclayout(img_bgr: "np.ndarray"):
    """
    Perform layout analysis on an in-memory BGR numpy array.

    Avoids writing a temp file — use this in the server hot path instead of
    layout() to save ~0.3–0.5 s per page.

    Args:
        img_bgr: BGR uint8 numpy array (H, W, 3)

    Returns:
        List of (geometry, type) tuples for detected layout elements
    """
    model = get_yolo_model()

    if model is None:
        raise RuntimeError(
            "doclayout_yolo is not available. Install it with: pip install doclayout-yolo. "
            "If installed, make sure the model file exists at: " + str(MODEL_PATH)
        )

    device = _CACHED_YOLO_DEVICE if _CACHED_YOLO_DEVICE else get_device_for_yolo(model)

    t0 = time.perf_counter()

    # ultralytics YOLO predict() accepts BGR numpy arrays directly
    det_res = model.predict(
        img_bgr,
        imgsz=1024,
        conf=0.51,
        device=device,
    )

    img_height, img_width = img_bgr.shape[:2]
    result = _layout_from_det_res(det_res, img_height, img_width)
    t1 = time.perf_counter()
    print(f"[timing] layout pass1 (conf=0.51): {t1-t0:.3f}s  blocks={len(result)}", file=__import__('sys').stderr)

    # Second pass at lower confidence to recover blocks missed at the standard
    # threshold (e.g. large decorative banner titles, mixed-content regions).
    det_res_low = model.predict(
        img_bgr,
        imgsz=1024,
        conf=0.10,
        device=device,
    )
    result_low = _layout_from_det_res(det_res_low, img_height, img_width)
    t2 = time.perf_counter()

    # Keep blocks from the low-conf pass that are NOT already substantially
    # covered by any block in the standard-conf result.
    existing_boxes = [geom for geom, _ in result]
    n_before = len(result)
    for geom_low, label_low in result_low:
        area_low = geom_low.area
        if area_low == 0:
            continue
        covered = False
        for existing in existing_boxes:
            if not geom_low.intersects(existing):
                continue
            inter = geom_low.intersection(existing)
            if inter.area / area_low >= 0.5:
                covered = True
                break
        if not covered:
            result.append((geom_low, label_low))

    t3 = time.perf_counter()
    print(f"[timing] layout pass2 (conf=0.10): {t2-t1:.3f}s  added={len(result)-n_before}", file=__import__('sys').stderr)
    print(f"[timing] _layout_from_array_doclayout TOTAL:  {t3-t0:.3f}s  total_blocks={len(result)}", file=__import__('sys').stderr)

    return result


def _layout_from_array_yolo26(img_bgr: "np.ndarray", conf: float = 0.10):
    """Run YOLOv26 on an image and return (polygon, label) pairs."""
    model = get_yolo26_model()
    if model is None:
        return []

    res = model.predict(img_bgr, imgsz=1280, conf=conf, verbose=False)[0]
    img_h, img_w = img_bgr.shape[:2]
    results = []
    from shapely.geometry import box as _box26
    for box_coords, cls_id in zip(res.boxes.xyxy, res.boxes.cls):
        x1, y1, x2, y2 = box_coords.tolist()
        class_name = res.names[int(cls_id)]
        mapped = _YOLO26_TO_DOCLAYOUT.get(class_name, "plain text")
        geom = _box26(
            max(0, x1), max(0, y1),
            min(img_w, x2), min(img_h, y2),
        )
        results.append((geom, mapped))
    return results


def layout_from_array(img_bgr: "np.ndarray") -> list:
    """Ensemble layout: doclayout-yolo primary + YOLOv26 for missed regions.

    Runs doclayout-yolo first for its reliable compound labels and grouping,
    then YOLOv26 for additional detections that doclayout-yolo missed.
    Merged with IoU-based deduplication.
    """
    t0 = time.perf_counter()

    # Primary pass: doclayout-yolo
    primary = _layout_from_array_doclayout(img_bgr)
    t1 = time.perf_counter()

    # Secondary pass: YOLOv26
    secondary = _layout_from_array_yolo26(img_bgr)
    t2 = time.perf_counter()

    # Merge: keep secondary detections that don't overlap existing ones
    existing_boxes = [geom for geom, _ in primary]
    existing_labels = [l for _, l in primary]
    added = 0
    for geom_s, label_s in secondary:
        area_s = geom_s.area
        if area_s == 0:
            continue
        # Skip if substantially covered by any existing detection
        covered = False
        keep_asymmetric = False
        for existing, label_e in zip(existing_boxes, existing_labels):
            if not geom_s.intersects(existing):
                continue
            inter = geom_s.intersection(existing)
            if inter.area / area_s >= 0.3:
                # Asymmetric case: small text detection inside much larger title region
                # This happens when a banner/headline box swallows adjacent plain text
                if "plain" in label_s and "title" in label_e:
                    area_e = existing.area
                    if area_e > area_s * 3 and inter.area / area_e < 0.2:
                        keep_asymmetric = True
                        continue  # check if other overlaps also qualify
                covered = True
                break
        if keep_asymmetric:
            covered = False  # override: mixed content inside oversized region
        if not covered:
            primary.append((geom_s, label_s))
            existing_boxes.append(geom_s)
            existing_labels.append(label_s)
            added += 1

    t3 = time.perf_counter()

    # Post-processing: reclassify banner-like figures at the top of the page as titles.
    # Stylized/artistic newspaper mastheads and decorative headers are often classified
    # as `figure` by the layout model because the non-standard font doesn't match its
    # "text" training data. Recover them by checking aspect ratio and position.
    img_h, img_w = img_bgr.shape[:2]
    reclassified = 0
    for i, (geom, label) in enumerate(primary):
        if label != "figure":
            continue
        x1, y1, x2, y2 = geom.bounds
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue
        aspect = box_w / box_h
        in_top_half = y2 < img_h * 0.5
        is_wide = aspect > 2.5 and box_w > 500
        is_banner = aspect > 1.5 and box_w > 700 and y2 < img_h * 0.33
        if (is_wide or is_banner) and in_top_half:
            primary[i] = (geom, "title")
            reclassified += 1

    if reclassified:
        print(f"[ensemble] reclassified {reclassified} banner-like figure(s) as title",
              file=__import__('sys').stderr)

    # Dedup overlapping titles after reclassification (same banner detected in 2 passes)
    title_idx = [i for i, (_, l) in enumerate(primary) if l == "title"]
    remove = set()
    for i in range(len(title_idx)):
        for j in range(i + 1, len(title_idx)):
            ai = title_idx[i]
            aj = title_idx[j]
            if ai in remove or aj in remove:
                continue
            gi, gj = primary[ai][0], primary[aj][0]
            xi1, yi1, xi2, yi2 = gi.bounds
            xj1, yj1, xj2, yj2 = gj.bounds
            ix1, iy1 = max(xi1, xj1), max(yi1, yj1)
            ix2, iy2 = min(xi2, xj2), min(yi2, yj2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (xi2 - xi1) * (yi2 - yi1) + (xj2 - xj1) * (yj2 - yj1) - inter
            iou = inter / union if union > 0 else 0
            if iou > 0.40:
                remove.add(min(ai, aj, key=lambda k: (
                    (primary[k][0].bounds[2] - primary[k][0].bounds[0])
                    * (primary[k][0].bounds[3] - primary[k][0].bounds[1]))))
    if remove:
        primary = [p for i, p in enumerate(primary) if i not in remove]
        print(f"[ensemble] deduped {len(remove)} overlapping title(s)",
              file=__import__('sys').stderr)

    t4 = time.perf_counter()
    print(f"[timing] ensemble: doclayout {t1-t0:.3f}s + yolo26 {t2-t1:.3f}s + merge {t3-t2:.3f}s + post {t4-t3:.3f}s = {t4-t0:.3f}s total, {added} extra from yolo26",
          file=__import__('sys').stderr)

    return primary

