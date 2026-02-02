import sys
import os
import logging

# Configure logging FIRST before any imports that might use it
# This ensures logging from imported modules (like layout.py) is visible
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,  # Shows device detection, model loading, etc.
        format='%(levelname)s: %(message)s'
    )

# Add current directory to path for imports when running as script
# This needs to happen before any local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from math import ceil

# Set up logger for this module
logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE OPTIMIZATION: Model Caching
# Models are expensive to load (~10 seconds). Cache them as module-level
# singletons so they're only loaded once per Python session.
# ============================================================================
_CACHED_DOCTR_MODEL = None
_CACHED_DOCTR_DEVICE = None

# ============================================================================
# PERFORMANCE OPTIMIZATION: Lazy Imports
# Import only what's needed at module load time. Heavy imports are deferred.
# ============================================================================
import cv2
from operator import itemgetter
from dataclasses import dataclass

# Lazy imports - loaded on first use
# from doctr.models import detection_predictor  # ~2 seconds import time
# from doctr.io import DocumentFile  # ~1 second import time
# from shapely import LineString, box  # Already imported, keep it
from shapely import LineString, box

# Use conditional imports to support both script and module usage
try:
    # Try script-style imports first (when run with python src/ocr_reflow/main.py)
    from device_utils import get_device_for_doctr
    from reflow import create_page_with_word_wrapping
    from divide_conquer_4d import divide_conquer_4d, Point4D
    from layout import layout as analyze_layout
except ImportError as e1:
    # Fall back to package-style imports (when imported as module)
    try:
        from .device_utils import get_device_for_doctr
    except ImportError:
        def get_device_for_doctr():
            return "cpu"
        logger.warning("device_utils not available, defaulting to CPU")

    try:
        from .reflow import create_page_with_word_wrapping
    except ImportError:
        logger.error("Could not import reflow module. This is required.")
        raise

    try:
        from .divide_conquer_4d import divide_conquer_4d, Point4D
    except ImportError:
        logger.error("Could not import divide_conquer_4d module. This is required.")
        raise

    # Layout is optional
    try:
        from .layout import layout as analyze_layout
    except ImportError as e:
        logger.warning(f"Could not import layout module: {e}. Layout analysis will not be available.")
        analyze_layout = None

@dataclass
class Letter:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    bl: int


def get_doctr_model():
    """
    Get or create the cached DocTR model.

    PERFORMANCE OPTIMIZATION: Models take ~10 seconds to load from disk.
    Cache the model as a module-level singleton so it's only loaded once
    per Python session. This provides massive speedup for batch processing.

    Returns:
        tuple: (model, device) - The detection model and the device it's on
    """
    global _CACHED_DOCTR_MODEL, _CACHED_DOCTR_DEVICE

    if _CACHED_DOCTR_MODEL is not None:
        logger.debug(f"Using cached DocTR model on device: {_CACHED_DOCTR_DEVICE}")
        return _CACHED_DOCTR_MODEL, _CACHED_DOCTR_DEVICE

    logger.info("Loading DocTR model (first time - will be cached)...")

    # Lazy import - only import when actually needed
    from doctr.models import detection_predictor

    # Get optimal device
    device = get_device_for_doctr()

    # Load model
    model = detection_predictor(pretrained=True)

    # Move to optimal device (GPU if available)
    if hasattr(model, 'to'):
        try:
            model = model.to(device)
            logger.info(f"DocTR model loaded on device: {device}")
        except Exception as e:
            logger.warning(f"Could not move DocTR model to {device}: {e}. Using default device.")
            device = "cpu"

    # Cache for future use
    _CACHED_DOCTR_MODEL = model
    _CACHED_DOCTR_DEVICE = device

    return model, device


def find_rects(img, line_words):
    rects = []
    for xmin,ymin,xmax,ymax in line_words:
        word_height = ymax - ymin
        word_width = xmax - xmin
        r = img[ymin:ymax,xmin:xmax,:].copy()
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r, 8, cv2.CV_32S)

        # Collect valid components and filter noise
        valid_components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter out tiny noise components
            # Must be at least 3x3 pixels and have reasonable area
            if w >= 3 and h >= 3 and area >= 9:
                # Filter out components that are too small relative to word height
                # (likely noise from image artifacts)
                if h >= word_height * 0.2:  # At least 20% of word height
                    valid_components.append((x, y, w, h))

        # If filtering removed everything, be more lenient
        if len(valid_components) == 0 and num_labels > 1:
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                if w >= 2 and h >= 2:
                    valid_components.append((x, y, w, h))

        # Add valid components to rects with padding for angled text
        # Add 1-2 pixels padding to ensure we capture all letter pixels
        padding = 2
        for x, y, w, h in valid_components:
            # Apply padding but stay within word bounds
            padded_x = max(0, x - padding)
            padded_y = max(0, y - padding)
            padded_w = min(w + 2 * padding, word_width - padded_x)
            padded_h = min(h + 2 * padding, word_height - padded_y)

            rects.append((padded_x+xmin, padded_y+ymin, padded_x+padded_w+xmin, padded_y+padded_h+ymin))

    rectangles = [(int(xmin), int(xmax), int(ymin), int(ymax)) for xmin, ymin, xmax, ymax in rects]

    points4 = [
        Point4D(l, b, -r, -t, index=i)
        for i, (l, r, b, t) in enumerate(rectangles)
    ]
    pairs = divide_conquer_4d(points4)
    ind_to_remove = [i for i, j in sorted(pairs)]
    # for i, j in sorted(pairs):
    #     print(f"  Rectangle R{i} encloses Rectangle R{j}")
    #     print(f"Enclosing {rectangles[j]}")
    #     print(f"Enclosed {rectangles[i]}")
    # print(rects)

    rects = [v for i, v in enumerate(rects) if i not in ind_to_remove]
    return rects

def margins(words):
    """
    Detect left and right margins of text lines using clustering algorithm.

    Based on the paper "Text Line Processing for High-Confidence Skew Detection"
    (Rosner et al., Section 3.3 - Clustering)

    Algorithm:
    1. For each character, construct a rectangular neighborhood:
       - Height: equal to the character's height
       - Width: twice the character's height
       - Position: starts from the character's right bottom pixel
    2. Another character is in this neighborhood if its middle-y-line intersects it
    3. Characters in the same neighborhood belong to the same line cluster
    4. Use Union-Find to merge overlapping clusters into text lines
    """
    if len(words) < 2:
        return [], []

    # Calculate median word height for filtering subscripts/superscripts
    word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
    median_height = np.median(word_heights)
    height_threshold = median_height * 0.60

    # Filter and prepare entities (characters/words)
    entities = []
    for xmin, ymin, xmax, ymax, conf in words:
        height = ymax - ymin
        if height >= height_threshold:
            entities.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'height': height,
                'bottom_middle': (xmax, (ymin + ymax) / 2)  # Right bottom pixel + middle Y
            })

    if len(entities) < 2:
        return [], []

    # Union-Find data structure
    parent = list(range(len(entities)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Clustering: For each entity, check which other entities fall in its neighborhood
    for i, entity in enumerate(entities):
        # Construct rectangular neighborhood for this entity
        # Starting from right bottom pixel, extending right by 2*height
        neighborhood_xmin = entity['xmax']
        neighborhood_xmax = entity['xmax'] + 2 * entity['height']
        neighborhood_ymin = entity['ymin']
        neighborhood_ymax = entity['ymax']

        # Check all other entities to see if their middle-y-line intersects this neighborhood
        for j, other in enumerate(entities):
            if i == j:
                continue

            # Calculate middle-y of the other entity
            other_middle_y = (other['ymin'] + other['ymax']) / 2

            # Check if other entity's X range overlaps with neighborhood X range
            x_overlaps = not (other['xmax'] < neighborhood_xmin or other['xmin'] > neighborhood_xmax)

            # Check if other entity's middle-y falls within neighborhood Y range
            y_in_range = neighborhood_ymin <= other_middle_y <= neighborhood_ymax

            if x_overlaps and y_in_range:
                # Other entity is in this entity's neighborhood -> same line
                union(i, j)

    # Group entities by their cluster (line)
    clusters = {}
    for i in range(len(entities)):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # For each cluster (line), find leftmost and rightmost entities
    left_margins = []
    right_margins = []

    # Sort clusters by their topmost entity's Y position
    sorted_clusters = sorted(clusters.items(),
                            key=lambda item: min(entities[idx]['ymin'] for idx in item[1]))

    for cluster_root, entity_indices in sorted_clusters:
        # Find leftmost entity (minimum xmin)
        leftmost_idx = min(entity_indices, key=lambda i: entities[i]['xmin'])
        left_entity = entities[leftmost_idx]
        left_y = (left_entity['ymin'] + left_entity['ymax']) / 2
        left_margins.append((int(left_entity['xmin']), int(left_y)))

        # Find rightmost entity (maximum xmax)
        rightmost_idx = max(entity_indices, key=lambda i: entities[i]['xmax'])
        right_entity = entities[rightmost_idx]
        right_y = (right_entity['ymin'] + right_entity['ymax']) / 2
        right_margins.append((int(right_entity['xmax']), int(right_y)))

    return left_margins, right_margins



def visualize_detected_lines(image, words, left_margins, right_margins, output_path=None):
    """
    Visualize detected text lines with leftmost and rightmost points.

    Args:
        image: Input image (BGR format)
        words: Array of word bounding boxes [(xmin, ymin, xmax, ymax, conf), ...]
        left_margins: List of (x, y) tuples for leftmost points
        right_margins: List of (x, y) tuples for rightmost points
        output_path: Optional path to save visualization

    Returns:
        Visualization image (BGR format)
    """
    vis_img = image.copy()

    # Draw all detected words in light gray
    for xmin, ymin, xmax, ymax, _ in words:
        cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (200, 200, 200), 1)

    # Colors for different lines
    line_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]

    # Draw detected lines
    for i, (l, r) in enumerate(zip(left_margins, right_margins)):
        color = line_colors[i % len(line_colors)]

        # Draw line connecting left and right margins
        cv2.line(vis_img, l, r, color, 2)

        # Draw circles at leftmost point (blue)
        cv2.circle(vis_img, l, 8, (255, 0, 0), -1)  # Blue filled circle
        cv2.circle(vis_img, l, 8, (255, 255, 255), 2)  # White border

        # Draw circles at rightmost point (yellow)
        cv2.circle(vis_img, r, 8, (0, 255, 255), -1)  # Yellow filled circle
        cv2.circle(vis_img, r, 8, (255, 255, 255), 2)  # White border

        # Add line number label
        mid_x = (l[0] + r[0]) // 2
        mid_y = (l[1] + r[1]) // 2
        cv2.putText(vis_img, f"L{i+1}", (mid_x - 20, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, vis_img)

    return vis_img


def merge_close_lines(left_margins, right_margins, words, y_threshold=50):
    """
    Merge lines that are very close together in Y-position.
    This helps handle superscripts, subscripts, and other small elements
    that shouldn't be treated as separate lines.

    Args:
        left_margins: List of (x, y) tuples for left margin points
        right_margins: List of (x, y) tuples for right margin points
        words: Array of word bounding boxes
        y_threshold: Maximum Y-distance to consider lines as mergeable (default 50 pixels)

    Returns:
        Tuple of (merged_left_margins, merged_right_margins)
    """
    if len(left_margins) == 0 or len(right_margins) == 0:
        return left_margins, right_margins

    # Calculate average line spacing to use as reference
    if len(left_margins) > 1:
        y_positions = [ly for _, ly in left_margins]
        gaps = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 50
        # Use the smaller of provided threshold or 0.8x average gap
        # Increased from 0.3 to 0.8 to handle documents where subscripts/superscripts
        # are further apart but still should be merged with main line
        adaptive_threshold = min(y_threshold, avg_gap * 0.8)
    else:
        adaptive_threshold = y_threshold

    # Count how many words are on each line
    line_word_counts = []
    line_word_heights = []  # Track average word height per line
    for i, (lx, ly) in enumerate(left_margins):
        word_count = 0
        heights = []
        for xmin, ymin, xmax, ymax, _ in words:
            word_center_y = (ymin + ymax) / 2
            if abs(word_center_y - ly) < y_threshold / 2:
                word_count += 1
                heights.append(ymax - ymin)
        line_word_counts.append(word_count)
        line_word_heights.append(np.median(heights) if heights else 0)

    # Merge lines - multiple passes
    merged_left = list(left_margins)
    merged_right = list(right_margins)
    word_counts = list(line_word_counts)
    word_heights = list(line_word_heights)

    changed = True
    max_iterations = 10  # Increased iterations
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        new_left = []
        new_right = []
        new_counts = []
        new_heights = []

        i = 0
        while i < len(merged_left):
            current_left = merged_left[i]
            current_right = merged_right[i] if i < len(merged_right) else current_left
            current_word_count = word_counts[i]
            current_height = word_heights[i]

            should_merge = False
            if i + 1 < len(merged_left):
                next_left = merged_left[i + 1]
                next_right = merged_right[i + 1] if i + 1 < len(merged_right) else next_left
                next_word_count = word_counts[i + 1]
                next_height = word_heights[i + 1]

                y_distance = abs(next_left[1] - current_left[1])

                # Multiple merge criteria:
                # 1. Very close lines with few words (superscripts/subscripts)
                if y_distance < 30 and (current_word_count <= 3 or next_word_count <= 3):
                    should_merge = True
                    changed = True
                # 2. Very very close lines (within 30 pixels) regardless of word count
                # Increased to 30 to catch the 26px gap in out5.png
                elif y_distance < 30:
                    should_merge = True
                    changed = True
                # 3. Lines where one is much smaller in height (likely super/subscript)
                # Use more generous distance for very small text (subscripts)
                elif current_height > 0 and next_height > 0:
                    height_ratio = min(current_height, next_height) / max(current_height, next_height)
                    # For very small height ratio (< 0.65), allow larger distance (< 40px)
                    # For moderate height ratio (< 0.7), use adaptive threshold
                    if height_ratio < 0.65 and y_distance < 40 and (current_word_count <= 3 or next_word_count <= 3):
                        should_merge = True
                        changed = True
                    elif height_ratio < 0.7 and y_distance < adaptive_threshold:
                        should_merge = True
                        changed = True

            if should_merge:
                next_left = merged_left[i + 1]
                next_right = merged_right[i + 1] if i + 1 < len(merged_right) else next_left
                next_word_count = word_counts[i + 1]
                next_height = word_heights[i + 1]

                # Use the position with more words and larger text
                if current_word_count >= next_word_count and current_height >= next_height:
                    merged_left_point = current_left
                    merged_right_point = (max(current_right[0], next_right[0]), current_right[1])
                    merged_height = current_height
                else:
                    merged_left_point = next_left
                    merged_right_point = (max(current_right[0], next_right[0]), next_right[1])
                    merged_height = next_height

                new_left.append(merged_left_point)
                new_right.append(merged_right_point)
                new_counts.append(current_word_count + next_word_count)
                new_heights.append(max(current_height, merged_height))
                i += 2
            else:
                new_left.append(current_left)
                new_right.append(current_right)
                new_counts.append(current_word_count)
                new_heights.append(current_height)
                i += 1

        merged_left = new_left
        merged_right = new_right
        word_counts = new_counts
        word_heights = new_heights

    return merged_left, merged_right


def process_document(filename, zoom_factor=2.5, new_page_width=2000):
    # PERFORMANCE OPTIMIZATION: Use cached model instead of loading fresh
    model, device = get_doctr_model()

    # Lazy import DocumentFile only when needed
    from doctr.io import DocumentFile

    # PERFORMANCE: Read image once instead of multiple times
    img = cv2.imread(filename)
    img_h, img_w, _ = img.shape

    # Lazy import DocumentFile only when needed
    from doctr.io import DocumentFile

    # Load and process image
    docs = DocumentFile.from_images([filename])
    result = model(docs)
    words = result[0]["words"]
    # Add more padding to word boxes to prevent letter clipping, especially for angled text
    # Increased from 2 to 5 pixels to ensure full letter capture
    words[:, 0] = (words[:, 0] * img_w).astype(np.int32) - 5  # left: expand left
    words[:, 1] = (words[:, 1] * img_h).astype(np.int32) - 5  # top: expand up
    words[:, 2] = (words[:, 2] * img_w).astype(np.int32) + 5  # right: expand right
    words[:, 3] = (words[:, 3] * img_h).astype(np.int32) + 5  # bottom: expand down
    # Clamp to image bounds
    words[:, 0] = np.maximum(words[:, 0], 0)
    words[:, 1] = np.maximum(words[:, 1], 0)
    words[:, 2] = np.minimum(words[:, 2], img_w)
    words[:, 3] = np.minimum(words[:, 3], img_h)
    words = words.astype(np.int32)

    # PERFORMANCE: Read image once, removed redundant reads
    # Previously read the same image 3 times (img, img1, img2) for debug visualization
    # Now we only read once and reuse the same array
    left_margins, right_margins = margins(words)

    # Merge lines that are too close together (fixes superscript/subscript issues)
    # Increased threshold to 30 to handle documents with slightly larger spacing between subscripts
    left_margins, right_margins = merge_close_lines(left_margins, right_margins, words, y_threshold=30)

    rectangles = dict([(box(xmin, ymin, xmax, ymax), (int(xmin), int(ymin), int(xmax), int(ymax))) for (xmin, ymin, xmax, ymax, p) in words])

    lines = []
    for l,r in zip(left_margins, right_margins):
        line = LineString([(l[0], l[1]), (r[0], r[1])])
        line_words = []
        for b in rectangles:
            if line.intersects(b):
                line_words.append(rectangles[b])
        # PERFORMANCE: Removed debug visualization (img2 rectangle drawing)
        # lw = line_words.copy()
        # for xmin, ymin, xmax, ymax in lw:
        #     cv2.rectangle(img2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        lines.append(sorted(line_words))


    # Detect background color from the original image
    # Use the median color value of the image as background
    # This works well for documents with light backgrounds
    flat_img = img.reshape(-1, 3)
    background_color = np.median(flat_img, axis=0).astype(np.uint8)
    logger.debug(f"Detected background color (BGR): {background_color}")

    all_letters = []
    all_lines = []
    for ln ,line in enumerate(lines):
        line_letters = find_rects(img, line)
        line_letters = sorted(line_letters, key=itemgetter(0))

        if not line_letters:
            continue

        # PERFORMANCE: Use NumPy arrays instead of list comprehensions
        line_letters_arr = np.array(line_letters)
        heights = line_letters_arr[:, 3] - line_letters_arr[:, 1]  # ymax - ymin

        m_height = np.median(heights)
        values, counts = np.unique(heights, return_counts=True)
        fh = values[np.argmax(counts)]
        sd = np.std(heights)

        # Filter normal letters
        normal_mask = np.abs(heights - m_height) < sd
        normal_letters = [tuple(ll) for i, ll in enumerate(line_letters) if normal_mask[i]]

        if not normal_letters:
            normal_letters = line_letters

        lower_points = [((xmin+xmax)/2, ymax) for xmin, ymin, xmax, ymax in normal_letters]

        try:
            if len(lower_points) > 1:
                x_coords = [x for x, y in lower_points]
                y_coords = [y for x, y in lower_points]
                m, c = np.polyfit(x_coords, y_coords, 1)
            else:
                m, c = 0, 0
        except:
            m, c = 0, 0
        letters = [Letter(xmin,ymin,xmax,ymax,ymax-ceil(m*((xmin+xmax)/2)+c)) for xmin,ymin,xmax,ymax in line_letters]
        all_letters.extend(letters)
        all_lines.append(letters)
       
        # PERFORMANCE: Removed debug visualization (img1 rectangle drawing)
        # red = (255,0,0)
        # green = (0,255,0)
        # for l in letters:
        #     if ln%2 == 0:
        #         cv2.rectangle(img1, (l.xmin,l.ymin), (l.xmax, l.ymax), red, 1)
        #     else:
        #         cv2.rectangle(img1, (l.xmin,l.ymin), (l.xmax, l.ymax), green, 1)

    page_with_letters = create_page_with_word_wrapping(all_lines, img, zoom_factor, new_page_width, background_color=tuple(background_color))
    return page_with_letters


def process_document_with_layout(filename, zoom_factor=2.5, new_page_width=2000):
    """
    Process a document using layout analysis to identify different content types.
    Plain text and titles are reflowed, while figures, tables, formulas etc. are zoomed and placed as-is.

    Args:
        filename: Path to the input image
        zoom_factor: Scaling factor for non-text content
        new_page_width: Width of the new page

    Returns:
        Reflowed page image
    """
    # Load the image
    img = cv2.imread(filename)
    img_h, img_w, _ = img.shape

    # Detect background color from the original image
    flat_img = img.reshape(-1, 3)
    background_color = np.median(flat_img, axis=0).astype(np.uint8)
    logger.debug(f"Detected background color (BGR): {background_color}")

    # Run layout analysis
    logger.debug("Running layout analysis...")
    try:
        layout_boxes = analyze_layout(filename)
    except RuntimeError as e:
        logger.error(f"Layout analysis failed: {e}")
        logger.info("Falling back to standard text-only processing...")
        return process_document(filename)
    except Exception as e:
        logger.error(f"Unexpected error during layout analysis: {e}")
        logger.info("Falling back to standard text-only processing...")
        return process_document(filename)

    # Sort boxes by y position (top to bottom), then x position (left to right)
    layout_boxes_sorted = sorted(layout_boxes, key=lambda item: (item[0].bounds[1], item[0].bounds[0]))

    logger.debug(f"Detected {len(layout_boxes_sorted)} layout boxes:")
    for box_geom, box_type in layout_boxes_sorted:
        bounds = box_geom.bounds
        logger.debug(f"  {box_type}: ({bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f})")

    # PERFORMANCE OPTIMIZATION: Use cached model instead of loading fresh
    model, device = get_doctr_model()

    # Lazy import DocumentFile only when needed
    from doctr.io import DocumentFile

    # Configuration
    left_margin = 50
    right_margin = 50
    top_margin = 50
    current_y = top_margin

    # Calculate available width for content
    available_width = new_page_width - left_margin - right_margin

    # Start with a reasonably sized page (will expand if needed)
    initial_page_height = 3000
    new_page = np.ones((initial_page_height, new_page_width, 3), dtype=np.uint8)
    new_page[:] = background_color

    # Process each layout box
    for box_geom, box_type in layout_boxes_sorted:
        bounds = box_geom.bounds
        xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])

        logger.debug(f"\nProcessing {box_type} box at y={ymin}")

        # Handle plain text and title - reflow these
        if box_type in ["plain text", "title"]:
            # Extract the region
            box_img = img[ymin:ymax, xmin:xmax].copy()

            # Save box_img temporarily to process with doctr
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, box_img)

            # Run text detection on this region
            docs = DocumentFile.from_images([tmp_path])
            result = model(docs)

            # Clean up temp file
            import os
            os.unlink(tmp_path)
            words = result[0]["words"]

            # Convert normalized coordinates to absolute
            box_h, box_w, _ = box_img.shape
            # Add more padding to word boxes to prevent letter clipping, especially for angled text
            # Increased from 2 to 5 pixels to ensure full letter capture
            words[:, 0] = (words[:, 0] * box_w).astype(np.int32) - 5  # left: expand left
            words[:, 1] = (words[:, 1] * box_h).astype(np.int32) - 5  # top: expand up
            words[:, 2] = (words[:, 2] * box_w).astype(np.int32) + 5  # right: expand right
            words[:, 3] = (words[:, 3] * box_h).astype(np.int32) + 5  # bottom: expand down
            # Clamp to image bounds
            words[:, 0] = np.maximum(words[:, 0], 0)
            words[:, 1] = np.maximum(words[:, 1], 0)
            words[:, 2] = np.minimum(words[:, 2], box_w)
            words[:, 3] = np.minimum(words[:, 3], box_h)
            words = words.astype(np.int32)

            if len(words) == 0:
                continue

            # Find left and right margins
            left_margins, right_margins = margins(words)

            # Merge lines that are too close together (fixes superscript/subscript issues)
            # Increased threshold to 30 to handle documents with slightly larger spacing between subscripts
            left_margins, right_margins = merge_close_lines(left_margins, right_margins, words, y_threshold=30)

            if len(left_margins) == 0 or len(right_margins) == 0:
                continue

            # Create rectangles for words
            rectangles = dict([
                (box(w_xmin, w_ymin, w_xmax, w_ymax),
                 (int(w_xmin), int(w_ymin), int(w_xmax), int(w_ymax)))
                for (w_xmin, w_ymin, w_xmax, w_ymax, _) in words
            ])

            # Group words into lines
            lines = []
            for l, r in zip(left_margins, right_margins):
                line = LineString([(l[0], l[1]), (r[0], r[1])])
                line_words = []
                for b in rectangles:
                    if line.intersects(b):
                        line_words.append(rectangles[b])
                if line_words:
                    lines.append(sorted(line_words))

            # Extract letters from lines
            all_lines = []
            for line in lines:
                line_letters = find_rects(box_img, line)
                line_letters = sorted(line_letters, key=itemgetter(0))

                if len(line_letters) == 0:
                    continue

                heights = [l_ymax - l_ymin for l_xmin, l_ymin, l_xmax, l_ymax in line_letters]
                m_height = np.median(heights)
                sd = np.std(heights) if len(heights) > 1 else 0

                normal_letters = [
                    (l_xmin, l_ymin, l_xmax, l_ymax)
                    for l_xmin, l_ymin, l_xmax, l_ymax in line_letters
                    if abs((l_ymax - l_ymin) - m_height) < sd
                ]

                if len(normal_letters) > 1:
                    lower_points = [((l_xmin + l_xmax) / 2, l_ymax) for l_xmin, l_ymin, l_xmax, l_ymax in normal_letters]
                    try:
                        x_coords = [x for x, y in lower_points]
                        y_coords = [y for x, y in lower_points]
                        m, c = np.polyfit(x_coords, y_coords, 1)
                    except:
                        m, c = 0, 0
                else:
                    m, c = 0, 0

                letters = [
                    Letter(l_xmin, l_ymin, l_xmax, l_ymax, l_ymax - ceil(m * ((l_xmin + l_xmax) / 2) + c))
                    for l_xmin, l_ymin, l_xmax, l_ymax in line_letters
                ]
                all_lines.append(letters)

            if len(all_lines) == 0:
                continue

            # Create a temporary page with reflowed text
            temp_page = create_page_with_word_wrapping(
                all_lines, box_img, zoom_factor, new_page_width,
                left_margin=left_margin, right_margin=right_margin,
                top_margin=0, bottom_margin=0,
                background_color=tuple(background_color)
            )

            # Find the actual height of content in temp_page
            # (scan from bottom to find last non-background row)
            temp_h = temp_page.shape[0]
            content_height = temp_h
            for row in range(temp_h - 1, -1, -1):
                if not np.all(temp_page[row] == background_color):
                    content_height = row + 1
                    break

            # Ensure we have enough space on the new page
            required_height = current_y + content_height + 50
            if required_height > new_page.shape[0]:
                # Expand the page
                new_height = max(required_height, new_page.shape[0] + 1000)
                expanded_page = np.ones((new_height, new_page_width, 3), dtype=np.uint8)
                expanded_page[:] = background_color
                expanded_page[:new_page.shape[0], :] = new_page
                new_page = expanded_page

            # Copy the reflowed content to the new page
            new_page[current_y:current_y + content_height, :] = temp_page[:content_height, :]
            current_y += content_height + 30  # Add spacing after text block

        else:
            # For figures, tables, formulas, etc. - zoom and place as-is
            box_img = img[ymin:ymax, xmin:xmax].copy()
            box_h, box_w, _ = box_img.shape

            # Calculate zoomed dimensions
            zoomed_h = int(box_h * zoom_factor)
            zoomed_w = int(box_w * zoom_factor)

            # If zoomed content is wider than available width, resize it
            if zoomed_w > available_width:
                scale = available_width / zoomed_w
                zoomed_w = available_width
                zoomed_h = int(zoomed_h * scale)

            # Resize the box
            resized_box = cv2.resize(box_img, (zoomed_w, zoomed_h), interpolation=cv2.INTER_CUBIC)

            # Ensure we have enough space on the new page
            required_height = current_y + zoomed_h + 50
            if required_height > new_page.shape[0]:
                # Expand the page
                new_height = max(required_height, new_page.shape[0] + 1000)
                expanded_page = np.ones((new_height, new_page_width, 3), dtype=np.uint8)
                expanded_page[:] = background_color
                expanded_page[:new_page.shape[0], :] = new_page
                new_page = expanded_page

            # Center the box horizontally
            x_offset = left_margin + (available_width - zoomed_w) // 2

            # Place the box on the new page
            new_page[current_y:current_y + zoomed_h, x_offset:x_offset + zoomed_w] = resized_box
            current_y += zoomed_h + 40  # Add spacing after non-text block

    # Crop the page to actual content
    final_height = current_y + 50  # Add bottom margin
    new_page = new_page[:final_height, :]

    return new_page


if __name__ == "__main__":
    import argparse

    # PERFORMANCE OPTIMIZATION: Better command-line argument parsing
    parser = argparse.ArgumentParser(description='Process document images with OCR and reflow')
    parser.add_argument('filename', help='Input image file path')
    parser.add_argument('--layout', action='store_true', help='Use layout-based processing')
    parser.add_argument('--no-output', action='store_true', help='Skip writing output images (for benchmarking)')
    parser.add_argument('--show-words', action='store_true', help='Generate word segmentation visualization')
    parser.add_argument('--page-width', type=int, default=2000, help='Width of the new page in pixels (default: 2000)')
    parser.add_argument('--zoom-factor', type=float, default=2.5, help='Scaling factor for letters (default: 2.5)')

    args = parser.parse_args()

    # Store as variables with the expected names
    new_page_width = args.page_width
    zoom_factor = args.zoom_factor

    filename = args.filename
    use_layout = args.layout

    if use_layout:
        if analyze_layout is None:
            logger.error("Layout analysis is not available (doclayout_yolo not installed).")
            logger.info("Falling back to standard text-only processing...")
            use_layout = False

    if use_layout:
        logger.info("Using layout-based processing...")
        page_with_letters = process_document_with_layout(filename, zoom_factor=zoom_factor, new_page_width=new_page_width)
    else:
        logger.info("Using original text-only processing...")
        page_with_letters = process_document(filename, zoom_factor=zoom_factor, new_page_width=new_page_width)

    # PERFORMANCE OPTIMIZATION: Make output writes optional
    if not args.no_output:
        output_filename = "output_reflowed.png"
        cv2.imwrite(output_filename, page_with_letters)
        logger.info(f"Output saved to: {output_filename}")
    else:
        logger.info("Skipping output write (--no-output flag)")

    # Word segmentation visualization (optional, off by default for performance)
    if args.show_words:
        logger.info("Creating word segmentation visualization...")
        img_with_words = cv2.imread(filename).copy()

        # Use cached model
        model, device = get_doctr_model()

        # Lazy import
        from doctr.io import DocumentFile

        docs = DocumentFile.from_images([filename])
        result = model(docs)
        words = result[0]["words"]

        # Convert normalized coordinates to absolute
        img_h, img_w, _ = img_with_words.shape
        words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
        words[:, 1] = (words[:, 1] * img_h).astype(np.int32)
        words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
        words[:, 3] = (words[:, 3] * img_h).astype(np.int32)
        words = words.astype(np.int32)

        logger.info(f"  Total words detected: {len(words)}")

        if not args.no_output:
            words_output_filename = "output_word_segmentation.png"
            cv2.imwrite(words_output_filename, img_with_words)
            logger.info(f"Word segmentation saved to: {words_output_filename}")
    else:
        logger.debug("Skipping word segmentation visualization (use --show-words to enable)")
