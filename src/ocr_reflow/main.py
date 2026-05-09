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
    from binarization import binarize_document, normalize_image
    from layout import layout as analyze_layout
    from skew_detection import detect_and_correct_skew
    from toc_detection_mtd import detect_toc_page_mtd, get_toc_entry_lines
    from mtd_toc_detector import detect_toc_with_mtd, extract_entities_with_ocr
    from layoutlm_toc_detector import detect_toc_with_layoutlm
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

    try:
        from .binarization import binarize_document, normalize_image
    except ImportError as e:
        logger.warning(f"Could not import binarization module: {e}. Binarization will not be available.")
        binarize_document = None
        normalize_image = None

    # Layout is optional
    try:
        from .layout import layout as analyze_layout
    except ImportError as e:
        logger.warning(f"Could not import layout module: {e}. Layout analysis will not be available.")
        analyze_layout = None

    # TOC detection algorithms
    try:
        from .toc_detection_mtd import detect_toc_page_mtd, get_toc_entry_lines
    except ImportError as e:
        logger.warning(f"Could not import toc_detection_mtd module: {e}. Simple MTD algorithm will not be available.")
        detect_toc_page_mtd = None
        get_toc_entry_lines = None

    # Full MTD implementation with neural networks
    try:
        from .mtd_toc_detector import detect_toc_with_mtd, extract_entities_with_ocr
    except ImportError as e:
        logger.warning(f"Could not import mtd_toc_detector module: {e}. Full MTD algorithm will not be available.")
        detect_toc_with_mtd = None
        extract_entities_with_ocr = None

    # LayoutLMv3 pre-trained model
    try:
        from .layoutlm_toc_detector import detect_toc_with_layoutlm
    except ImportError as e:
        logger.warning(f"Could not import layoutlm_toc_detector module: {e}. LayoutLMv3 algorithm will not be available.")
        detect_toc_with_layoutlm = None

    # Skew detection is optional
    try:
        from .skew_detection import detect_and_correct_skew
    except ImportError as e:
        logger.warning(f"Could not import skew_detection module: {e}. Skew correction will not be available.")
        detect_and_correct_skew = None

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


def find_rects(img, line_words, debug=False, use_binarization=False):
    rects = []
    # Handle both formats: (xmin, ymin, xmax, ymax) or (xmin, ymin, xmax, ymax, confidence)
    for word_idx, word in enumerate(line_words):
        if len(word) == 5:
            xmin, ymin, xmax, ymax, _ = word  # Unpack 5 values, ignore confidence
        else:
            xmin, ymin, xmax, ymax = word  # Unpack 4 values
        word_height = ymax - ymin
        word_width = xmax - xmin

        if debug:
            print(f"    [find_rects] Word {word_idx}: box=({xmin},{ymin})→({xmax},{ymax}), size={word_width}x{word_height}")

        r = img[ymin:ymax,xmin:xmax,:].copy()
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r, 8, cv2.CV_32S)

        if debug:
            print(f"    [find_rects] Found {num_labels-1} connected components")


        # Strategy: First find "main" letter components (tall ones),
        # then include smaller components (dots, accents) that are vertically near them


        # Step 1: Identify main letter components (at least 30% of word height)
        main_components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter out obvious noise (too tiny)
            if w < 2 or h < 2 or area < 4:
                continue

            # Main letter bodies are typically at least 30% of word height
            if h >= word_height * 0.3:
                main_components.append(i)

        # Step 2: Collect all valid components
        # Include main components + small components near them (dots, accents, etc.)
        valid_components = []

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Filter out obvious noise (too tiny)
            if w < 2 or h < 2 or area < 4:
                continue

            # If it's a main component, always include it
            if i in main_components:
                valid_components.append((x, y, w, h))
                continue

            # For smaller components (dots, accents), check if they're vertically near a main component
            # This preserves dots on 'i', 'j', accents, etc.
            cy = y + h / 2

            is_near_main = False
            for main_idx in main_components:
                main_x = stats[main_idx, cv2.CC_STAT_LEFT]
                main_y = stats[main_idx, cv2.CC_STAT_TOP]
                main_w = stats[main_idx, cv2.CC_STAT_WIDTH]
                main_h = stats[main_idx, cv2.CC_STAT_HEIGHT]
                main_bottom = main_y + main_h

                # Check vertical proximity: component should be within reasonable distance
                # (above, below, or overlapping with main component)
                # Allow up to 40% of word height distance above the main component (for dots)
                max_distance_above = word_height * 0.4

                if y < main_bottom + max_distance_above and y + h > main_y - max_distance_above:
                    # Also check horizontal proximity (should be reasonably aligned)
                    # Components should have some horizontal overlap or be very close
                    horizontal_gap = max(0, max(x - (main_x + main_w), main_x - (x + w)))

                    if horizontal_gap < word_width * 0.3:  # Within 30% of word width
                        is_near_main = True
                        break

            if is_near_main:
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

        # Step 3: MERGE DOTS WITH BASE LETTERS
        # This ensures dots on 'i', 'j' stay perfectly aligned during reflow
        # Strategy: Find dot-letter pairs and merge them into single bounding boxes

        # Calculate median component height to classify dots vs letters
        if len(valid_components) > 0:
            component_heights = [h for x, y, w, h in valid_components]
            median_height = np.median(component_heights) if len(component_heights) > 0 else word_height * 0.5
        else:
            median_height = word_height * 0.5

        # Classify components as dots or main letters
        dots_to_merge = []
        main_letters_to_merge = []

        for comp_idx, (x, y, w, h) in enumerate(valid_components):
            # Diacritics: dots (i, j), breves (й), tildes, accents, Swedish ö/å/ä rings and dots, etc.
            # Use ONLY relative measures - no hardcoded pixels for resolution independence
            # Relative to median letter height and word dimensions

            # When binarization is enabled, be more lenient with diacritic detection
            # because binarization can make diacritics slightly larger
            if use_binarization:
                # More lenient thresholds for binarized images
                is_diacritic = (h < median_height * 0.5 and  # Increased from 0.4 to 0.5
                               w < median_height * 1.0 and  # Increased from 0.8 to 1.0 for wider diacritics
                               w * h < (median_height ** 2) * 0.4 and  # Increased from 0.3 to 0.4
                               h < word_height * 0.3 and  # Increased from 0.25 to 0.3
                               w < word_width * 0.6)  # Increased from 0.5 to 0.6
            else:
                # Original thresholds for non-binarized images
                is_diacritic = (h < median_height * 0.4 and
                               w < median_height * 0.8 and
                               w * h < (median_height ** 2) * 0.3 and
                               h < word_height * 0.25 and
                               w < word_width * 0.5)

            if is_diacritic:
                dots_to_merge.append((comp_idx, x, y, w, h))
            else:
                main_letters_to_merge.append((comp_idx, x, y, w, h))

        # PRE-PROCESSING: Merge horizontally adjacent main letter components
        # Handles letters like Russian и, ш, ж that are split into multiple vertical stems
        if len(main_letters_to_merge) > 1:
            merged_main_indices = set()
            merged_main_components = []


            for i, (idx_i, x_i, y_i, w_i, h_i) in enumerate(main_letters_to_merge):
                if idx_i in merged_main_indices:
                    continue

                # Find all components that should merge with this one
                merge_group = [(idx_i, x_i, y_i, w_i, h_i)]

                for j, (idx_j, x_j, y_j, w_j, h_j) in enumerate(main_letters_to_merge):
                    if i == j or idx_j in merged_main_indices:
                        continue

                    # Check if horizontally adjacent and vertically aligned
                    horizontal_gap = max(0, max(x_i - (x_j + w_j), x_j - (x_i + w_i)))
                    vertical_overlap = min(y_i + h_i, y_j + h_j) - max(y_i, y_j)

                    # For binarized images, also check vertical center alignment
                    # This helps merge split ö parts that are at the same baseline
                    y_center_i = y_i + h_i / 2
                    y_center_j = y_j + h_j / 2
                    vertical_center_distance = abs(y_center_i - y_center_j)

                    # Check height similarity to prevent tall letters (f, l, etc.) from merging with shorter ones
                    height_ratio = max(h_i, h_j) / max(min(h_i, h_j), 1)

                    # Determine if should merge based on image type
                    min_height = min(h_i, h_j)

                    if use_binarization:
                        # For binarized images: be more aggressive to handle split ö
                        # Split ö parts are at same vertical level with small horizontal gap
                        # BUT: don't merge letters with very different heights (e.g., f with ö)
                        should_merge = (horizontal_gap < median_height * 0.4 and  # Allow larger gap
                                       vertical_center_distance < median_height * 0.5 and  # Similar vertical position
                                       vertical_overlap > min_height * 0.3 and  # Some overlap required
                                       height_ratio < 1.4)  # Heights should be similar (< 40% difference)
                    else:
                        # Original logic for non-binarized images.
                        # Use a tighter gap threshold (0.3× median_height instead of 1×)
                        # and require similar heights to prevent narrow letters like "I"
                        # from being merged into adjacent wider letters like "n" or "d".
                        height_ratio = max(h_i, h_j) / max(min(h_i, h_j), 1)
                        should_merge = (horizontal_gap < median_height * 0.3 and
                                       vertical_overlap > min_height * 0.5 and
                                       height_ratio < 1.5)

                    if should_merge:
                        merge_group.append((idx_j, x_j, y_j, w_j, h_j))
                        merged_main_indices.add(idx_j)

                if len(merge_group) > 1:
                    # Merge all components in group
                    all_x = [x for _, x, y, w, h in merge_group]
                    all_y = [y for _, x, y, w, h in merge_group]
                    all_right = [x + w for _, x, y, w, h in merge_group]
                    all_bottom = [y + h for _, x, y, w, h in merge_group]

                    merged_x = min(all_x)
                    merged_y = min(all_y)
                    merged_w = max(all_right) - merged_x
                    merged_h = max(all_bottom) - merged_y

                    merged_main_components.append((idx_i, merged_x, merged_y, merged_w, merged_h))
                    merged_main_indices.add(idx_i)
                else:
                    # No merge needed, keep as-is
                    merged_main_components.append((idx_i, x_i, y_i, w_i, h_i))
                    merged_main_indices.add(idx_i)

            # Replace main_letters_to_merge with merged version
            main_letters_to_merge = merged_main_components


        # Find diacritic-letter groups and merge them
        # Key insight: A diacritic may sit above MULTIPLE base components (e.g., Russian й with two stems)
        merged_indices = set()
        merged_components = []

        for dot_idx, dx, dy, dw, dh in dots_to_merge:
            dot_cx = dx + dw / 2
            dot_bottom = dy + dh
            dot_left = dx
            dot_right = dx + dw

            # Find ALL main letter components below this diacritic
            matching_components = []

            for main_idx, mx, my, mw, mh in main_letters_to_merge:
                if main_idx in merged_indices:
                    continue

                main_cx = mx + mw / 2
                main_top = my
                main_left = mx
                main_right = mx + mw
                main_bottom = my + mh

                # Calculate vertical relationship
                # Diacritic can be: above (gap), touching, or slightly overlapping
                vertical_gap = main_top - dot_bottom  # Positive if gap exists

                # Diacritic should be above or very close to the main letter
                # Allow up to median_height distance (accommodates various font sizes)
                if vertical_gap < median_height:  # Diacritic is close enough above or overlapping
                    # Check horizontal alignment
                    horizontal_overlap = min(dot_right, main_right) - max(dot_left, main_left)

                    # CRITICAL FIX: Diacritics must have ACTUAL horizontal overlap with the letter
                    # This prevents dots above ö from merging with adjacent letters like B or k
                    # Only merge if there's positive overlap (diacritic is above the letter, not to the side)
                    is_horizontally_aligned = (horizontal_overlap > 0)

                    # Additional check: diacritic center should be reasonably aligned with letter
                    # Allow generous tolerance (50% of letter width) for slight misalignment
                    # This handles rings/dots that may not be perfectly centered
                    if is_horizontally_aligned:
                        horizontal_center_distance = abs(dot_cx - main_cx)
                        max_center_offset = mw * 0.5  # 50% of letter width (was 20%, too strict)
                        is_center_aligned = horizontal_center_distance < max_center_offset
                    else:
                        is_center_aligned = False

                    # And diacritic is above or at most slightly overlapping (not below)
                    is_vertically_ok = (dot_bottom <= main_bottom)

                    is_aligned = is_horizontally_aligned and is_center_aligned and is_vertically_ok

                    if is_aligned:
                        matching_components.append((main_idx, mx, my, mw, mh))

            if matching_components:
                # Merge diacritic with ALL matching base components
                all_components = [(dx, dy, dw, dh)] + [(mx, my, mw, mh) for _, mx, my, mw, mh in matching_components]

                merged_x = min(x for x, y, w, h in all_components)
                merged_y = min(y for x, y, w, h in all_components)
                merged_right = max(x + w for x, y, w, h in all_components)
                merged_bottom = max(y + h for x, y, w, h in all_components)
                merged_w = merged_right - merged_x
                merged_h = merged_bottom - merged_y

                merged_components.append((merged_x, merged_y, merged_w, merged_h))
                merged_indices.add(dot_idx)
                for main_idx, _, _, _, _ in matching_components:
                    merged_indices.add(main_idx)

        # Add letter components (including horizontally merged ones) that weren't merged with diacritics
        for main_idx, mx, my, mw, mh in main_letters_to_merge:
            if main_idx not in merged_indices:
                merged_components.append((mx, my, mw, mh))
                merged_indices.add(main_idx)

        # Add non-merged components as-is (diacritics that didn't match any letter)
        # IMPORTANT: Also skip components that were merged horizontally (tracked in merged_main_indices)
        # to avoid adding the same component twice (once as part of merged letter, once as original)
        skip_indices = set(merged_indices)
        if 'merged_main_indices' in locals():
            skip_indices.update(merged_main_indices)

        for comp_idx, (x, y, w, h) in enumerate(valid_components):
            if comp_idx not in skip_indices:
                merged_components.append((x, y, w, h))

        # Step 4: SPLIT WIDE COMPONENTS (for touching letters)
        # DISABLED: Previous splitting approaches caused issues:
        # - Morphological opening: Could damage letter shapes
        # - Full height projection: Too aggressive (1.5x threshold)
        # - Baseline projection: Caused skewed output on some pages
        #
        # For italic text issue: The real solution is to improve the word detection
        # rather than trying to fix merged letters after the fact

        final_components = merged_components
        splits_performed = 0



        # Step 5: FILTER SMALL FRAGMENTS
        # After merging and splitting, filter out very small components that are likely noise
        # These might be specs, artifacts, or letter fragments
        if len(final_components) > 2:  # Only filter if we have multiple components
            component_areas = [w * h for x, y, w, h in final_components]
            median_area = np.median(component_areas)

            # Keep components that are at least 25% of median area
            # This filters out specs while keeping legitimate small letters (like 'i' without dot)
            filtered_components = []
            for x, y, w, h in final_components:
                area = w * h
                # Keep if large enough area OR if tall enough — preserves narrow but
                # full-height letters like "I" whose area is small but height is normal.
                if area >= median_area * 0.25 or h >= median_height * 0.5:
                    filtered_components.append((x, y, w, h))

            if len(filtered_components) >= 2:  # Make sure we don't filter everything
                final_components = filtered_components

        # Now add all components (split, merged, and filtered) to rects with padding
        padding = 2
        for x, y, w, h in final_components:
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


def process_document(filename, zoom_factor=2.5, new_page_width=2000, apply_binarization=False, word_reflow=False, lang=None):
    """Process a document page and return a reflowed image.

    Args:
        filename: Path to an image file, OR a pre-loaded BGR numpy array
                  (e.g. from document_loader.load_page()).
        zoom_factor: Scaling factor for non-text content.
        new_page_width: Width of the output page.
        apply_binarization: Whether to binarize the image before processing.

    Returns:
        Reflowed page as a BGR numpy array.
    """
    import tempfile

    # Accept either a file path or a pre-loaded numpy array
    if isinstance(filename, np.ndarray):
        img = filename
        # Write to a temp file so DocumentFile.from_images() can read it
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            _preloaded_tmp = tmp.name
            cv2.imwrite(_preloaded_tmp, img)
        filename_for_processing = _preloaded_tmp
        _cleanup_preloaded = True
    else:
        _preloaded_tmp = None
        _cleanup_preloaded = False
        # Load image first
        img = cv2.imread(filename)

        # Detect and correct skew before any other processing
        if detect_and_correct_skew is not None:
            logger.info("Detecting and correcting skew...")
            img, skew_angle = detect_and_correct_skew(img)
            logger.info(f"Skew corrected: {skew_angle:.2f}°")

            # Save the corrected image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                corrected_filename = tmp.name
                cv2.imwrite(corrected_filename, img)

            # Use the corrected image for processing
            filename_for_processing = corrected_filename
        else:
            logger.warning("Skew detection not available, processing without skew correction")
            filename_for_processing = filename

    # PERFORMANCE OPTIMIZATION: Use cached model instead of loading fresh
    model, device = get_doctr_model()

    # Lazy import DocumentFile only when needed
    from doctr.io import DocumentFile

    # PERFORMANCE: Read image once instead of multiple times
    # If we created a corrected file, reload it; otherwise use the original img
    if not _cleanup_preloaded and filename_for_processing != filename:
        img = cv2.imread(filename_for_processing)

    img_h, img_w, _ = img.shape

    # Load and process image
    docs = DocumentFile.from_images([filename_for_processing])
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

    if word_reflow:
        try:
            from reflow_words import create_page_word_reflow, words_to_wordlines
        except ImportError:
            from ocr_reflow.reflow_words import create_page_word_reflow, words_to_wordlines

        word_lines = words_to_wordlines(lines)
        page_with_letters = create_page_word_reflow(
            word_lines, img, zoom_factor, new_page_width,
            find_rects_fn=find_rects,
            background_color=tuple(background_color),
            use_binarization=apply_binarization,
            lang=lang,
        )
    else:
        all_letters = []
        all_lines = []
        for ln ,line in enumerate(lines):
            line_letters = find_rects(img, line, use_binarization=apply_binarization)
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

        page_with_letters = create_page_with_word_wrapping(all_lines, img, zoom_factor, new_page_width, background_color=tuple(background_color))

    # Clean up temporary file if created for skew correction
    if detect_and_correct_skew is not None and not isinstance(filename, np.ndarray) and filename_for_processing != filename:
        try:
            os.unlink(filename_for_processing)
        except:
            pass

    # Clean up temp file written for pre-loaded numpy arrays
    if _cleanup_preloaded and _preloaded_tmp:
        try:
            os.unlink(_preloaded_tmp)
        except:
            pass

    return page_with_letters


def process_document_with_layout(filename, zoom_factor=2.5, new_page_width=2000, toc_algorithm='original', apply_binarization=False, word_reflow=False, lang=None):
    """
    Process a document using layout analysis to identify different content types.
    Plain text and titles are reflowed, while figures, tables, formulas etc. are zoomed and placed as-is.

    Args:
        filename: Path to the input image, OR a pre-loaded BGR numpy array
                  (e.g. from document_loader.load_page()).
        zoom_factor: Scaling factor for non-text content
        new_page_width: Width of the new page
        toc_algorithm: TOC detection algorithm to use: 'original' or 'mtd'

    Returns:
        Reflowed page image
    """
    import tempfile

    # Accept either a file path or a pre-loaded numpy array
    if isinstance(filename, np.ndarray):
        img = filename
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            _preloaded_tmp = tmp.name
            cv2.imwrite(_preloaded_tmp, img)
        filename = _preloaded_tmp
        _cleanup_preloaded = True
    else:
        _preloaded_tmp = None
        _cleanup_preloaded = False
        # Load the image
        img = cv2.imread(filename)

    img_h, img_w, _ = img.shape

    # Keep a copy of the original image for title processing
    # Title text with decorative fonts doesn't handle rotation well
    img_original = img.copy()

    # STEP 1: Run initial layout analysis to identify text regions
    logger.info("Running initial layout analysis for skew detection...")
    try:
        initial_layout_boxes = analyze_layout(filename)
    except RuntimeError as e:
        logger.error(f"Layout analysis failed: {e}")
        logger.info("Falling back to standard text-only processing...")
        return process_document(filename)
    except Exception as e:
        logger.error(f"Unexpected error during layout analysis: {e}")
        logger.info("Falling back to standard text-only processing...")
        return process_document(filename)

    # STEP 2: Detect and correct skew ONLY in text regions
    skew_corrected = False
    filename_for_layout = filename

    if detect_and_correct_skew is not None:
        # Import the text-region-aware skew detection
        try:
            from skew_detection import detect_skew_in_text_regions
        except ImportError:
            try:
                from .skew_detection import detect_skew_in_text_regions
            except ImportError:
                logger.warning("detect_skew_in_text_regions not available")
                detect_skew_in_text_regions = None

        if detect_skew_in_text_regions is not None:
            logger.info("Detecting skew in text regions only (ignoring figures/formulas)...")
            skew_angle = detect_skew_in_text_regions(img, initial_layout_boxes)

            if abs(skew_angle) > 0.1:
                logger.info(f"Correcting skew: {skew_angle:.2f}°")
                print(f"✓ Skew detected and corrected: {skew_angle:.2f}°")
                # Import rotate_image
                try:
                    from skew_detection import rotate_image
                except ImportError:
                    from .skew_detection import rotate_image

                img = rotate_image(img, skew_angle)
                skew_corrected = True

                # Save the corrected image temporarily for layout analysis
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    corrected_filename = tmp.name
                    cv2.imwrite(corrected_filename, img)

                filename_for_layout = corrected_filename
                logger.info(f"Skew corrected, image size now: {img_w}x{img_h}")
            else:
                logger.info(f"Skew angle {skew_angle:.2f}° too small, no correction needed")
                print(f"✓ Skew angle {skew_angle:.2f}° too small, no correction applied")
        else:
            logger.warning("Text-region skew detection not available, skipping skew correction")
    else:
        logger.warning("Skew detection not available, processing without skew correction")

    # Detect background color from the (possibly corrected) image
    flat_img = img.reshape(-1, 3)
    background_color = np.median(flat_img, axis=0).astype(np.uint8)
    logger.debug(f"Detected background color (BGR): {background_color}")

    # STEP 3: Run layout analysis on the corrected image
    logger.info("Running layout analysis on corrected image...")
    try:
        layout_boxes = analyze_layout(filename_for_layout)
    except RuntimeError as e:
        logger.error(f"Layout analysis failed: {e}")
        logger.info("Falling back to standard text-only processing...")

        # Clean up temp file if created
        if skew_corrected and filename_for_layout != filename:
            try:
                os.unlink(filename_for_layout)
            except:
                pass

        return process_document(filename)
    except Exception as e:
        logger.error(f"Unexpected error during layout analysis: {e}")
        logger.info("Falling back to standard text-only processing...")

        # Clean up temp file if created
        if skew_corrected and filename_for_layout != filename:
            try:
                os.unlink(filename_for_layout)
            except:
                pass

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

    img_h, img_w, _ = img.shape
    # zoom_factor is used as-is for letter scaling.
    # new_page_width is the output page width — independent of zoom_factor.

    # Derive margins from the first text block's x-position in the original image,
    # scaled by zoom_factor — no hardcoded pixel values.
    min_gap = max(1, int(img_h * zoom_factor * 0.003))  # ~0.3% of scaled page height
    text_geoms = [g for g, t in layout_boxes_sorted if t in ("plain text", "title")]
    if text_geoms:
        min_xmin = min(g.bounds[0] for g in text_geoms)
        max_xmax = max(g.bounds[2] for g in text_geoms)
        left_margin  = max(1, int((min_xmin / img_w) * new_page_width))
        right_margin = max(1, int(((img_w - max_xmax) / img_w) * new_page_width))
    else:
        left_margin  = int(new_page_width * 0.025)
        right_margin = int(new_page_width * 0.025)
    # Initial current_y = 0; gap_before for the first block will naturally encode
    # the distance from the top of the image to the first block.
    current_y = 0
    # Calculate available width for content
    available_width = new_page_width - left_margin - right_margin

    # NOTE: Fixed line height calculation removed - reflow.py now calculates optimal
    # line spacing per block using typography best practices (1.5x text height)
    # This allows each text block to have appropriate spacing for its font size
    """
    # PREPROCESSING: Calculate global fixed line height for consistent spacing
    # This ensures all text blocks use the same line height
    logger.info("Calculating global line height for consistent spacing...")
    global_max_letter_height = 0

    for box_geom, box_type in layout_boxes_sorted:
        if box_type in ["plain text", "title"]:
            bounds = box_geom.bounds
            xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
            box_img = img[ymin:ymax, xmin:xmax].copy()

            # Quick OCR to get letter sizes
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, box_img)

            try:
                docs = DocumentFile.from_images([tmp_path])
                result = model(docs)
                words = result[0]["words"]

                if len(words) > 0:
                    box_h, box_w, _ = box_img.shape
                    words[:, 0] = (words[:, 0] * box_w).astype(np.int32) - 5
                    words[:, 1] = (words[:, 1] * box_h).astype(np.int32) - 5
                    words[:, 2] = (words[:, 2] * box_w).astype(np.int32) + 5
                    words[:, 3] = (words[:, 3] * box_h).astype(np.int32) + 5
                    words[:, 0] = np.maximum(words[:, 0], 0)
                    words[:, 1] = np.maximum(words[:, 1], 0)
                    words[:, 2] = np.minimum(words[:, 2], box_w)
                    words[:, 3] = np.minimum(words[:, 3], box_h)

                    # Get letters to find max height
                    word_list = [(int(w[0]), int(w[1]), int(w[2]), int(w[3])) for w in words]
                    letters = find_rects(box_img, word_list, use_binarization=apply_binarization)

                    for lx1, ly1, lx2, ly2 in letters:
                        scaled_height = int((ly2 - ly1) * zoom_factor)
                        if scaled_height > global_max_letter_height:
                            global_max_letter_height = scaled_height
            finally:
                os.unlink(tmp_path)

    # Calculate fixed line height with extra space to prevent clipping
    # Use 1.5x max letter height to ensure enough space for descenders (g, p, y)
    # and tall letters with dots (merged i, j)
    if global_max_letter_height > 0:
        fixed_line_height = int(global_max_letter_height * 1.5)
        print(f"✓ Calculated fixed line height: {fixed_line_height}px (1.5x max letter: {global_max_letter_height}px)")
        logger.info(f"Using fixed line height: {fixed_line_height}px (max letter height: {global_max_letter_height}px)")
    else:
        fixed_line_height = 60  # Fallback
        print(f"⚠️  Using fallback line height: 60px")
        logger.warning("Could not determine letter heights, using fallback line height: 60px")
    """


    # Start with a reasonably sized page (will expand if needed)
    initial_page_height = 3000
    new_page = np.ones((initial_page_height, new_page_width, 3), dtype=np.uint8)
    new_page[:] = background_color

    # FIRST PASS: Detect if this is a TOC page by checking all plain text blocks
    # If ANY block looks like TOC, the whole page is very likely a TOC page
    page_is_toc = False
    print("\n" + "="*80)
    print("FIRST PASS: Checking if page is a Table of Contents")
    print(f"Using algorithm: {toc_algorithm.upper()}")
    print("="*80)

    # Choose TOC detection algorithm
    if toc_algorithm == 'layoutlm':
        # Use LayoutLMv3 pre-trained model (best option if available)
        if detect_toc_with_layoutlm is not None:
            print("[LayoutLMv3 Algorithm] Analyzing page structure...")
            print("  Using Microsoft's pre-trained LayoutLMv3 model")
            print("  Pre-trained on 11M documents for document understanding")

            # Run LayoutLMv3 detection on the entire page
            is_toc, confidence, metadata = detect_toc_with_layoutlm(filename, min_toc_entries=4)

            if is_toc:
                page_is_toc = True
                print(f"✓ DETECTED: Page is TOC (LayoutLMv3 confidence={confidence:.2f})")
                print(f"  → {metadata}")
                print(f"  → Treating ENTIRE PAGE as Table of Contents")
            else:
                print(f"✗ NOT TOC: LayoutLMv3 confidence={confidence:.2f}")
                if 'reason' in metadata:
                    print(f"  → {metadata['reason']}")
        else:
            print("[Warning] LayoutLMv3 requested but not available")
            print("  Install with: pip install transformers")
            print("  Falling back to original algorithm")
            toc_algorithm = 'original'

    elif toc_algorithm == 'mtd':
        # Check which MTD implementation is available
        has_full_mtd = detect_toc_with_mtd is not None
        has_simple_mtd = detect_toc_page_mtd is not None

        if has_full_mtd:
            # Use full MTD (Multimodal Tree Decoder) implementation with neural networks
            print("[MTD Algorithm with Neural Networks] Analyzing page structure...")
            print("  Using ResNet-34+FPN for vision, BERT for text, BiGRU for classification")

            # Run full MTD detection on the entire page
            is_toc, confidence, metadata = detect_toc_with_mtd(filename, min_headings=4)

            if is_toc:
                page_is_toc = True
                print(f"✓ DETECTED: Page is TOC (MTD confidence={confidence:.2f})")
                print(f"  → {metadata}")
                print(f"  → Treating ENTIRE PAGE as Table of Contents")
            else:
                print(f"✗ NOT TOC: MTD confidence={confidence:.2f}")
                if 'reason' in metadata:
                    print(f"  → {metadata['reason']}")

        elif has_simple_mtd:
            # Fallback to simple MTD if full implementation not available
            print("[MTD Algorithm - Simplified] Analyzing page structure...")
            print("  Note: Full MTD not available, using geometric heuristics")

            for box_geom, box_type in layout_boxes_sorted:
                if box_type not in ["plain text"]:
                    continue

                bounds = box_geom.bounds
                xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
                box_img = img[ymin:ymax, xmin:xmax].copy()
                box_h, box_w = box_img.shape[:2]

                # Quick word detection
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, box_img)

                docs = DocumentFile.from_images([tmp_path])
                result = model(docs)
                os.unlink(tmp_path)

                # Try to get text content from doctr result
                word_texts = []
                try:
                    doc_export = result.export()
                    if doc_export and 'pages' in doc_export:
                        page_data = doc_export['pages'][0]
                        for block in page_data.get('blocks', []):
                            for line in block.get('lines', []):
                                for word_data in line.get('words', []):
                                    word_texts.append(word_data.get('value', ''))
                except (AttributeError, KeyError, IndexError):
                    pass

                words = result[0]["words"]

                # Convert to absolute coordinates
                words[:, 0] = (words[:, 0] * box_w).astype(np.int32)
                words[:, 1] = (words[:, 1] * box_h).astype(np.int32)
                words[:, 2] = (words[:, 2] * box_w).astype(np.int32)
                words[:, 3] = (words[:, 3] * box_h).astype(np.int32)
                words = words.astype(np.int32)

                # Extract text for MTD algorithm
                word_list = [(int(w[0]), int(w[1]), int(w[2]), int(w[3])) for w in words]

                if not word_texts or len(word_texts) != len(word_list):
                    word_texts = [f"w{i}" for i in range(len(word_list))]

                # Run simplified MTD detection
                is_toc, confidence, metadata = detect_toc_page_mtd(
                    word_list,
                    word_texts,
                    box_w,
                    box_h,
                    min_toc_entries=4,
                    min_confidence=0.5
                )

                print(f"  [MTD] Block at y={ymin}: {metadata.get('num_entries', 0)} TOC entries, confidence={confidence:.2f}")

                if is_toc:
                    page_is_toc = True
                    print(f"✓ DETECTED: Block at y={ymin} is TOC (MTD confidence={confidence:.2f})")
                    print(f"  → {metadata}")
                    print(f"  → Treating ENTIRE PAGE as Table of Contents")
                    break

    else:
        # Use original rule-based algorithm
        if toc_algorithm == 'mtd':
            print("[Warning] MTD algorithm requested but not available, falling back to original")
        print("[Original Algorithm] Analyzing page structure...")

        for box_geom, box_type in layout_boxes_sorted:
            if box_type not in ["plain text"]:
                continue

            bounds = box_geom.bounds
            xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
            box_img = img[ymin:ymax, xmin:xmax].copy()
            box_h, box_w = box_img.shape[:2]

            # Quick word detection
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, box_img)

            docs = DocumentFile.from_images([tmp_path])
            result = model(docs)
            os.unlink(tmp_path)
            words = result[0]["words"]

            # Convert to absolute coordinates
            words[:, 0] = (words[:, 0] * box_w).astype(np.int32)
            words[:, 1] = (words[:, 1] * box_h).astype(np.int32)
            words[:, 2] = (words[:, 2] * box_w).astype(np.int32)
            words[:, 3] = (words[:, 3] * box_h).astype(np.int32)
            words = words.astype(np.int32)

            # Check TOC pattern
            num_words = len(words)
            print(f"  [TOC Check] Block at y={ymin}: {num_words} words")

            if num_words > 15:
                word_list = [(int(w[0]), int(w[1]), int(w[2]), int(w[3])) for w in words]
                from collections import defaultdict
                lines_dict = defaultdict(list)
                for word in word_list:
                    line_y = word[1] // 20
                    lines_dict[line_y].append(word)

                unique_y = len(lines_dict)
                print(f"  [TOC Check] Block at y={ymin}: {unique_y} lines detected")

                if unique_y >= 5:  # TOC blocks can have as few as 5 entries (lowered from 8 to catch smaller TOC sections)
                    right_aligned_lines = 0
                    rightmost_x_values = []
                    rightmost_widths = []  # Track widths of rightmost words
                    total_multi_word_lines = 0  # Count lines with 2+ words

                    for line_words in lines_dict.values():
                        if len(line_words) >= 2:
                            total_multi_word_lines += 1
                            sorted_words = sorted(line_words, key=lambda w: w[2])
                            rightmost_word = sorted_words[-1]
                            if rightmost_word[2] > box_w * 0.7:
                                right_aligned_lines += 1
                                rightmost_x_values.append(rightmost_word[2])
                                # Calculate width of rightmost word
                                word_width = rightmost_word[2] - rightmost_word[0]
                                rightmost_widths.append(word_width)

                    print(f"  [TOC Check] Block at y={ymin}: {right_aligned_lines} right-aligned lines (need ≥4)")

                    if right_aligned_lines >= 4 and len(rightmost_x_values) >= 4:
                        # Calculate what percentage of lines are right-aligned
                        alignment_ratio = right_aligned_lines / total_multi_word_lines if total_multi_word_lines > 0 else 0

                        # TOC pages should have at least 50% of lines right-aligned
                        # Justified text may have some right-aligned lines but not the majority
                        if alignment_ratio < 0.5:
                            print(f"  → Only {alignment_ratio:.0%} lines right-aligned - likely justified text, NOT TOC")
                            continue

                        x_std = np.std(rightmost_x_values)
                        x_median = np.median(rightmost_x_values)
                        alignment_score = x_std / x_median if x_median > 0 else 1.0

                        # Additional check: TOC pages have small rightmost words (page numbers)
                        # Justified text has normal-width words at right edge
                        median_rightmost_width = np.median(rightmost_widths)
                        avg_word_width = np.median([w[2] - w[0] for w in word_list])

                        # Four-tier threshold to distinguish TOC from justified text:
                        # Balanced to catch real TOC pages while rejecting justified text
                        # 1. Definitely: ratio < 0.60 = tiny page numbers (1-2 digits)
                        # 2. Probably: ratio < 0.70 = small page numbers (1-3 digits)
                        # 3. Likely: ratio < 0.78 = medium page numbers (Roman numerals like "xiii")
                        # 4. Possibly: ratio < 0.85 = larger page numbers (requires very tight alignment)
                        ratio = median_rightmost_width / avg_word_width if avg_word_width > 0 else 1.0

                        # CRITICAL CHECK: If ratio >= 0.9, rightmost words are too wide to be page numbers
                        # This is justified text, not TOC
                        if ratio >= 0.9:
                            print(f"  → Ratio {ratio:.2f} >= 0.9: rightmost words too wide - likely justified text, NOT TOC")
                            continue

                        # Tiered detection with progressively stricter alignment requirements
                        # Made slightly more lenient to catch all real TOC pages
                        is_definitely_toc = ratio < 0.60 and alignment_score < 0.08   # Tiny numbers, loose alignment OK
                        is_probably_toc = ratio < 0.70 and alignment_score < 0.05     # Small numbers, moderate alignment
                        is_likely_toc = ratio < 0.78 and alignment_score < 0.025      # Medium numbers (Roman), tight alignment
                        is_possibly_toc = ratio < 0.85 and alignment_score < 0.008    # Larger numbers, very tight alignment

                        if is_definitely_toc or is_probably_toc or is_likely_toc or is_possibly_toc:
                            page_is_toc = True
                            print(f"✓ DETECTED: Block at y={ymin} is TOC (alignment={alignment_score:.4f}, ratio={ratio:.2f}, {median_rightmost_width:.0f}px vs {avg_word_width:.0f}px, {alignment_ratio:.0%} lines aligned)")
                            print(f"  → Treating ENTIRE PAGE as Table of Contents")
                            break  # Stop checking - one TOC block means the whole page is TOC
                        else:
                            print(f"✗ NOT TOC at y={ymin}: Alignment={alignment_score:.4f}, ratio={ratio:.2f}")
                            print(f"  → Needs: ratio<0.85+align<0.005, or ratio<0.78+align<0.02, or ratio<0.70+align<0.04, or ratio<0.60")

    if not page_is_toc:
        print("✗ This page is NOT a Table of Contents - using regular reflow")

    print("="*80 + "\n")

    # Compute median reflowable block width to detect narrow blocks (verse stanzas, etc.)
    plain_text_widths = [g.bounds[2] - g.bounds[0] for g, t in layout_boxes_sorted
                         if t in ("plain text", "titled_block_body")]
    median_plain_width = float(np.median(plain_text_widths)) if plain_text_widths else img_w

    # SECOND PASS: Process each layout box
    pending_titled_title_img = None   # holds titled_block_title image until body arrives
    pending_titled_gap_before = None  # gap_before of the titled_block_title
    for idx, (box_geom, box_type) in enumerate(layout_boxes_sorted):
        bounds = box_geom.bounds
        xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])

        # Compute original gaps to previous/next block, scaled to output space
        prev_ymax = layout_boxes_sorted[idx - 1][0].bounds[3] if idx > 0 else 0
        next_ymin = layout_boxes_sorted[idx + 1][0].bounds[1] if idx < len(layout_boxes_sorted) - 1 else img_h
        gap_before = max(min_gap, int((bounds[1] - prev_ymax) * zoom_factor))
        gap_after  = max(min_gap, int((next_ymin - bounds[3]) * zoom_factor))

        logger.debug(f"\nProcessing {box_type} box at y={ymin}")

        # Narrow reflowable blocks (verse stanzas, indented quotes) are narrower than
        # typical prose — place them as a zoomed image to preserve line structure.
        block_width = xmax - xmin
        is_narrow_block = (box_type in ("plain text", "titled_block_body") and
                           block_width < median_plain_width * 0.65)
        # Handle plain text, title, and titled_block_body - reflow these (unless narrow)
        if box_type in ["plain text", "title", "titled_block_body"] and not is_narrow_block:

            # If this is a titled_block_body, first place the pending title image above it
            if box_type == "titled_block_body" and pending_titled_title_img is not None:
                t_img = pending_titled_title_img
                t_h, t_w = t_img.shape[:2]
                t_zoomed_h = int(t_h * zoom_factor)
                t_zoomed_w = int(t_w * zoom_factor)
                if t_zoomed_w > available_width:
                    scale = available_width / t_zoomed_w
                    t_zoomed_w = available_width
                    t_zoomed_h = int(t_zoomed_h * scale)
                t_resized = cv2.resize(t_img, (t_zoomed_w, t_zoomed_h), interpolation=cv2.INTER_CUBIC)
                t_ah, t_aw = t_resized.shape[:2]
                # Use the title's gap_before
                t_gap = pending_titled_gap_before if pending_titled_gap_before is not None else gap_before
                current_y += t_gap
                required_height = current_y + t_ah + 50
                if required_height > new_page.shape[0]:
                    new_height = max(required_height, new_page.shape[0] + 1000)
                    expanded_page = np.ones((new_height, new_page_width, 3), dtype=np.uint8)
                    expanded_page[:] = background_color
                    expanded_page[:new_page.shape[0], :] = new_page
                    new_page = expanded_page
                t_x = left_margin + (available_width - t_aw) // 2
                new_page[current_y:current_y + t_ah, t_x:t_x + t_aw] = t_resized
                current_y += t_ah
                pending_titled_title_img = None
                pending_titled_gap_before = None
                # gap_before for the body itself is the gap between title and body
                gap_before = max(min_gap, int((ymin - layout_boxes_sorted[idx-1][0].bounds[3]) * zoom_factor))

            # Extract the region from deskewed image (coordinates match layout boxes)
            box_img = img[ymin:ymax, xmin:xmax].copy()

            # Mask out any intersecting non-text regions (figures, tables, formulas)
            # to prevent OCR from trying to read table/figure content as text
            for other_geom, other_type in layout_boxes_sorted:
                # Skip if it's also a text box
                if other_type in ["plain text", "title", "titled_block_body"]:
                    continue

                # Check if this non-text box intersects with current text box
                if box_geom.intersects(other_geom):
                    # Calculate intersection in the text box's local coordinates
                    intersection = box_geom.intersection(other_geom)
                    inter_bounds = intersection.bounds

                    # Convert to local coordinates relative to the text box
                    local_xmin = max(0, int(inter_bounds[0] - xmin))
                    local_ymin = max(0, int(inter_bounds[1] - ymin))
                    local_xmax = min(box_img.shape[1], int(inter_bounds[2] - xmin))
                    local_ymax = min(box_img.shape[0], int(inter_bounds[3] - ymin))

                    # Fill the intersection area with background color to mask it out
                    if local_xmax > local_xmin and local_ymax > local_ymin:
                        box_img[local_ymin:local_ymax, local_xmin:local_xmax] = background_color
                        logger.debug(f"  Masked out intersection with {other_type} at local coords "
                                   f"({local_xmin}, {local_ymin}) → ({local_xmax}, {local_ymax})")

            # Save box_img temporarily to process with doctr
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, box_img)

            # Run text detection on this region
            docs = DocumentFile.from_images([tmp_path])
            result = model(docs)

            # Clean up temp file
            os.unlink(tmp_path)
            words = result[0]["words"]

            # Convert normalized coordinates to absolute
            box_h, box_w, _ = box_img.shape
            # Add padding to word boxes to prevent letter clipping
            # Use larger padding for titles which often have larger letters with descenders
            if box_type == "title":
                # Titles need generous padding for large letters and to prevent edge clipping
                padding = 35  # Very generous padding for decorative title letters (was 25, then 15)
            else:
                padding = 5  # Standard padding for normal text

            words[:, 0] = (words[:, 0] * box_w).astype(np.int32) - padding  # left
            words[:, 1] = (words[:, 1] * box_h).astype(np.int32) - padding  # top
            words[:, 2] = (words[:, 2] * box_w).astype(np.int32) + padding  # right
            words[:, 3] = (words[:, 3] * box_h).astype(np.int32) + padding  # bottom
            # Clamp to image bounds
            words[:, 0] = np.maximum(words[:, 0], 0)
            words[:, 1] = np.maximum(words[:, 1], 0)
            words[:, 2] = np.minimum(words[:, 2], box_w)
            words[:, 3] = np.minimum(words[:, 3], box_h)

            # Debug: show word count for titles
            if box_type == "title":
                print(f"  [Title at y={ymin}] Detected {len(words)} word(s)")

            # For title text, merge all word boxes into ONE to prevent over-segmentation
            if box_type == "title" and len(words) > 1:
                print(f"  [Title] Merging {len(words)} word boxes into one")
                logger.debug(f"  Title has {len(words)} word boxes, merging into one")
                # Create a single bounding box containing all words
                # Add extra padding to prevent letter clipping (especially first/last letters)
                extra_padding = 20  # Increased from 10 to better handle edge letters
                merged_xmin = max(0, int(np.min(words[:, 0])) - extra_padding)
                merged_ymin = max(0, int(np.min(words[:, 1])) - extra_padding)
                merged_xmax = min(box_w, int(np.max(words[:, 2])) + extra_padding)
                merged_ymax = min(box_h, int(np.max(words[:, 3])) + extra_padding)
                words = np.array([[merged_xmin, merged_ymin, merged_xmax, merged_ymax]])
                print(f"  [Title] Merged box: ({merged_xmin}, {merged_ymin}) → ({merged_xmax}, {merged_ymax})")

            words = words.astype(np.int32)

            if len(words) == 0:
                continue

            # Special handling for title blocks with single merged word
            if box_type == "title" and len(words) == 1:
                print(f"  [Title] Single merged word, processing directly")
                wx1, wy1, wx2, wy2 = words[0][:4]
                print(f"  [Title] Word box: ({wx1}, {wy1}) → ({wx2}, {wy2}), size: {wx2-wx1}x{wy2-wy1}")

                if word_reflow:
                    word_lines = [[(wx1, wy1, wx2, wy2)]]
                    all_lines = [[None]]  # non-empty placeholder
                else:
                    line_letters = find_rects(box_img, [(wx1, wy1, wx2, wy2)], debug=True, use_binarization=apply_binarization)
                    line_letters = sorted(line_letters, key=itemgetter(0))

                    # Debug: show first few letter boxes
                    if len(line_letters) > 0:
                        print(f"  [Title] Extracted {len(line_letters)} letters:")
                        for i, (lx1, ly1, lx2, ly2) in enumerate(line_letters[:5]):
                            print(f"    Letter {i}: ({lx1}, {ly1}) → ({lx2}, {ly2}), size: {lx2-lx1}x{ly2-ly1}")
                        if len(line_letters) > 5:
                            print(f"    ... and {len(line_letters)-5} more")

                    if len(line_letters) > 0:
                        heights = [l_ymax - l_ymin for l_xmin, l_ymin, l_xmax, l_ymax in line_letters]
                        m_height = np.median(heights)
                        sd = np.std(heights) if len(heights) > 1 else 0

                        normal_letters = [
                            (l_xmin, l_ymin, l_xmax, l_ymax)
                            for l_xmin, l_ymin, l_xmax, l_ymax in line_letters
                            if abs((l_ymax - l_ymin) - m_height) < sd
                        ]

                        # For titles: if no skew was corrected, assume horizontal baseline
                        # Decorative fonts can have slight variations that would create unwanted angles
                        if len(normal_letters) > 1:
                            if not skew_corrected:
                                # No skew detected/corrected -> force horizontal baseline
                                lower_points = [((l_xmin + l_xmax) / 2, l_ymax) for l_xmin, l_ymin, l_xmax, l_ymax in normal_letters]
                                y_coords = [y for x, y in lower_points]
                                m, c = 0, np.mean(y_coords)  # Horizontal baseline
                                print(f"  [Title] No skew detected -> forcing horizontal baseline (m=0, c={c:.1f})")
                            else:
                                # Skew was corrected -> calculate baseline angle normally
                                lower_points = [((l_xmin + l_xmax) / 2, l_ymax) for l_xmin, l_ymin, l_xmax, l_ymax in normal_letters]
                                try:
                                    x_coords = [x for x, y in lower_points]
                                    y_coords = [y for x, y in lower_points]
                                    m, c = np.polyfit(x_coords, y_coords, 1)
                                    print(f"  [Title] Skew was corrected -> calculated baseline angle (m={m:.4f})")
                                except:
                                    m, c = 0, 0
                        else:
                            m, c = 0, 0

                        letters = [
                            Letter(l_xmin, l_ymin, l_xmax, l_ymax, l_ymax - ceil(m * ((l_xmin + l_xmax) / 2) + c))
                            for l_xmin, l_ymin, l_xmax, l_ymax in line_letters
                        ]
                        all_lines = [letters]
                        print(f"  [Title] Extracted {len(letters)} letters from single word")
                    else:
                        all_lines = []
            else:
                # Normal processing for multi-word blocks
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
                if word_reflow:
                    word_lines = lines   # List[List[tuple(xmin,ymin,xmax,ymax)]]
                    all_lines = [line for line in lines if line]  # non-empty placeholder
                else:
                    all_lines = []
                    for line in lines:
                        line_letters = find_rects(box_img, line, use_binarization=apply_binarization)
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

            if box_type == "title":
                print(f"  [Title] Extracted {len(all_lines)} lines with total {sum(len(line) for line in all_lines)} letters")

            if len(all_lines) == 0 and not (word_reflow and word_lines):
                if box_type == "title":
                    print(f"  [Title] WARNING: all_lines is empty, skipping!")
                continue

            # Use page-level TOC flag determined in first pass
            is_toc_block = page_is_toc and box_type == "plain text"

            if is_toc_block:
                print(f"  [TOC] Using TOC reflow for plain text block at y={ymin}")

            is_preserve_lines = False
            block_alignment = 'left'

            # Create a temporary page with reflowed text
            # Use TOC-specific reflow if page is TOC, otherwise use regular reflow
            if is_toc_block:
                print(f"  ►►► USING TOC-SPECIFIC REFLOW ◄◄◄")
                try:
                    from .reflow import create_toc_page_with_right_alignment
                except ImportError:
                    from reflow import create_toc_page_with_right_alignment
                temp_page = create_toc_page_with_right_alignment(
                    all_lines, box_img, zoom_factor, new_page_width,
                    left_margin=left_margin, right_margin=right_margin,
                    top_margin=0, bottom_margin=0,
                    background_color=tuple(background_color)
                )
            elif word_reflow:
                try:
                    from reflow_words import create_page_word_reflow, words_to_wordlines
                except ImportError:
                    from ocr_reflow.reflow_words import create_page_word_reflow, words_to_wordlines
                temp_page = create_page_word_reflow(
                words_to_wordlines(word_lines), box_img, zoom_factor, new_page_width,
                find_rects_fn=find_rects,
                left_margin=left_margin, right_margin=right_margin,
                top_margin=0, bottom_margin=0,
                background_color=tuple(background_color),
                is_title=(box_type == "title"),
                use_binarization=apply_binarization,
                lang=lang,
                )
            else:
                # Use regular word wrapping reflow
                temp_page = create_page_with_word_wrapping(
                    all_lines, box_img, zoom_factor, new_page_width,
                    left_margin=left_margin, right_margin=right_margin,
                    top_margin=0, bottom_margin=0,
                    background_color=tuple(background_color),
                    is_title=(box_type == "title"),
                    preserve_line_breaks=is_preserve_lines,
                    alignment=block_alignment,
                )

            # Find the actual height of content in temp_page
            # (scan from bottom to find last non-background row)
            temp_h = temp_page.shape[0]
            content_height = temp_h
            for row in range(temp_h - 1, -1, -1):
                if not np.all(temp_page[row] == background_color):
                    content_height = row + 1
                    break

            if box_type == "title":
                print(f"  [Title] Reflowed page size: {temp_page.shape}, content_height: {content_height}, placing at y={current_y}")
                print(f"  [Title] len(all_lines)={len(all_lines)}")

            # Add gap before this block (from original spacing), for all text types
            current_y += gap_before

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

            # Advance by content height + gap after
            current_y += content_height + gap_after

        else:
            # For figures, tables, formulas, etc. - zoom and place as-is
            box_img = img[ymin:ymax, xmin:xmax].copy()

            # titled_block_title: hold until body arrives
            if box_type == "titled_block_title":
                pending_titled_title_img = box_img
                pending_titled_gap_before = gap_before
                continue

            # titled_block_body: stack title on top, then zoom and place as one image
            if box_type == "titled_block_body":
                if pending_titled_title_img is not None:
                    t_h, t_w = pending_titled_title_img.shape[:2]
                    b_h, b_w = box_img.shape[:2]
                    combined_w = max(t_w, b_w)
                    canvas_t = np.ones((t_h, combined_w, 3), dtype=np.uint8)
                    canvas_t[:] = background_color
                    canvas_t[:, :t_w] = pending_titled_title_img
                    canvas_b = np.ones((b_h, combined_w, 3), dtype=np.uint8)
                    canvas_b[:] = background_color
                    canvas_b[:, :b_w] = box_img
                    box_img = np.vstack([canvas_t, canvas_b])
                    # Use the title's gap_before (distance from prev block to title)
                    if pending_titled_gap_before is not None:
                        gap_before = pending_titled_gap_before
                    pending_titled_title_img = None
                    pending_titled_gap_before = None

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
            actual_h, actual_w = resized_box.shape[:2]  # use actual shape to avoid int rounding mismatch

            # Add gap before this block (from original spacing)
            current_y += gap_before

            # Ensure we have enough space on the new page
            required_height = current_y + actual_h + 50
            if required_height > new_page.shape[0]:
                # Expand the page
                new_height = max(required_height, new_page.shape[0] + 1000)
                expanded_page = np.ones((new_height, new_page_width, 3), dtype=np.uint8)
                expanded_page[:] = background_color
                expanded_page[:new_page.shape[0], :] = new_page
                new_page = expanded_page

            # Center the box horizontally
            x_offset = left_margin + (available_width - actual_w) // 2

            # Place the box on the new page
            new_page[current_y:current_y + actual_h, x_offset:x_offset + actual_w] = resized_box
            current_y += actual_h + gap_after

    # Crop the page to actual content
    # Bottom margin: scale the distance from last block bottom to image bottom
    if layout_boxes_sorted:
        last_ymax = layout_boxes_sorted[-1][0].bounds[3]
        bottom_margin = max(min_gap, int((img_h - last_ymax) * zoom_factor))
    else:
        bottom_margin = int(new_page_width * 0.025)
    final_height = current_y + bottom_margin
    new_page = new_page[:final_height, :]

    # Clean up temporary file if created for skew correction
    if skew_corrected and filename_for_layout != filename:
        try:
            os.unlink(filename_for_layout)
        except:
            pass

    # Clean up temp file written for pre-loaded numpy arrays
    if _cleanup_preloaded and _preloaded_tmp:
        try:
            os.unlink(_preloaded_tmp)
        except:
            pass

    return new_page


if __name__ == "__main__":
    import argparse
    import tempfile

    # PERFORMANCE OPTIMIZATION: Better command-line argument parsing
    parser = argparse.ArgumentParser(description='Process document images with OCR and reflow')
    parser.add_argument('filename', help='Input file path (image, PDF, or DjVu)')
    parser.add_argument('--layout', action='store_true', help='Use layout-based processing')
    parser.add_argument('--bin', action='store_true', help='Apply Otsu binarization before processing (helps with Swedish/Czech diacritics)')
    parser.add_argument('--no-output', action='store_true', help='Skip writing output images (for benchmarking)')
    parser.add_argument('--show-words', action='store_true', help='Generate word segmentation visualization')
    parser.add_argument('--page-width', type=int, default=2000, help='Width of the new page in pixels (default: 2000)')
    parser.add_argument('--zoom-factor', type=float, default=2.5, help='Scaling factor for letters (default: 2.5)')
    parser.add_argument('--toc-algorithm', type=str, default='layoutlm', choices=['original', 'mtd', 'layoutlm'],
                        help='TOC detection algorithm: "original" (rule-based), "mtd" (Multimodal Tree Decoder), or "layoutlm" (LayoutLMv3 pre-trained) (default: original)')
    parser.add_argument('--page', type=int, default=0, metavar='N',
                        help='0-based page number to process for PDF and DjVu files (default: 0)')
    parser.add_argument('--word-reflow', action='store_true',
                        help='Use word-level reflow (whole word crops as atomic units) instead of letter-level reflow')
    parser.add_argument('--lang', type=str, default=None,
                        help='Language code for hyphenation (e.g. ru, en, sv). Used with --word-reflow to select '
                             'pyphen dictionary and Tesseract OCR language for grammatical word splitting.')

    args = parser.parse_args()

    # Ensure LayoutLMv3 model is downloaded before processing begins
    if args.toc_algorithm == 'layoutlm':
        try:
            import model_manager as _mm
        except ImportError:
            try:
                from . import model_manager as _mm
            except ImportError:
                _mm = None

        if _mm is not None:
            try:
                _mm.get_layoutlmv3_toc_path()
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                sys.exit(1)
        else:
            print("WARNING: model_manager not available, model download check skipped.")

    # Store as variables with the expected names
    new_page_width = args.page_width
    zoom_factor = args.zoom_factor

    filename = args.filename
    use_layout = args.layout
    apply_binarization = args.bin

    # Load the page using document_loader (supports image, PDF, DjVu)
    try:
        from document_loader import load_page
    except ImportError:
        try:
            from ocr_reflow.document_loader import load_page
        except ImportError:
            load_page = None

    _main_tmp_file = None  # track any temp file we create here

    if load_page is not None:
        try:
            loaded_img = load_page(filename, args.page)
        except Exception as _e:
            logger.error(f"Could not load '{filename}' page {args.page}: {_e}")
            sys.exit(1)
        # Write to a temp PNG so the rest of the pipeline (which needs a path) works
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as _tmp:
            _main_tmp_file = _tmp.name
            cv2.imwrite(_main_tmp_file, loaded_img)
        filename = _main_tmp_file
    else:
        logger.warning("document_loader not available, falling back to cv2.imread")

    # Apply binarization if requested
    if apply_binarization:
        if binarize_document is None:
            logger.error("Binarization module is not available.")
            logger.info("Proceeding without binarization...")
            apply_binarization = False
        else:
            logger.info("Applying Otsu binarization to input image...")
            original_img = cv2.imread(filename)
            if original_img is None:
                logger.error(f"Could not read image: {filename}")
                sys.exit(1)

            # Apply normalization first for better results
            normalized_img = normalize_image(original_img)

            # Apply Otsu binarization
            binarized_img = binarize_document(normalized_img, method="otsu")

            # Otsu already gives us black text on white background, so no inversion needed

            # Convert back to BGR for consistency with the rest of the pipeline
            if len(binarized_img.shape) == 2:
                binarized_img = cv2.cvtColor(binarized_img, cv2.COLOR_GRAY2BGR)

            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_filename = tmp_file.name
                cv2.imwrite(temp_filename, binarized_img)
                filename = temp_filename  # Use binarized image for processing

            logger.info(f"Binarization complete. Processing binarized image...")

    if use_layout:
        if analyze_layout is None:
            logger.error("Layout analysis is not available (doclayout_yolo not installed).")
            logger.info("Falling back to standard text-only processing...")
            use_layout = False

    if use_layout:
        logger.info("Using layout-based processing...")
        page_with_letters = process_document_with_layout(filename, zoom_factor=zoom_factor, new_page_width=new_page_width, toc_algorithm=args.toc_algorithm, apply_binarization=apply_binarization, word_reflow=args.word_reflow, lang=args.lang)
    else:
        logger.info("Using original text-only processing...")
        page_with_letters = process_document(filename, zoom_factor=zoom_factor, new_page_width=new_page_width, apply_binarization=apply_binarization, word_reflow=args.word_reflow, lang=args.lang)

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
        img_with_words = cv2.imread(filename)
        if img_with_words is None:
            logger.error(f"Could not read image for word visualization: {filename}")
        else:
            img_with_words = img_with_words.copy()

            # Use cached model
            model, device = get_doctr_model()

            # Lazy import
            from doctr.io import DocumentFile

            docs = DocumentFile.from_images([filename])
            result = model(docs)
            words = result[0]["words"]

            # Convert normalized coordinates to absolute
            img_h, img_w, _ = img_with_words.shape
            words[:, 0] = (words[:, 0] * img_w).astype(np.int32) - 1
            words[:, 1] = (words[:, 1] * img_h).astype(np.int32) -1
            words[:, 2] = (words[:, 2] * img_w).astype(np.int32) + 1
            words[:, 3] = (words[:, 3] * img_h).astype(np.int32) + 1
            words = words.astype(np.int32)

            logger.info(f"  Total words detected: {len(words)}")

            if not args.no_output:
                words_output_filename = "output_word_segmentation.png"
                cv2.imwrite(words_output_filename, img_with_words)
                logger.info(f"Word segmentation saved to: {words_output_filename}")
    else:
        logger.debug("Skipping word segmentation visualization (use --show-words to enable)")

    # Clean up the temp file created from PDF/DjVu page rendering
    if _main_tmp_file:
        try:
            os.unlink(_main_tmp_file)
        except Exception:
            pass
