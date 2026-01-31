import numpy as np
from math import ceil
import sys
import os

# Add current directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

from doctr.models import (
    detection_predictor,
)
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from doctr.io import DocumentFile
from shapely import LineString, box
import shapely
from operator import itemgetter
from dataclasses import dataclass

# Use conditional imports to support both script and module usage
try:
    from reflow import create_page_with_word_wrapping
    from divide_conquer_4d import divide_conquer_4d, Point4D
    from layout import layout as analyze_layout
except ImportError:
    from .reflow import create_page_with_word_wrapping
    from .divide_conquer_4d import divide_conquer_4d, Point4D
    from .layout import layout as analyze_layout

@dataclass
class Letter:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    bl: int


def find_rects(img, line_words):
    rects = []
    for xmin,ymin,xmax,ymax in line_words:
        r = img[ymin:ymax,xmin:xmax,:].copy()
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        _, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(r, 8, cv2.CV_32S)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            rects.append((x+xmin,y+ymin,x+w+xmin,y+h+ymin))
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
    Detect left and right margins of text lines from word bounding boxes.
    Returns lists of (x, y) points representing the margin positions.

    Enhanced to filter out subscripts/superscripts by using 75% of median height threshold.
    """
    # Return empty margins if too few words
    if len(words) < 2:
        return [], []

    # Calculate median word height for reference
    word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
    median_height = np.median(word_heights)

    # Height threshold - words must be at least 75% of median height to be margin candidates
    # This filters subscripts/superscripts which are typically 50-70% of normal text
    height_threshold = median_height * 0.75

    left_margin = []
    right_margin = []
    left_points = np.array(
        [[xmin, (ymin + ymax) / 2] for xmin, ymin, xmax, ymax, _ in words]
    )
    right_points = np.array(
        [[xmax, (ymin + ymax) / 2] for xmin, ymin, xmax, ymax, _ in words]
    )

    points = np.vstack((left_points, right_points))

    left_point_to_word = dict(
        [
            ((xmin, (ymin + ymax) / 2), (xmin, ymin, xmax, ymax))
            for xmin, ymin, xmax, ymax, _ in words
        ]
    )
    right_point_to_word = dict(
        [
            ((xmax, (ymin + ymax) / 2), (xmin, ymin, xmax, ymax))
            for xmin, ymin, xmax, ymax, _ in words
        ]
    )

    point_to_word = left_point_to_word | right_point_to_word

    kdtree = KDTree(points)
    # Limit k to the actual number of points available
    k_neighbors = min(50, len(points))
    dists_left, inds_left = kdtree.query(left_points, k=k_neighbors)
    dists_right, inds_right = kdtree.query(right_points, k=k_neighbors)

    # Process left margins
    for nbs_inds in inds_left:
        p_ind = nbs_inds[0]
        nbs_inds = nbs_inds[1:]
        # Filter out any invalid indices
        nbs_inds = nbs_inds[nbs_inds < len(points)]
        nbs = points[nbs_inds]
        x, y = points[p_ind]
        xmin1, ymin1, xmax1, ymax1 = point_to_word[(x, y)]


        # Check if this word is tall enough to be a margin candidate
        word_height = ymax1 - ymin1
        if word_height < height_threshold:
            continue  # Skip small words (subscripts/superscripts)

        points_to_side = []
        for nb in nbs:
            xmin, ymin, xmax, ymax = point_to_word[(nb[0], nb[1])]

            # Also check neighbor height - only consider similar-sized words
            neighbor_height = ymax - ymin
            if neighbor_height < height_threshold:
                continue  # Skip small neighbors

            ls1 = LineString([(0, ymin), (0, ymax)])
            ls2 = LineString([(0, ymin1), (0, ymax1)])
            s = shapely.intersection(ls1, ls2)
            m = min(abs(xmin-xmax), abs(xmin1-xmax1))
            mv = min(abs(ymin-ymax), abs(ymin1-ymax1))
            # Require 70% Y-overlap to consider words as being on the same line
            if (nb[0] <= x or abs(x-nb[0]) < m/2) and not s.is_empty and (s.length > 0.7*mv):
                points_to_side.append((nb[0], nb[1]))
        if len(points_to_side) == 0:
            left_margin.append((int(x), int(y)))

    # Process right margins
    for nbs_inds in inds_right:
        p_ind = nbs_inds[0]
        nbs_inds = nbs_inds[1:]
        # Filter out any invalid indices
        nbs_inds = nbs_inds[nbs_inds < len(points)]
        nbs = points[nbs_inds]
        x, y = points[p_ind]
        xmin1, ymin1, xmax1, ymax1 = point_to_word[(x, y)]


        # Check if this word is tall enough to be a margin candidate
        word_height = ymax1 - ymin1
        if word_height < height_threshold:
            continue  # Skip small words (subscripts/superscripts)

        points_to_side = []
        for nb in nbs:
            xmin, ymin, xmax, ymax = point_to_word[(nb[0], nb[1])]

            # Also check neighbor height - only consider similar-sized words
            neighbor_height = ymax - ymin
            if neighbor_height < height_threshold:
                continue  # Skip small neighbors

            ls1 = LineString([(0, ymin), (0, ymax)])
            ls2 = LineString([(0, ymin1), (0, ymax1)])
            s = shapely.intersection(ls1, ls2)
            m = min(abs(xmin-xmax), abs(xmin1-xmax1))
            mv = min(abs(ymin-ymax), abs(ymin1-ymax1))
            # Require 70% Y-overlap to consider words as being on the same line
            if (nb[0] >= x or abs(x-nb[0]) < m/2) and not s.is_empty and (s.length > 0.7*mv):
                points_to_side.append((nb[0], nb[1]))
        if len(points_to_side) == 0:
            right_margin.append((int(x), int(y)))

    return sorted(left_margin, key=itemgetter(1)), sorted(
        right_margin, key=itemgetter(1)
    )


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
        # Use the smaller of provided threshold or 0.3x average gap
        adaptive_threshold = min(y_threshold, avg_gap * 0.3)
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
                if y_distance < y_threshold and (current_word_count <= 3 or next_word_count <= 3):
                    should_merge = True
                    changed = True
                # 2. Very very close lines (within 20 pixels) regardless of word count
                elif y_distance < 20:
                    should_merge = True
                    changed = True
                # 3. Lines where one is much smaller in height (likely super/subscript)
                elif y_distance < adaptive_threshold and current_height > 0 and next_height > 0:
                    height_ratio = min(current_height, next_height) / max(current_height, next_height)
                    if height_ratio < 0.7:  # One line has significantly smaller text
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


def process_document(filename):
    model = detection_predictor(pretrained=True)
    # filename = "dvurog_p007.png"
    docs = DocumentFile.from_images([filename])
    img = cv2.imread(filename)
    img_h, img_w, _ = img.shape
    result = model(docs)
    words = result[0]["words"]
    words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
    words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
    words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
    words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
    words = words.astype(np.int32)

    img = cv2.imread(filename)
    img1 = cv2.imread(filename)
    img2 = cv2.imread(filename)
    left_margins, right_margins = margins(words)

    # Merge lines that are too close together (fixes superscript/subscript issues)
    left_margins, right_margins = merge_close_lines(left_margins, right_margins, words, y_threshold=20)

    rectangles = dict([(box(xmin, ymin, xmax, ymax), (int(xmin), int(ymin), int(xmax), int(ymax))) for (xmin, ymin, xmax, ymax, p) in words])

    lines = []
    for l,r in zip(left_margins, right_margins):
        line = LineString([(l[0], l[1]), (r[0], r[1])])
        line_words = []
        for b in rectangles:
            if line.intersects(b):
                line_words.append(rectangles[b])
        lw = line_words.copy()
        for xmin, ymin, xmax,ymax in lw:
            cv2.rectangle(img2, (xmin,ymin), (xmax, ymax), (255,0,0), 1)
        lines.append(sorted(lw))

    # Configuration parameters moved outside the loop
    zoom_factor = 2.5
    new_page_width = 2000

    # Detect background color from the original image
    # Use the median color value of the image as background
    # This works well for documents with light backgrounds
    flat_img = img.reshape(-1, 3)
    background_color = np.median(flat_img, axis=0).astype(np.uint8)
    print(f"Detected background color (BGR): {background_color}")

    all_letters = []
    all_lines = []
    for ln ,line in enumerate(lines):
        line_letters = find_rects(img, line)
        line_letters = sorted(line_letters, key=itemgetter(0))
        heights = [ymax - ymin for xmin,ymin,xmax,ymax in line_letters]
        m_height = np.median(heights)
        values, counts = np.unique(heights, return_counts=True)
        fh = values[np.argmax(counts)]
        sd = np.std(heights)
        normal_letters = [(xmin,ymin,xmax,ymax) for xmin,ymin,xmax,ymax in line_letters if abs((ymax-ymin)-m_height) < sd]
        lower_points = [((xmin+xmax)/2,ymax) for xmin,ymin,xmax,ymax in normal_letters]
        try:
            x_coords = [x for x,y in lower_points]
            y_coords = [y for x,y in lower_points]
            m, c = np.polyfit(x_coords, y_coords, 1)
            # cv2.line(img, (int(x_coords[0]), int(m*x_coords[0]+c)), (int(x_coords[-1]), int(np.ceil(m*x_coords[-1]+c))), (255,0,0), 2)
        except:
            m, c = 0, 0
        letters = [Letter(xmin,ymin,xmax,ymax,ymax-ceil(m*((xmin+xmax)/2)+c)) for xmin,ymin,xmax,ymax in line_letters]
        all_letters.extend(letters)
        all_lines.append(letters)
       
        red = (255,0,0)
        green = (0,255,0)
        for l in letters:
            if ln%2 == 0:
                cv2.rectangle(img1, (l.xmin,l.ymin), (l.xmax, l.ymax), red, 1)
            else:
                cv2.rectangle(img1, (l.xmin,l.ymin), (l.xmax, l.ymax), green, 1)

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
    print(f"Detected background color (BGR): {background_color}")

    # Run layout analysis
    print("Running layout analysis...")
    layout_boxes = analyze_layout(filename)

    # Sort boxes by y position (top to bottom), then x position (left to right)
    layout_boxes_sorted = sorted(layout_boxes, key=lambda item: (item[0].bounds[1], item[0].bounds[0]))

    print(f"Detected {len(layout_boxes_sorted)} layout boxes:")
    for box_geom, box_type in layout_boxes_sorted:
        bounds = box_geom.bounds
        print(f"  {box_type}: ({bounds[0]:.1f}, {bounds[1]:.1f}, {bounds[2]:.1f}, {bounds[3]:.1f})")

    # Initialize the doctr model for text detection within text boxes
    model = detection_predictor(pretrained=True)

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

        print(f"\nProcessing {box_type} box at y={ymin}")

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
            words[:, 0] = (words[:, 0] * box_w).astype(np.int32)
            words[:, 1] = (words[:, 1] * box_h).astype(np.int32) + 2
            words[:, 2] = (words[:, 2] * box_w).astype(np.int32)
            words[:, 3] = (words[:, 3] * box_h).astype(np.int32) - 2
            words = words.astype(np.int32)

            if len(words) == 0:
                continue

            # Find left and right margins
            left_margins, right_margins = margins(words)

            # Merge lines that are too close together (fixes superscript/subscript issues)
            left_margins, right_margins = merge_close_lines(left_margins, right_margins, words, y_threshold=20)

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
    filename = sys.argv[1]

    # Check if user wants layout-based processing
    use_layout = len(sys.argv) > 2 and sys.argv[2] == "--layout"

    if use_layout:
        print("Using layout-based processing...")
        page_with_letters = process_document_with_layout(filename)
    else:
        print("Using original text-only processing...")
        page_with_letters = process_document(filename)

    # Save the output
    output_filename = "output_reflowed.png"
    cv2.imwrite(output_filename, page_with_letters)
    print(f"\nOutput saved to: {output_filename}")
    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(page_with_letters, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("output_reflowed_preview.png", dpi=150, bbox_inches='tight')
    print(f"Preview saved to: output_reflowed_preview.png")

    # Create word segmentation visualization
    print("\nCreating word segmentation visualization...")
    img_with_words = cv2.imread(filename).copy()

    # Run text detection to get words
    model = detection_predictor(pretrained=True)
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

    # Draw red rectangles around each word
    for xmin, ymin, xmax, ymax, _ in words:
        cv2.rectangle(img_with_words, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    # Save the visualization
    words_output_filename = "output_word_segmentation.png"
    cv2.imwrite(words_output_filename, img_with_words)
    print(f"Word segmentation visualization saved to: {words_output_filename}")
    print(f"  Total words detected: {len(words)}")
