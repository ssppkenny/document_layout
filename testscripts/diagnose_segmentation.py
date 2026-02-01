#!/usr/bin/env python3
"""
Diagnostic script to analyze word segmentation for kf_16_par.png
"""

import sys
import os
import cv2
import numpy as np

# Don't import from ocr_reflow to avoid module issues
# Instead, we'll copy the margins function here
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from scipy.spatial import KDTree
from shapely import LineString, box
import shapely
from operator import itemgetter

def margins(words):
    """
    Detect left and right margins of text lines from word bounding boxes.
    Returns lists of (x, y) points representing the margin positions.

    Strategy:
    1. Filter out small words (subscripts/superscripts) using height threshold
    2. Cluster words by Y-position to identify lines using DBSCAN-like approach
    3. For each line, select the leftmost and rightmost word
    """
    # Return empty margins if too few words
    if len(words) < 2:
        return [], []

    # Calculate median word height for reference
    word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
    median_height = np.median(word_heights)

    # Height threshold - words must be at least 70% of median height to be margin candidates
    # This filters subscripts/superscripts which are typically 50-60% of normal text
    height_threshold = median_height * 0.70

    print(f"\nMargin detection parameters:")
    print(f"  Median word height: {median_height:.1f}px")
    print(f"  Height threshold (70%): {height_threshold:.1f}px")
    print(f"  Words below {height_threshold:.1f}px will be ignored as margin candidates")

    # Filter words by height - only keep normal-sized words for line detection
    filtered_words = []
    for xmin, ymin, xmax, ymax, conf in words:
        word_height = ymax - ymin
        if word_height >= height_threshold:
            filtered_words.append((xmin, ymin, xmax, ymax, conf))

    if len(filtered_words) < 2:
        return [], []

    # Cluster words by Y-position to identify lines
    # Use center Y coordinate for clustering
    word_data = []
    for xmin, ymin, xmax, ymax, conf in filtered_words:
        center_y = (ymin + ymax) / 2
        word_data.append({
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'center_y': center_y, 'height': ymax - ymin
        })

    # Sort by center Y
    word_data.sort(key=lambda w: w['center_y'])

    # Calculate adaptive threshold based on gap distribution
    # Strategy: Analyze gap distribution to separate intra-line from inter-line gaps
    gaps = []
    for i in range(1, len(word_data)):
        gap = word_data[i]['center_y'] - word_data[i-1]['center_y']
        gaps.append(gap)

    if len(gaps) < 3:
        # Too few gaps - use median
        gap_threshold = np.median(gaps) if gaps else median_height * 0.35
        print(f"\nAdaptive gap threshold: {gap_threshold:.1f}px (using median, few gaps)")
    else:
        # Use 75th percentile - conservative approach that works well across documents
        # This means 75% of gaps (intra-line) are below threshold, 25% (inter-line) above
        gap_threshold = np.percentile(gaps, 75)
        print(f"\nAdaptive gap threshold: {gap_threshold:.1f}px (using p75 percentile)")

    print(f"  Ratio to median height: {gap_threshold/median_height:.3f}×")

    # Group words into lines by detecting gaps in Y-position
    # Words with large Y-gaps between them are on different lines
    lines = []
    current_line = [word_data[0]]


    for i in range(1, len(word_data)):
        prev_word = word_data[i-1]
        curr_word = word_data[i]

        # Calculate gap between consecutive words (sorted by center Y)
        y_gap = curr_word['center_y'] - prev_word['center_y']

        if y_gap > gap_threshold:
            # Large gap - start new line
            lines.append(current_line)
            current_line = [curr_word]
        else:
            # Small gap - same line
            current_line.append(curr_word)

    # Don't forget the last line
    if current_line:
        lines.append(current_line)

    # For each line, find leftmost and rightmost word
    left_margin = []
    right_margin = []

    for line_words in lines:
        if not line_words:
            continue

        # Calculate average center_y for this line
        avg_y = sum(w['center_y'] for w in line_words) / len(line_words)

        # Find leftmost word (minimum xmin)
        leftmost = min(line_words, key=lambda w: w['xmin'])
        left_margin.append((int(leftmost['xmin']), int(avg_y)))

        # Find rightmost word (maximum xmax)
        rightmost = max(line_words, key=lambda w: w['xmax'])
        right_margin.append((int(rightmost['xmax']), int(avg_y)))

    return sorted(left_margin, key=itemgetter(1)), sorted(
        right_margin, key=itemgetter(1)
    )


def analyze_word_segmentation(image_path):
    """Analyze word segmentation and line detection."""

    print(f"Analyzing: {image_path}")
    print("=" * 80)

    # Load image
    img = cv2.imread(image_path)
    img_h, img_w, _ = img.shape
    print(f"\nImage dimensions: {img_w} x {img_h}")

    # Run text detection
    print("\nRunning text detection...")
    model = detection_predictor(pretrained=True)
    docs = DocumentFile.from_images([image_path])
    result = model(docs)
    words = result[0]["words"]

    print(f"Total words detected: {len(words)}")

    # Convert normalized coordinates to absolute
    words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
    words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
    words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
    words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
    words = words.astype(np.int32)

    # Analyze word positions
    print("\n" + "=" * 80)
    print("Word Details:")
    print("=" * 80)
    for i, (xmin, ymin, xmax, ymax, conf) in enumerate(words):
        width = xmax - xmin
        height = ymax - ymin
        center_y = (ymin + ymax) / 2
        print(f"Word {i:3d}: x=[{xmin:4d}, {xmax:4d}] y=[{ymin:4d}, {ymax:4d}] "
              f"w={width:3d} h={height:2d} center_y={center_y:6.1f} conf={conf:.3f}")

    # Detect margins
    print("\n" + "=" * 80)
    print("Detecting line margins...")
    print("=" * 80)

    left_margins, right_margins = margins(words)

    print(f"\nLeft margins found (before merging): {len(left_margins)}")
    for i, (x, y) in enumerate(left_margins):
        print(f"  Line {i}: left at ({x}, {y})")

    print(f"\nRight margins found (before merging): {len(right_margins)}")
    for i, (x, y) in enumerate(right_margins):
        print(f"  Line {i}: right at ({x}, {y})")

    # Create visualization BEFORE merging
    vis_img_before = img.copy()

    # Draw word rectangles in red
    for xmin, ymin, xmax, ymax, _ in words:
        cv2.rectangle(vis_img_before, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    # Draw detected lines BEFORE merging in orange
    for i, (l, r) in enumerate(zip(left_margins, right_margins)):
        cv2.line(vis_img_before, l, r, (0, 165, 255), 3)  # Orange line (BGR)
        # Draw circles at endpoints
        cv2.circle(vis_img_before, l, 6, (255, 0, 0), -1)  # Blue for left
        cv2.circle(vis_img_before, r, 6, (0, 255, 255), -1)  # Yellow for right
        # Add line number label
        mid_x = (l[0] + r[0]) // 2
        mid_y = (l[1] + r[1]) // 2
        cv2.putText(vis_img_before, f"L{i}", (mid_x - 20, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save visualization BEFORE merging
    output_path_before = "diagnostic_lines_before_merge.png"
    cv2.imwrite(output_path_before, vis_img_before)
    print(f"\n📊 Visualization BEFORE merging saved to: {output_path_before}")
    print(f"   - Orange lines: {len(left_margins)} detected lines")
    print(f"   - Red rectangles: detected words")
    print(f"   - Blue circles: left margins")
    print(f"   - Yellow circles: right margins")

    # Merge close lines
    from scipy.spatial import KDTree

    # Store the original line count before merging
    original_left_count = len(left_margins)
    original_right_count = len(right_margins)

    def merge_close_lines(left_margins, right_margins, words, y_threshold=50):
        """Improved merging with height analysis and adaptive threshold."""
        if len(left_margins) == 0 or len(right_margins) == 0:
            return left_margins, right_margins

        # Calculate adaptive threshold
        if len(left_margins) > 1:
            y_positions = [ly for _, ly in left_margins]
            gaps = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 50
            # Increased from 0.3 to 0.8 to handle documents where subscripts/superscripts
            # are further apart but still should be merged with main line
            adaptive_threshold = min(y_threshold, avg_gap * 0.8)
        else:
            adaptive_threshold = y_threshold

        print(f"\nAdaptive threshold: {adaptive_threshold:.1f} pixels")

        # Count words and analyze heights per line
        line_word_counts = []
        line_word_heights = []
        for i, (lx, ly) in enumerate(left_margins):
            word_count = 0
            heights = []
            for xmin, ymin, xmax, ymax, _ in words:
                word_center_y = (ymin + ymax) / 2
                if abs(word_center_y - ly) < y_threshold / 2:
                    word_count += 1
                    heights.append(ymax - ymin)
            line_word_counts.append(word_count)
            avg_height = np.median(heights) if heights else 0
            line_word_heights.append(avg_height)
            print(f"  Line {i} (y={ly}): {word_count} words, avg height={avg_height:.1f}px")

        # Merge lines - multiple passes
        merged_left = list(left_margins)
        merged_right = list(right_margins)
        word_counts = list(line_word_counts)
        word_heights = list(line_word_heights)

        changed = True
        max_iterations = 10
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
                merge_reason = ""
                if i + 1 < len(merged_left):
                    next_left = merged_left[i + 1]
                    next_right = merged_right[i + 1] if i + 1 < len(merged_right) else next_left
                    next_word_count = word_counts[i + 1]
                    next_height = word_heights[i + 1]

                    y_distance = abs(next_left[1] - current_left[1])

                    # Merge criteria
                    # 1. Very close lines with few words (superscripts/subscripts)
                    # Only merge if REALLY close (< 25 pixels) to avoid merging genuine short lines
                    if y_distance < 25 and (current_word_count <= 3 or next_word_count <= 3):
                        should_merge = True
                        merge_reason = f"few words (d={y_distance:.1f}px)"
                        changed = True
                    # 2. Very very close lines (within 20 pixels) regardless of word count
                    elif y_distance < 20:
                        should_merge = True
                        merge_reason = f"very close (d={y_distance:.1f}px)"
                        changed = True
                    # 3. Lines where one is much smaller in height (likely super/subscript)
                    # Use more generous distance for very small text (subscripts)
                    elif current_height > 0 and next_height > 0:
                        height_ratio = min(current_height, next_height) / max(current_height, next_height)
                        # For very small height ratio (< 0.65), allow larger distance (< 40px)
                        # For moderate height ratio (< 0.7), use adaptive threshold
                        if height_ratio < 0.65 and y_distance < 40 and (current_word_count <= 3 or next_word_count <= 3):
                            should_merge = True
                            merge_reason = f"small subscript (d={y_distance:.1f}px, ratio={height_ratio:.2f})"
                            changed = True
                        elif height_ratio < 0.7 and y_distance < adaptive_threshold:
                            should_merge = True
                            merge_reason = f"height diff (d={y_distance:.1f}px, ratio={height_ratio:.2f})"
                            changed = True

                if should_merge:
                    print(f"  Iteration {iteration}: Merging line {i} (y={current_left[1]}) with line {i+1} (y={next_left[1]}) - {merge_reason}")
                    next_left = merged_left[i + 1]
                    next_right = merged_right[i + 1] if i + 1 < len(merged_right) else next_left
                    next_word_count = word_counts[i + 1]
                    next_height = word_heights[i + 1]

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

    # Apply merging with increased threshold to handle slightly larger spacing
    merged_left_margins, merged_right_margins = merge_close_lines(left_margins, right_margins, words, y_threshold=30)

    print(f"\n" + "=" * 80)
    print(f"After merging close lines:")
    print("=" * 80)
    print(f"\nMerged left margins: {len(merged_left_margins)}")
    for i, (x, y) in enumerate(merged_left_margins):
        print(f"  Line {i}: left at ({x}, {y})")

    print(f"\nMerged right margins: {len(merged_right_margins)}")
    for i, (x, y) in enumerate(merged_right_margins):
        print(f"  Line {i}: right at ({x}, {y})")

    # Use merged margins for visualization
    left_margins = merged_left_margins
    right_margins = merged_right_margins

    # Create visualization AFTER merging
    vis_img_after = img.copy()

    # Draw word rectangles in red
    for xmin, ymin, xmax, ymax, _ in words:
        cv2.rectangle(vis_img_after, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    # Draw detected lines AFTER merging in green (thicker to show they are final)
    for i, (l, r) in enumerate(zip(left_margins, right_margins)):
        cv2.line(vis_img_after, l, r, (0, 255, 0), 4)  # Green line (thicker)
        # Draw circles at endpoints
        cv2.circle(vis_img_after, l, 7, (255, 0, 0), -1)  # Blue for left
        cv2.circle(vis_img_after, r, 7, (0, 255, 255), -1)  # Yellow for right
        # Add line number label
        mid_x = (l[0] + r[0]) // 2
        mid_y = (l[1] + r[1]) // 2
        cv2.putText(vis_img_after, f"L{i}", (mid_x - 20, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save visualization AFTER merging
    output_path_after = "diagnostic_lines_after_merge.png"
    cv2.imwrite(output_path_after, vis_img_after)
    print(f"\n✅ Visualization AFTER merging saved to: {output_path_after}")
    print(f"   - Green lines: {len(left_margins)} merged lines (FINAL)")
    print(f"   - Red rectangles: detected words")
    print(f"   - Blue circles: left margins")
    print(f"   - Yellow circles: right margins")

    # Also keep the old filename for backward compatibility
    output_path_legacy = "diagnostic_word_lines.png"
    cv2.imwrite(output_path_legacy, vis_img_after)
    print(f"\n   Legacy file: {output_path_legacy} (same as after_merge)")

    # Create a side-by-side comparison
    # Resize images if needed to fit side by side
    h1, w1 = vis_img_before.shape[:2]
    h2, w2 = vis_img_after.shape[:2]
    max_h = max(h1, h2)

    # Create canvas for comparison
    comparison = np.ones((max_h + 100, w1 + w2 + 30, 3), dtype=np.uint8) * 255

    # Add title
    cv2.putText(comparison, "BEFORE MERGING (Orange)", (w1//2 - 150, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(comparison, "AFTER MERGING (Green)", (w1 + 30 + w2//2 - 150, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Add line counts
    cv2.putText(comparison, f"{original_left_count} lines (before merge)",
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
    cv2.putText(comparison, f"{len(merged_left_margins)} lines (after merge)",
               (w1 + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Place images
    comparison[100:100+h1, 15:15+w1] = vis_img_before
    comparison[100:100+h2, w1+30:w1+30+w2] = vis_img_after

    # Save comparison
    output_path_compare = "diagnostic_lines_comparison.png"
    cv2.imwrite(output_path_compare, comparison)
    print(f"\n📊 Side-by-side comparison saved to: {output_path_compare}")
    print(f"   Shows both before ({original_left_count} lines) and after ({len(merged_left_margins)} lines) merging")
    print("  - Red rectangles: detected words")
    print("  - Green lines: detected text lines")
    print("  - Blue circles: left margins")
    print("  - Yellow circles: right margins")

    # Analyze Y-positions to see line grouping
    print("\n" + "=" * 80)
    print("Y-Position Analysis:")
    print("=" * 80)

    y_centers = [(ymin + ymax) / 2 for _, ymin, _, ymax, _ in words]
    y_centers_sorted = sorted(y_centers)

    # Find gaps between consecutive words (potential line breaks)
    print("\nY-center gaps between consecutive words:")
    for i in range(len(y_centers_sorted) - 1):
        gap = y_centers_sorted[i+1] - y_centers_sorted[i]
        if gap > 10:  # Only show significant gaps
            print(f"  Gap {i}: {gap:.1f} pixels (from y={y_centers_sorted[i]:.1f} to y={y_centers_sorted[i+1]:.1f})")

    # Cluster words by Y-position to identify lines
    print("\n" + "=" * 80)
    print("Line Clustering (by Y-position):")
    print("=" * 80)

    lines_dict = {}
    threshold = 15  # pixels - words within this Y distance are on the same line

    for i, (xmin, ymin, xmax, ymax, _) in enumerate(words):
        y_center = (ymin + ymax) / 2

        # Find which line this word belongs to
        found_line = False
        for line_y in lines_dict.keys():
            if abs(y_center - line_y) < threshold:
                lines_dict[line_y].append((xmin, i))
                found_line = True
                break

        if not found_line:
            lines_dict[y_center] = [(xmin, i)]

    # Sort lines by Y position
    sorted_lines = sorted(lines_dict.items(), key=lambda x: x[0])

    print(f"\nDetected {len(sorted_lines)} lines:")
    for line_num, (y_pos, word_list) in enumerate(sorted_lines):
        word_list_sorted = sorted(word_list, key=lambda x: x[0])
        word_indices = [idx for _, idx in word_list_sorted]
        print(f"\nLine {line_num + 1} (y ≈ {y_pos:.1f}):")
        print(f"  Words: {len(word_list)}")
        print(f"  Word indices: {word_indices}")

        # Show first and last word
        if word_indices:
            first_word = words[word_indices[0]]
            last_word = words[word_indices[-1]]
            line_width = last_word[2] - first_word[0]
            print(f"  X range: [{first_word[0]}, {last_word[2]}] (width: {line_width})")

    return len(sorted_lines), len(words), len(merged_left_margins)

if __name__ == "__main__":
    # Accept image path from command line, default to out2.png
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "images/out2.png"

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    num_lines, num_words, num_margins = analyze_word_segmentation(image_path)

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Expected lines: 7")
    print(f"Detected lines (by Y-clustering): {num_lines}")
    print(f"Detected lines (by margin detection, AFTER MERGING): {num_margins}")
    print(f"Total words detected: {num_words}")

    # Check the merged margin count (the accurate one) not Y-clustering
    if num_margins != 7:
        print(f"\n⚠️  WARNING: Expected 7 lines but margin detection found {num_margins}")
        if num_margins < 7:
            print("   Possible causes:")
            print("   - Lines were over-merged")
            print("   - Some lines are missing in detection")
        else:
            print("   Possible causes:")
            print("   - Superscripts/subscripts not fully merged")
            print("   - Need to adjust y_threshold or merge criteria")
    else:
        print(f"\n✅ Margin detection (merged) matches expected 7 lines!")
        if num_lines != 7:
            print(f"   Note: Y-clustering shows {num_lines} lines (less accurate method)")
