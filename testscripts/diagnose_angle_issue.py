"""
Diagnostic script to visualize line angle detection issues.
This script shows how the leftmost and rightmost points are detected
and whether they are properly centered in the middle of letter heights.
"""
import numpy as np
import cv2
import sys
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
from operator import itemgetter

def diagnose_angle_issue(filename):
    """Diagnose and visualize line angle detection issues."""

    # Load image
    img = cv2.imread(filename)
    if img is None:
        print(f"Error: Could not load image {filename}")
        return

    img_h, img_w, _ = img.shape

    # Run text detection
    model = detection_predictor(pretrained=True)
    docs = DocumentFile.from_images([filename])
    result = model(docs)
    words = result[0]["words"]

    # Convert to absolute coordinates
    words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
    words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
    words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
    words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2
    words = words.astype(np.int32)

    print(f"Detected {len(words)} words")

    # Calculate median word height
    word_heights = [(ymax - ymin) for _, ymin, _, ymax, _ in words]
    median_height = np.median(word_heights)
    height_threshold = median_height * 0.70

    print(f"Median word height: {median_height:.1f}")
    print(f"Height threshold: {height_threshold:.1f}")

    # Filter words by height
    filtered_words = []
    for xmin, ymin, xmax, ymax, conf in words:
        word_height = ymax - ymin
        if word_height >= height_threshold:
            filtered_words.append((xmin, ymin, xmax, ymax, conf))

    print(f"Filtered to {len(filtered_words)} normal-sized words")

    # Cluster words by Y-position
    word_data = []
    for xmin, ymin, xmax, ymax, conf in filtered_words:
        center_y = (ymin + ymax) / 2
        word_data.append({
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'center_y': center_y, 'height': ymax - ymin
        })

    # Sort by X position (left to right)
    word_data.sort(key=lambda w: w['xmin'])

    # Group words into lines using vertical overlap
    # Two words are on the same line if their Y-ranges overlap significantly
    lines = []
    used = [False] * len(word_data)

    for i, word in enumerate(word_data):
        if used[i]:
            continue

        # Start a new line with this word
        current_line = [word]
        used[i] = True

        # Find all other words that belong to this line
        # A word belongs if it has significant Y-overlap with any word already in the line
        changed = True
        while changed:
            changed = False
            for j, other_word in enumerate(word_data):
                if used[j]:
                    continue

                # Check if this word overlaps with any word in the current line
                for line_word in current_line:
                    # Calculate vertical overlap
                    overlap_top = max(other_word['ymin'], line_word['ymin'])
                    overlap_bottom = min(other_word['ymax'], line_word['ymax'])
                    overlap = max(0, overlap_bottom - overlap_top)

                    # Words are on the same line if they overlap by at least 40% of the smaller height
                    min_height = min(other_word['height'], line_word['height'])
                    if overlap >= 0.4 * min_height:
                        current_line.append(other_word)
                        used[j] = True
                        changed = True
                        break

        # Sort words in this line by X position
        current_line.sort(key=lambda w: w['xmin'])
        lines.append(current_line)

    # Sort lines by their vertical position (use the leftmost word's center_y)
    lines.sort(key=lambda line: line[0]['center_y'])

    print(f"\nDetected {len(lines)} lines")

    # Create visualization
    vis_img = img.copy()

    # For each line, analyze and visualize
    for line_idx, line_words in enumerate(lines):
        if not line_words:
            continue

        print(f"\n--- Line {line_idx + 1} ---")
        print(f"  Words in line: {len(line_words)}")

        # Calculate average center_y (OLD METHOD)
        avg_y = sum(w['center_y'] for w in line_words) / len(line_words)

        # Find leftmost and rightmost words
        leftmost = min(line_words, key=lambda w: w['xmin'])
        rightmost = max(line_words, key=lambda w: w['xmax'])

        # Calculate center Y for leftmost and rightmost words (NEW METHOD)
        leftmost_center_y = leftmost['center_y']
        rightmost_center_y = rightmost['center_y']

        print(f"  Leftmost word: x={leftmost['xmin']}-{leftmost['xmax']}, "
              f"y={leftmost['ymin']}-{leftmost['ymax']}, center_y={leftmost_center_y:.1f}")
        print(f"  Rightmost word: x={rightmost['xmin']}-{rightmost['xmax']}, "
              f"y={rightmost['ymin']}-{rightmost['ymax']}, center_y={rightmost_center_y:.1f}")
        print(f"  Y difference (rightmost - leftmost): {rightmost_center_y - leftmost_center_y:.1f}")

        # Draw rectangles around all words in the line
        color = ((line_idx * 50) % 256, (line_idx * 100) % 256, (line_idx * 150) % 256)
        for w in line_words:
            cv2.rectangle(vis_img, (w['xmin'], w['ymin']), (w['xmax'], w['ymax']), color, 2)

        # Draw NEW method line (using actual center_y of each word) in GREEN
        new_left_point = (int(leftmost['xmin']), int(leftmost_center_y))
        new_right_point = (int(rightmost['xmax']), int(rightmost_center_y))
        cv2.line(vis_img, new_left_point, new_right_point, (0, 255, 0), 3)
        cv2.circle(vis_img, new_left_point, 8, (0, 255, 0), -1)
        cv2.circle(vis_img, new_right_point, 8, (0, 255, 0), -1)

        # Highlight the leftmost and rightmost words
        cv2.rectangle(vis_img, (leftmost['xmin'], leftmost['ymin']),
                     (leftmost['xmax'], leftmost['ymax']), (255, 0, 255), 3)
        cv2.rectangle(vis_img, (rightmost['xmin'], rightmost['ymin']),
                     (rightmost['xmax'], rightmost['ymax']), (255, 255, 0), 3)

    # Save visualization
    output_file = "diagnostic_angle_issue.png"
    cv2.imwrite(output_file, vis_img)
    print(f"\n✓ Diagnostic visualization saved to: {output_file}")
    print("\nColor coding:")
    print("  GREEN line: Detected line using actual center Y of leftmost/rightmost words")
    print("  MAGENTA box: Leftmost word")
    print("  CYAN box: Rightmost word")

    return vis_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_angle_issue.py <image_path>")
        sys.exit(1)

    filename = sys.argv[1]
    diagnose_angle_issue(filename)
