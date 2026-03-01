#!/usr/bin/env python3
"""
Visualize doctr word detection to debug letter clipping issues.
Shows the exact bounding boxes that doctr detects for words.
"""

import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import sys

def visualize_word_detection(image_path, output_path="word_detection_visualization.png"):
    """
    Visualize doctr word detection on an image.
    Draws rectangles around detected words and labels them.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    img_h, img_w = img.shape[:2]
    print(f"Image size: {img_w}x{img_h}")

    # Create a copy for drawing
    vis_img = img.copy()

    # Initialize doctr model
    print("Loading doctr model...")
    model = ocr_predictor(pretrained=True)

    # Run detection
    print("Running word detection...")
    docs = DocumentFile.from_images([image_path])
    result = model(docs)

    # Get words from doctr result - need to extract from blocks
    all_words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # Extract bounding box coordinates (geometry is relative)
                    bbox = word.geometry
                    # bbox format: ((xmin, ymin), (xmax, ymax))
                    all_words.append([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])

    words = np.array(all_words)
    print(f"Detected {len(words)} words")

    # Convert normalized coordinates to absolute
    abs_words = []
    for i, word in enumerate(words):
        xmin = int(word[0] * img_w)
        ymin = int(word[1] * img_h)
        xmax = int(word[2] * img_w)
        ymax = int(word[3] * img_h)
        abs_words.append((xmin, ymin, xmax, ymax))

        # Draw rectangle
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Add label with word index
        label = f"W{i}"
        cv2.putText(vis_img, label, (xmin, ymin - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print word info
        width = xmax - xmin
        height = ymax - ymin
        print(f"  Word {i}: ({xmin}, {ymin}) → ({xmax}, {ymax}), size: {width}x{height}")

    # Save visualization
    cv2.imwrite(output_path, vis_img)
    print(f"\nVisualization saved to: {output_path}")

    # Also show what happens with padding
    print("\n" + "="*80)
    print("WITH TITLE PADDING (35px):")
    print("="*80)

    padding = 35
    for i, (xmin, ymin, xmax, ymax) in enumerate(abs_words):
        padded_xmin = max(0, xmin - padding)
        padded_ymin = max(0, ymin - padding)
        padded_xmax = min(img_w, xmax + padding)
        padded_ymax = min(img_h, ymax + padding)

        width = padded_xmax - padded_xmin
        height = padded_ymax - padded_ymin
        print(f"  Word {i} WITH PADDING: ({padded_xmin}, {padded_ymin}) → ({padded_xmax}, {padded_ymax}), size: {width}x{height}")

    return abs_words

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_word_detection.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "word_detection_visualization.png"

    visualize_word_detection(image_path, output_path)
