#!/usr/bin/env python3
"""
Diagnose Swedish diacritic segmentation issues with binarized images.
Specifically focusing on ö and å characters.
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src/ocr_reflow')

from doctr.models import detection_predictor
import torch

def apply_otsu_binarization(img):
    """Apply Otsu binarization to image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert so text is white on black
    binary = cv2.bitwise_not(binary)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# Load model once globally
print("Loading doctr model...")
detection_model = detection_predictor(arch='db_resnet50', pretrained=True, assume_straight_pages=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detection_model = detection_model.to(device)
print(f"Model loaded on {device}")

def visualize_word_components(img, word_idx, label, output_path):
    """Visualize connected components for a specific word."""
    # Apply binarization
    bin_img = apply_otsu_binarization(img)

    # Get word detection - use global model
    global detection_model
    result = detection_model([img])

    # Extract words from result - it's a list with one dict containing 'words' key
    words = result[0]['words']

    if word_idx >= len(words):
        print(f"Word index {word_idx} out of range (total: {len(words)})")
        return

    word = words[word_idx]

    # Get geometry - word is a 1D numpy array [x1, y1, x2, y2, confidence]
    h, w = img.shape[:2]

    # Safety check
    if not hasattr(word, 'shape') or len(word.shape) != 1 or word.shape[0] < 4:
        print(f"ERROR: Unexpected word format: {type(word)}, {word}")
        return

    xmin = int(word[0] * w)
    ymin = int(word[1] * h)
    xmax = int(word[2] * w)
    ymax = int(word[3] * h)

    print(f"\n{'='*80}")
    print(f"Analyzing {label} (W{word_idx+1}): ({xmin},{ymin}) -> ({xmax},{ymax})")
    print(f"Size: {xmax-xmin}x{ymax-ymin}")

    # Extract word region from binarized image
    word_region = bin_img[ymin:ymax, xmin:xmax].copy()
    word_gray = cv2.cvtColor(word_region, cv2.COLOR_BGR2GRAY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        word_gray, connectivity=8
    )

    print(f"Found {num_labels-1} connected components (excluding background)")

    # Create visualization
    vis_height = word_region.shape[0]
    vis_width = word_region.shape[1]

    # Create a large canvas for visualization
    canvas_height = vis_height * 3 + 80
    canvas_width = max(vis_width * 2 + 40, 800)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 1. Original word region (top left)
    y_offset = 10
    x_offset = 10
    canvas[y_offset:y_offset+vis_height, x_offset:x_offset+vis_width] = word_region
    cv2.putText(canvas, "Original (binarized)", (x_offset, y_offset-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 2. Original from source image (top right)
    x_offset2 = x_offset + vis_width + 20
    orig_word = img[ymin:ymax, xmin:xmax].copy()
    canvas[y_offset:y_offset+vis_height, x_offset2:x_offset2+vis_width] = orig_word
    cv2.putText(canvas, "Original (no binarization)", (x_offset2, y_offset-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 3. Connected components visualization (middle)
    y_offset2 = y_offset + vis_height + 30
    comp_vis = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 255

    # Color each component differently
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]

    component_info = []
    for i in range(1, num_labels):  # Skip background (label 0)
        mask = (labels == i).astype(np.uint8) * 255
        color = colors[(i-1) % len(colors)]
        comp_vis[mask > 0] = color

        # Get bounding box
        x, y, w_comp, h_comp, area = stats[i]
        cx, cy = centroids[i]

        component_info.append({
            'id': i,
            'bbox': (x, y, w_comp, h_comp),
            'area': area,
            'centroid': (cx, cy),
            'color': color
        })

        # Draw bounding box
        cv2.rectangle(comp_vis, (x, y), (x+w_comp, y+h_comp), color, 1)
        cv2.putText(comp_vis, f"{i}", (int(cx)-5, int(cy)+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    canvas[y_offset2:y_offset2+vis_height, x_offset:x_offset+vis_width] = comp_vis
    cv2.putText(canvas, "Connected Components (colored)", (x_offset, y_offset2-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # 4. Component details (bottom)
    y_text = y_offset2 + vis_height + 20
    cv2.putText(canvas, "Component Details:", (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    y_text += 20

    # Sort by x position (left to right)
    component_info.sort(key=lambda c: c['bbox'][0])

    for comp in component_info:
        x, y, w_comp, h_comp = comp['bbox']
        text = f"  #{comp['id']}: x={x:3d} y={y:3d} w={w_comp:2d} h={h_comp:2d} area={comp['area']:4d}"
        cv2.putText(canvas, text, (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, comp['color'], 1)
        y_text += 15

    # Print component info
    print("\nComponents (sorted left to right):")
    for comp in component_info:
        x, y, w_comp, h_comp = comp['bbox']
        print(f"  #{comp['id']}: x={x:3d} y={y:3d} w={w_comp:2d} h={h_comp:2d} area={comp['area']:4d}")

    # Save visualization
    cv2.imwrite(output_path, canvas)
    print(f"\n✓ Saved visualization to {output_path}")
    print(f"{'='*80}\n")

def main():
    img_path = 'images/gang_p023_lines1.png'

    if not os.path.exists(img_path):
        print(f"ERROR: {img_path} not found")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read {img_path}")
        return

    print(f"Analyzing Swedish diacritic issues in {img_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Words identified by user:
    # W16 - blåste (å), W18 - rök (ö), W51 - för (ö), W22 - såg (å), W5 - något (å), W23 - plötsligt (ö)

    print("\n" + "="*80)
    print("ANALYZING WORDS WITH ö (o with two dots)")
    print("="*80)

    visualize_word_components(img, 17, "W18 - rök (ö)", "/tmp/swedish_o_w18.png")
    visualize_word_components(img, 50, "W51 - för (ö)", "/tmp/swedish_o_w51.png")
    visualize_word_components(img, 22, "W23 - plötsligt (ö)", "/tmp/swedish_o_w23.png")

    print("\n" + "="*80)
    print("ANALYZING WORDS WITH å (a with ring)")
    print("="*80)

    visualize_word_components(img, 15, "W16 - blåste (å)", "/tmp/swedish_a_w16.png")
    visualize_word_components(img, 21, "W22 - såg (å)", "/tmp/swedish_a_w22.png")
    visualize_word_components(img, 4, "W5 - något (å)", "/tmp/swedish_a_w5.png")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nVisualization files created:")
    print("  Words with ö:")
    print("    - /tmp/swedish_o_w18.png (rök)")
    print("    - /tmp/swedish_o_w51.png (för)")
    print("    - /tmp/swedish_o_w23.png (plötsligt)")
    print("  Words with å:")
    print("    - /tmp/swedish_a_w16.png (blåste)")
    print("    - /tmp/swedish_a_w22.png (såg)")
    print("    - /tmp/swedish_a_w5.png (något)")

if __name__ == '__main__':
    main()
