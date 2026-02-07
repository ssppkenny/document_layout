"""
Visualize how intersection masking works on sedg_p598.png
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ocr_reflow.layout import layout

def visualize_masking(image_path):
    """Visualize the masking process"""
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect background color
    flat_img = img.reshape(-1, 3)
    background_color = np.median(flat_img, axis=0).astype(np.uint8)

    # Get layout
    layout_boxes = layout(image_path)
    layout_boxes_sorted = sorted(layout_boxes, key=lambda item: (item[0].bounds[1], item[0].bounds[0]))

    # Identify text and non-text boxes
    text_boxes = [(g, t) for g, t in layout_boxes_sorted if t in ["plain text", "title"]]
    non_text_boxes = [(g, t) for g, t in layout_boxes_sorted if t not in ["plain text", "title"]]

    # Colors for different box types
    colors = {
        "plain text": "blue",
        "title": "purple",
        "figure_and_caption": "red",
        "table_and_caption": "green",
        "formula": "orange",
        "abandon": "gray"
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Subplot 1: Original layout
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Layout', fontsize=14, weight='bold')
    axes[0].axis('off')

    for geom, box_type in layout_boxes_sorted:
        bounds = geom.bounds
        x, y, x2, y2 = bounds
        rect = patches.Rectangle((x, y), x2-x, y2-y,
                                linewidth=2, edgecolor=colors.get(box_type, 'white'),
                                facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x, y-5, box_type, color=colors.get(box_type, 'white'),
                   fontsize=9, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Subplot 2: Highlight intersecting text box
    axes[1].imshow(img_rgb)
    axes[1].set_title('Text Box #2 with Intersections', fontsize=14, weight='bold')
    axes[1].axis('off')

    # Draw text box 2
    box_geom, box_type = text_boxes[1]
    bounds = box_geom.bounds
    xmin, ymin, xmax, ymax = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                            linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.2)
    axes[1].add_patch(rect)
    axes[1].text(xmin, ymin-10, 'Text Box #2', color='blue',
               fontsize=12, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Highlight intersecting non-text regions
    for other_geom, other_type in non_text_boxes:
        if box_geom.intersects(other_geom):
            intersection = box_geom.intersection(other_geom)
            inter_bounds = intersection.bounds
            ix, iy, ix2, iy2 = int(inter_bounds[0]), int(inter_bounds[1]), int(inter_bounds[2]), int(inter_bounds[3])
            rect = patches.Rectangle((ix, iy), ix2-ix, iy2-iy,
                                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.5)
            axes[1].add_patch(rect)
            axes[1].text(ix, iy-5, f'{other_type}\n(MASKED)', color='red',
                       fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.9))

    # Subplot 3: Show masked text box
    axes[2].set_title('Text Box #2 After Masking', fontsize=14, weight='bold')
    axes[2].axis('off')

    # Extract and mask the text box
    box_img = img[ymin:ymax, xmin:xmax].copy()
    for other_geom, other_type in non_text_boxes:
        if box_geom.intersects(other_geom):
            intersection = box_geom.intersection(other_geom)
            inter_bounds = intersection.bounds

            local_xmin = max(0, int(inter_bounds[0] - xmin))
            local_ymin = max(0, int(inter_bounds[1] - ymin))
            local_xmax = min(box_img.shape[1], int(inter_bounds[2] - xmin))
            local_ymax = min(box_img.shape[0], int(inter_bounds[3] - ymin))

            if local_xmax > local_xmin and local_ymax > local_ymin:
                box_img[local_ymin:local_ymax, local_xmin:local_xmax] = background_color

    box_img_rgb = cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB)
    axes[2].imshow(box_img_rgb)

    plt.tight_layout()
    plt.savefig('intersection_masking_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: intersection_masking_visualization.png")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_masking.py <image_path>")
        sys.exit(1)

    visualize_masking(sys.argv[1])
