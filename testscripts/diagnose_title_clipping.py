#!/usr/bin/env python3
"""
Diagnose letter clipping in title blocks - check if descenders (p, g, y, q, j) are cut off
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

def diagnose_title_clipping(reflowed_path='output_reflowed.png', original_path='images/jtg_p033.png'):
    """Check if letters are being clipped in the output"""

    print("=" * 70)
    print("DIAGNOSING LETTER CLIPPING IN TITLE BLOCKS")
    print("=" * 70)

    # Load reflowed output
    reflowed = cv2.imread(reflowed_path)
    if reflowed is None:
        print(f"Error: Could not load {reflowed_path}")
        return

    original = cv2.imread(original_path)
    if original is None:
        print(f"Error: Could not load {original_path}")
        return

    print(f"\nReflowed image: {reflowed.shape[1]}x{reflowed.shape[0]}")
    print(f"Original image: {original.shape[1]}x{original.shape[0]}")

    # Analyze first 500 pixels of reflowed (where title likely is)
    sample = reflowed[:500, :, :]
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    # Find text regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (letters/words)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\nFound {len(contours)} text regions in first 500px")

    # Analyze letter heights and positions
    letter_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter noise
            letter_boxes.append((x, y, w, h))

    letter_boxes.sort(key=lambda b: (b[1], b[0]))  # Sort by y, then x

    print(f"Valid letter boxes: {len(letter_boxes)}")

    if len(letter_boxes) > 0:
        # Analyze first line (title)
        first_line = [b for b in letter_boxes if b[1] < 150]

        print(f"\nFirst line (title) letters: {len(first_line)}")

        if len(first_line) > 0:
            heights = [h for _, _, _, h in first_line]
            bottoms = [y + h for _, y, _, h in first_line]
            tops = [y for _, y, _, h in first_line]

            print(f"\nTitle letter statistics:")
            print(f"  Heights: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
            print(f"  Tops (y): min={min(tops)}, max={max(tops)}")
            print(f"  Bottoms (y+h): min={min(bottoms)}, max={max(bottoms)}")
            print(f"  Y range: {max(bottoms) - min(tops)} pixels")

            # Check if letters are touching top or bottom margins
            if min(tops) < 5:
                print(f"\n⚠️  WARNING: Letters touching top margin (y={min(tops)})")

            if max(bottoms) > 495:
                print(f"⚠️  WARNING: Letters touching bottom of sample (y={max(bottoms)})")

            # Visualize
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Original sample
            axes[0, 0].imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Reflowed Output (first 500px)')
            axes[0, 0].axis('off')

            # With bounding boxes
            vis = sample.copy()
            for x, y, w, h in first_line:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Mark top and bottom
                cv2.line(vis, (x, y), (x+w, y), (255, 0, 0), 1)  # Top
                cv2.line(vis, (x, y+h), (x+w, y+h), (0, 0, 255), 1)  # Bottom

            axes[0, 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Title with Bounding Boxes (Blue=top, Red=bottom)')
            axes[0, 1].axis('off')

            # Height distribution
            axes[1, 0].bar(range(len(first_line)), heights)
            axes[1, 0].axhline(y=np.mean(heights), color='r', linestyle='--', label=f'Mean: {np.mean(heights):.1f}')
            axes[1, 0].set_xlabel('Letter index')
            axes[1, 0].set_ylabel('Height (pixels)')
            axes[1, 0].set_title('Letter Heights in Title')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Vertical positions
            axes[1, 1].plot(range(len(first_line)), tops, 'b-', label='Top (y)', marker='o')
            axes[1, 1].plot(range(len(first_line)), bottoms, 'r-', label='Bottom (y+h)', marker='s')
            axes[1, 1].axhline(y=0, color='g', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('Letter index')
            axes[1, 1].set_ylabel('Y position')
            axes[1, 1].set_title('Letter Vertical Positions')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].invert_yaxis()

            plt.tight_layout()
            plt.savefig('notebooks/title_clipping_diagnosis.png', dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved diagnosis to notebooks/title_clipping_diagnosis.png")
            plt.close()

            # Check if there's clipping
            max_height_var = max(heights) - min(heights)
            if max_height_var > np.mean(heights) * 0.3:
                print(f"\n⚠️  HIGH HEIGHT VARIATION: {max_height_var} pixels")
                print("   This suggests some letters may be clipped (descenders cut off)")

            top_var = max(tops) - min(tops)
            if top_var > 20:
                print(f"\n⚠️  INCONSISTENT TOPS: {top_var} pixels variation")
                print("   Letters not aligned properly at top")

            bottom_var = max(bottoms) - min(bottoms)
            if bottom_var > 20:
                print(f"\n⚠️  INCONSISTENT BOTTOMS: {bottom_var} pixels variation")
                print("   This is NORMAL - descenders (g, p, y) extend below baseline")
                print("   But if too small, descenders might be clipped!")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    diagnose_title_clipping()
