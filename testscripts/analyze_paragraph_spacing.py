#!/usr/bin/env python3
"""
Analyze line spacing within text paragraphs only (excluding figures, formulas, etc.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_paragraph_spacing(image_path='output_reflowed.png'):
    """Analyze vertical line spacing within paragraphs"""

    print(f"Analyzing paragraph line spacing in: {image_path}")
    print("=" * 70)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find text rows by projecting horizontally
    projection = np.sum(255 - gray, axis=1)

    # Smooth projection
    window = 5
    smoothed = np.convolve(projection, np.ones(window)/window, mode='same')

    # Find peaks
    threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)

    peaks = []
    for i in range(10, len(smoothed) - 10):
        if smoothed[i] > threshold:
            is_peak = True
            for j in range(max(0, i-10), min(len(smoothed), i+10)):
                if j != i and smoothed[j] > smoothed[i]:
                    is_peak = False
                    break
            if is_peak and (len(peaks) == 0 or i - peaks[-1] > 10):
                peaks.append(i)

    peaks = np.array(peaks)
    print(f"Detected {len(peaks)} text lines")

    if len(peaks) < 2:
        print("Not enough lines detected")
        return

    # Calculate all spacings
    all_spacings = []
    for i in range(len(peaks) - 1):
        spacing = peaks[i+1] - peaks[i]
        all_spacings.append(spacing)

    # Filter spacings to identify paragraph breaks vs normal line spacing
    # Paragraph breaks are significantly larger than normal line spacing
    all_spacings = np.array(all_spacings)
    median_spacing = np.median(all_spacings)

    # Classify spacings as normal (within paragraph) or large (between paragraphs)
    # Normal spacing: within 2x median
    # Large spacing: > 2x median (likely paragraph break or figure)
    normal_spacings = all_spacings[all_spacings <= median_spacing * 2]
    large_spacings = all_spacings[all_spacings > median_spacing * 2]

    print(f"\nAll line spacings:")
    print(f"  Total spacings:     {len(all_spacings)}")
    print(f"  Min:                {np.min(all_spacings)} pixels")
    print(f"  Max:                {np.max(all_spacings)} pixels")
    print(f"  Mean:               {np.mean(all_spacings):.1f} pixels")
    print(f"  Median:             {median_spacing:.1f} pixels")
    print(f"  Std deviation:      {np.std(all_spacings):.1f} pixels")

    if len(normal_spacings) > 0:
        print(f"\nNormal line spacings (within paragraphs):")
        print(f"  Count:              {len(normal_spacings)}")
        print(f"  Min:                {np.min(normal_spacings)} pixels")
        print(f"  Max:                {np.max(normal_spacings)} pixels")
        print(f"  Mean:               {np.mean(normal_spacings):.1f} pixels")
        print(f"  Median:             {np.median(normal_spacings):.1f} pixels")
        print(f"  Std deviation:      {np.std(normal_spacings):.1f} pixels")

        # Variation coefficient
        variation = np.std(normal_spacings) / np.mean(normal_spacings)
        print(f"  Variation coeff:    {variation:.2%}")

        if variation < 0.10:
            print("  ✓ EXCELLENT - Very consistent spacing")
        elif variation < 0.15:
            print("  ✓ GOOD - Reasonably consistent spacing")
        elif variation < 0.25:
            print("  ⚠️  MODERATE - Some variation in spacing")
        else:
            print("  ❌ HIGH - Inconsistent spacing")

    if len(large_spacings) > 0:
        print(f"\nLarge spacings (paragraph breaks/figures):")
        print(f"  Count:              {len(large_spacings)}")
        print(f"  Min:                {np.min(large_spacings)} pixels")
        print(f"  Max:                {np.max(large_spacings)} pixels")
        print(f"  Mean:               {np.mean(large_spacings):.1f} pixels")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Original image (crop to show first part)
    crop_h = min(h, 2000)
    axes[0, 0].imshow(cv2.cvtColor(img[:crop_h], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Reflowed Output (top portion)')
    axes[0, 0].axis('off')

    # Projection with peaks
    axes[0, 1].plot(smoothed, range(len(smoothed)), 'b-', linewidth=0.5)
    axes[0, 1].plot(smoothed[peaks], peaks, 'rx', markersize=4)
    axes[0, 1].axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('Y position (pixels)')
    axes[0, 1].set_xlabel('Text density')
    axes[0, 1].set_title(f'Horizontal Projection ({len(peaks)} lines)')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3)

    # Normal spacing histogram
    if len(normal_spacings) > 0:
        axes[1, 0].hist(normal_spacings, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].axvline(x=np.mean(normal_spacings), color='r', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(normal_spacings):.1f}')
        axes[1, 0].axvline(x=np.median(normal_spacings), color='orange', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(normal_spacings):.1f}')
        axes[1, 0].set_xlabel('Line spacing (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Normal Spacing Distribution (within paragraphs)\nStd={np.std(normal_spacings):.1f}, Variation={variation:.1%}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # All spacing histogram
    axes[1, 1].hist(all_spacings, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[1, 1].axvline(x=median_spacing * 2, color='r', linestyle='--',
                      linewidth=2, label=f'2x median (threshold)')
    axes[1, 1].set_xlabel('Line spacing (pixels)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'All Spacing Distribution\n(includes paragraph breaks and figures)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '../notebooks/paragraph_spacing_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved analysis to {output_path}")
    plt.close()

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'output_reflowed.png'
    analyze_paragraph_spacing(image_path)
