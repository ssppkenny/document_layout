#!/usr/bin/env python3
"""
Analyze line spacing in reflowed output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_line_spacing(image_path='output_reflowed.png'):
    """Analyze vertical line spacing in reflowed output"""

    print(f"Analyzing line spacing in: {image_path}")
    print("=" * 70)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        print("Run reflow first: pixi run python src/ocr_reflow/main.py images/sedg_p598.png --layout")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find text rows by projecting horizontally
    # (sum of dark pixels in each row)
    projection = np.sum(255 - gray, axis=1)

    # Smooth projection with simple moving average
    window = 5
    smoothed = np.convolve(projection, np.ones(window)/window, mode='same')

    # Find peaks (text lines) manually
    threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)

    # Find local maxima above threshold
    peaks = []
    for i in range(10, len(smoothed) - 10):
        if smoothed[i] > threshold:
            # Check if local maximum
            is_peak = True
            for j in range(max(0, i-10), min(len(smoothed), i+10)):
                if j != i and smoothed[j] > smoothed[i]:
                    is_peak = False
                    break
            if is_peak and (len(peaks) == 0 or i - peaks[-1] > 10):
                peaks.append(i)

    peaks = np.array(peaks)

    print(f"Detected {len(peaks)} potential text lines")

    # Calculate line spacings
    line_spacings = []
    line_heights = []

    for i in range(len(peaks) - 1):
        spacing = peaks[i+1] - peaks[i]
        line_spacings.append(spacing)

        # Estimate line height (width of peak)
        y = peaks[i]
        # Find where projection drops significantly
        start = y
        while start > 0 and smoothed[start] > threshold * 0.5:
            start -= 1
        end = y
        while end < len(smoothed) - 1 and smoothed[end] > threshold * 0.5:
            end += 1
        line_heights.append(end - start)

    if len(line_spacings) > 0:
        print(f"\nLine spacing statistics:")
        print(f"  Min spacing:    {min(line_spacings)} pixels")
        print(f"  Max spacing:    {max(line_spacings)} pixels")
        print(f"  Mean spacing:   {np.mean(line_spacings):.1f} pixels")
        print(f"  Median spacing: {np.median(line_spacings):.1f} pixels")
        print(f"  Std deviation:  {np.std(line_spacings):.1f} pixels")

        print(f"\nLine height statistics:")
        print(f"  Min height:     {min(line_heights)} pixels")
        print(f"  Max height:     {max(line_heights)} pixels")
        print(f"  Mean height:    {np.mean(line_heights):.1f} pixels")
        print(f"  Median height:  {np.median(line_heights):.1f} pixels")

        # Check for problematic variations
        spacing_variation = np.std(line_spacings) / np.mean(line_spacings)
        print(f"\nSpacing variation coefficient: {spacing_variation:.2%}")

        if spacing_variation > 0.15:
            print("⚠️  HIGH VARIATION - Line spacing is inconsistent!")
        elif spacing_variation > 0.10:
            print("⚠️  MODERATE VARIATION - Some inconsistency in line spacing")
        else:
            print("✓ Low variation - Line spacing is reasonably consistent")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Reflowed Output')
    axes[0, 0].axis('off')

    # Projection profile
    axes[0, 1].plot(smoothed)
    axes[0, 1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    axes[0, 1].plot(peaks, smoothed[peaks], 'rx', markersize=10, label='Detected lines')
    axes[0, 1].set_xlabel('Y position (pixels)')
    axes[0, 1].set_ylabel('Text density')
    axes[0, 1].set_title(f'Horizontal Projection ({len(peaks)} lines detected)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Lines marked on image
    vis_lines = img.copy()
    for i, peak in enumerate(peaks):
        cv2.line(vis_lines, (0, peak), (w, peak), (0, 255, 0), 2)
        cv2.putText(vis_lines, f"L{i}", (10, peak-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    axes[1, 0].imshow(cv2.cvtColor(vis_lines, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Detected Line Centers')
    axes[1, 0].axis('off')

    # Line spacing distribution
    if len(line_spacings) > 0:
        axes[1, 1].hist(line_spacings, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=np.mean(line_spacings), color='r', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(line_spacings):.1f}')
        axes[1, 1].axvline(x=np.median(line_spacings), color='g', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(line_spacings):.1f}')
        axes[1, 1].set_xlabel('Line spacing (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Line Spacing Distribution (std={np.std(line_spacings):.1f})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add text with spacing details
        spacing_text = "Line spacings:\n"
        for i, spacing in enumerate(line_spacings[:min(10, len(line_spacings))]):
            spacing_text += f"L{i}→L{i+1}: {spacing}px\n"
        if len(line_spacings) > 10:
            spacing_text += f"... and {len(line_spacings)-10} more"

        axes[1, 1].text(0.98, 0.98, spacing_text,
                       transform=axes[1, 1].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=8, family='monospace')
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough lines detected',
                       ha='center', va='center')
        axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = '../notebooks/line_spacing_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved analysis to {output_path}")
    plt.close()

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'output_reflowed.png'
    analyze_line_spacing(image_path)
