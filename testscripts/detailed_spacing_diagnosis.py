#!/usr/bin/env python3
"""
Detailed diagnosis: show line spacings for each paragraph
"""

import cv2
import numpy as np

def detailed_spacing_diagnosis(image_path='output_reflowed.png'):
    """Show line spacing statistics per paragraph"""

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Project horizontally
    projection = np.sum(255 - gray, axis=1)
    window = 5
    smoothed = np.convolve(projection, np.ones(window)/window, mode='same')

    threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)

    # Find peaks
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

    print("Detailed Line Spacing Analysis")
    print("=" * 70)
    print(f"Found {len(peaks)} lines\n")

    # Calculate spacings
    spacings = []
    for i in range(len(peaks) - 1):
        spacing = peaks[i+1] - peaks[i]
        spacings.append((i, i+1, spacing))

    # Group by spacing similarity to identify paragraphs
    median = np.median([s[2] for s in spacings])

    print(f"Line-to-line spacings:")
    print(f"{'Line':<8} {'Next':<8} {'Spacing':<10} {'Note'}")
    print("-" * 70)

    for i, next_i, spacing in spacings[:50]:  # Show first 50
        note = ""
        if spacing > median * 2:
            note = "← PARAGRAPH BREAK / FIGURE"
        elif spacing < median * 0.5:
            note = "← Very tight"
        elif spacing > median * 1.5:
            note = "← Wider"

        print(f"{i:<8} {next_i:<8} {spacing:<10} {note}")

    if len(spacings) > 50:
        print(f"... and {len(spacings) - 50} more spacings")

    # Find groups of similar spacings (paragraphs)
    normal_spacings = [s[2] for s in spacings if s[2] <= median * 2]

    if len(normal_spacings) > 0:
        print(f"\n" + "=" * 70)
        print("Normal spacing statistics (within paragraphs):")
        print(f"  Count: {len(normal_spacings)}")
        print(f"  Min: {min(normal_spacings)}")
        print(f"  Max: {max(normal_spacings)}")
        print(f"  Range: {max(normal_spacings) - min(normal_spacings)}")
        print(f"  Mean: {np.mean(normal_spacings):.1f}")
        print(f"  Std: {np.std(normal_spacings):.1f}")
        print(f"  Variation: {np.std(normal_spacings) / np.mean(normal_spacings):.1%}")

        # Show distribution
        unique_spacings = {}
        for s in normal_spacings:
            if s not in unique_spacings:
                unique_spacings[s] = 0
            unique_spacings[s] += 1

        print(f"\nSpacing distribution (normal lines):")
        for spacing in sorted(unique_spacings.keys())[:20]:
            count = unique_spacings[spacing]
            bar = "█" * (count // 2)
            print(f"  {spacing:3d}px: {bar} ({count})")

    print("=" * 70)

if __name__ == '__main__':
    detailed_spacing_diagnosis()
