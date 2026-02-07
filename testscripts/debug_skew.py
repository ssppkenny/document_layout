"""
Debug skew detection to see what's happening
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ocr_reflow.skew_detection import (
    calculate_vertical_cross_correlation,
    calculate_horizontal_cross_correlation,
    calculate_total_variation,
    find_peaks,
    detect_skew_in_region
)

def debug_skew_detection(image_path):
    """Debug skew detection on an image"""
    print(f"Debugging skew detection on: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    print(f"Image size: {gray.shape[1]}x{gray.shape[0]}")

    # Test on full image first
    d = 75
    s_range = 25

    print(f"\n=== Testing on full image ===")
    print(f"Parameters: d={d}, s_range={s_range}")

    # Calculate correlations
    print("Calculating VCC (Vertical Cross-Correlation)...")
    R_V = calculate_vertical_cross_correlation(gray, d, s_range)
    delta_V = calculate_total_variation(R_V)
    print(f"VCC total variation: {delta_V:.2e}")

    print("Calculating HCC (Horizontal Cross-Correlation)...")
    R_H = calculate_horizontal_cross_correlation(gray, d, s_range)
    delta_H = calculate_total_variation(R_H)
    print(f"HCC total variation: {delta_H:.2e}")

    # Determine which to use
    if delta_V > delta_H:
        print(f"\nUsing VCC (horizontal text layout)")
        R_selected = R_V
    else:
        print(f"\nUsing HCC (vertical text layout)")
        R_selected = R_H

    # Find peaks
    peaks = find_peaks(R_selected, s_range, min_prominence=0.1)
    print(f"\nFound {len(peaks)} peaks:")
    for i, (s_value, peak_value) in enumerate(peaks[:5]):  # Show top 5
        angle = np.arctan(s_value / d) * 180.0 / np.pi
        print(f"  Peak {i+1}: s={s_value}, angle={angle:.2f}°, value={peak_value:.2e}")

    # Show correlation values around zero
    print(f"\nCorrelation values around s=0:")
    for s in range(-5, 6):
        idx = s + s_range
        if 0 <= idx < len(R_selected):
            print(f"  s={s:3d}: {R_selected[idx]:.2e}")

    # Test on a sample region
    print(f"\n=== Testing on sample region ===")
    height, width = gray.shape
    region_size = 150

    if height >= region_size and width >= region_size:
        # Try center region
        y = (height - region_size) // 2
        x = (width - region_size) // 2
        region = gray[y:y+region_size, x:x+region_size]

        print(f"Region at ({x}, {y}), size {region_size}x{region_size}")
        angle = detect_skew_in_region(region, d, s_range, d_prime=50)
        print(f"Detected angle in region: {angle}°")

    # Try detecting with different parameters
    print(f"\n=== Testing with different parameters ===")
    for d_test in [50, 75, 100]:
        for s_test in [15, 25, 35]:
            R_V_test = calculate_vertical_cross_correlation(gray, d_test, s_test)
            peaks_test = find_peaks(R_V_test, s_test, min_prominence=0.1)
            if peaks_test:
                s_p, _ = peaks_test[0]
                angle_test = np.arctan(s_p / d_test) * 180.0 / np.pi
                print(f"  d={d_test:3d}, s_range={s_test:2d}: angle={angle_test:6.2f}° (s_p={s_p:3d})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_skew.py <image_path>")
        sys.exit(1)

    debug_skew_detection(sys.argv[1])
