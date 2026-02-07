"""
Test correlation calculation on a simple case
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_correlation():
    """Test cross-correlation on a synthetic image"""
    # Create a synthetic skewed text image
    img = np.ones((500, 800), dtype=np.uint8) * 255

    # Draw horizontal lines with a skew
    skew_angle = 2.0  # degrees
    skew_rad = np.deg2rad(skew_angle)

    for i in range(10):
        y_left = 50 + i * 40
        y_right = y_left + int(800 * np.tan(skew_rad))

        # Draw a slanted line
        pts = []
        for x in range(0, 800, 10):
            y = int(y_left + x * np.tan(skew_rad))
            pts.append((x, y))

        for j in range(len(pts)-1):
            cv2.line(img, pts[j], pts[j+1], 0, 3)

    cv2.imwrite('/tmp/test_skewed_lines.png', img)
    print("Created synthetic skewed image")

    # Now test correlation
    from ocr_reflow.skew_detection import calculate_vertical_cross_correlation, find_peaks

    d = 75
    s_range = 25

    R_V = calculate_vertical_cross_correlation(img, d, s_range)

    print(f"\nCorrelation values:")
    print(f"Min: {R_V.min():.2e}, Max: {R_V.max():.2e}")
    print(f"Std: {R_V.std():.2e}")
    print(f"\nValues around zero:")
    for s in range(-5, 6):
        idx = s + s_range
        print(f"  s={s:3d}: {R_V[idx]:.2e}")

    peaks = find_peaks(R_V, s_range)
    print(f"\nPeaks found: {len(peaks)}")
    for i, (s_val, peak_val) in enumerate(peaks[:5]):
        angle = np.arctan(s_val / d) * 180.0 / np.pi
        print(f"  Peak {i+1}: s={s_val}, angle={angle:.2f}°")

if __name__ == "__main__":
    test_correlation()
