"""
Test script for skew detection and correction
"""
import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ocr_reflow.skew_detection import detect_and_correct_skew, detect_skew

def test_skew_detection(image_path, output_path=None):
    """Test skew detection on an image"""
    print(f"Testing skew detection on: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Detect skew angle only
    angle = detect_skew(img)
    print(f"Detected skew angle: {angle:.2f}°")

    # Correct skew
    corrected_img, detected_angle = detect_and_correct_skew(img)
    print(f"Corrected skew angle: {detected_angle:.2f}°")
    print(f"Corrected image size: {corrected_img.shape[1]}x{corrected_img.shape[0]}")

    # Save corrected image
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_deskewed{ext}"

    cv2.imwrite(output_path, corrected_img)
    print(f"Corrected image saved to: {output_path}")

    # Create side-by-side comparison
    comparison_path = output_path.replace("_deskewed", "_comparison")

    # Resize both images to same height for comparison
    h1, w1 = img.shape[:2]
    h2, w2 = corrected_img.shape[:2]

    target_height = min(h1, h2, 1000)  # Limit to 1000px for display

    scale1 = target_height / h1
    scale2 = target_height / h2

    img_resized = cv2.resize(img, (int(w1 * scale1), target_height))
    corrected_resized = cv2.resize(corrected_img, (int(w2 * scale2), target_height))

    # Make them same width by padding
    max_width = max(img_resized.shape[1], corrected_resized.shape[1])

    if img_resized.shape[1] < max_width:
        pad = max_width - img_resized.shape[1]
        img_resized = cv2.copyMakeBorder(img_resized, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    if corrected_resized.shape[1] < max_width:
        pad = max_width - corrected_resized.shape[1]
        corrected_resized = cv2.copyMakeBorder(corrected_resized, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_resized, f"Original (skew: {angle:.2f}°)", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(corrected_resized, f"Corrected (angle: {detected_angle:.2f}°)", (10, 30), font, 1, (0, 255, 0), 2)

    # Stack vertically
    comparison = np.vstack([img_resized, corrected_resized])
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison image saved to: {comparison_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_skew_detection.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    test_skew_detection(image_path, output_path)
