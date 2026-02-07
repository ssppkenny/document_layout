"""
Skew Detection and Correction Module

Implementation of MCCSD (Modified Cross-Correlation Skew Detection) algorithm
based on the paper: "A Robust Skew Detection Algorithm for Grayscale Document Image"
by Ming Chen and Xiaoqing Ding, Tsinghua University.

The algorithm uses horizontal and vertical cross-correlation to detect skew in
document images, handles both horizontal and vertical text layouts, and uses
randomly selected regions for efficiency.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional
import random

logger = logging.getLogger(__name__)


def calculate_vertical_cross_correlation(image: np.ndarray, d: int, s_range: int) -> np.ndarray:
    """
    Calculate vertical cross-correlation (VCC) for the entire image.

    VCC measures correlation between vertical lines separated by distance d,
    shifted by various s values.

    Args:
        image: Grayscale image (H x W)
        d: Distance between two vertical lines
        s_range: Range of shift values to test [-s_range, +s_range]

    Returns:
        R(s): Cross-correlation values for each shift s
    """
    height, width = image.shape
    R = np.zeros(2 * s_range + 1, dtype=np.float64)

    # For each shift value s
    for s_idx, s in enumerate(range(-s_range, s_range + 1)):
        correlation = 0.0
        count = 0

        # Sum over all pairs of vertical lines with distance d
        for x0 in range(width - d):
            # Calculate valid y range considering shift s
            if s >= 0:
                y_start = 0
                y_end = height - s
            else:
                y_start = -s
                y_end = height

            if y_end > y_start:
                # Calculate correlation between line at x0 and line at x0+d with shift s
                line1 = image[y_start:y_end, x0].astype(np.float64)
                line2 = image[y_start + s:y_end + s, x0 + d].astype(np.float64)

                # Normalize: subtract mean and divide by std to make correlation scale-independent
                mean1, std1 = line1.mean(), line1.std()
                mean2, std2 = line2.mean(), line2.std()

                if std1 > 1e-10 and std2 > 1e-10:  # Avoid division by zero
                    line1_norm = (line1 - mean1) / std1
                    line2_norm = (line2 - mean2) / std2
                    correlation += np.sum(line1_norm * line2_norm)
                    count += 1

        R[s_idx] = correlation / max(count, 1)

    return R


def calculate_horizontal_cross_correlation(image: np.ndarray, d: int, s_range: int) -> np.ndarray:
    """
    Calculate horizontal cross-correlation (HCC) for the entire image.

    HCC measures correlation between horizontal lines separated by distance d,
    shifted by various s values.

    Args:
        image: Grayscale image (H x W)
        d: Distance between two horizontal lines
        s_range: Range of shift values to test [-s_range, +s_range]

    Returns:
        R(s): Cross-correlation values for each shift s
    """
    height, width = image.shape
    R = np.zeros(2 * s_range + 1, dtype=np.float64)

    # For each shift value s
    for s_idx, s in enumerate(range(-s_range, s_range + 1)):
        correlation = 0.0
        count = 0

        # Sum over all pairs of horizontal lines with distance d
        for y0 in range(height - d):
            # Calculate valid x range considering shift s
            if s >= 0:
                x_start = 0
                x_end = width - s
            else:
                x_start = -s
                x_end = width

            if x_end > x_start:
                # Calculate correlation between line at y0 and line at y0+d with shift s
                line1 = image[y0, x_start:x_end].astype(np.float64)
                line2 = image[y0 + d, x_start + s:x_end + s].astype(np.float64)

                # Normalize: subtract mean and divide by std to make correlation scale-independent
                mean1, std1 = line1.mean(), line1.std()
                mean2, std2 = line2.mean(), line2.std()

                if std1 > 1e-10 and std2 > 1e-10:  # Avoid division by zero
                    line1_norm = (line1 - mean1) / std1
                    line2_norm = (line2 - mean2) / std2
                    correlation += np.sum(line1_norm * line2_norm)
                    count += 1

        R[s_idx] = correlation / max(count, 1)

    return R


def calculate_total_variation(R: np.ndarray) -> float:
    """
    Calculate total variation of cross-correlation function.

    Total variation is the sum of absolute differences between consecutive values.
    This helps determine whether the correlation function has distinct peaks.

    Args:
        R: Cross-correlation values

    Returns:
        Total variation ΔR(±S)
    """
    return np.sum(np.abs(np.diff(R)))


def find_peaks(R: np.ndarray, s_range: int, min_prominence: float = 0.1) -> list:
    """
    Find peaks in cross-correlation function.

    Args:
        R: Cross-correlation values
        s_range: Range of shift values
        min_prominence: Minimum prominence for peak detection (relative to max value)

    Returns:
        List of (s_value, peak_value) tuples
    """
    peaks = []

    # Find local maxima - any point higher than its neighbors
    for i in range(1, len(R) - 1):
        if R[i] > R[i-1] and R[i] > R[i+1]:
            s_value = i - s_range  # Convert index to actual s value
            peaks.append((s_value, R[i]))

    # If no peaks found, the function is too flat - find the global maximum
    if not peaks:
        max_idx = np.argmax(R)
        s_value = max_idx - s_range
        peaks.append((s_value, R[max_idx]))

    # Sort by peak value (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)

    return peaks


def detect_skew_in_region(region: np.ndarray, d: int, s_range: int, d_prime: int = None) -> Optional[float]:
    """
    Detect skew angle in a single region.

    Args:
        region: Image region (grayscale)
        d: Distance between lines for first correlation
        s_range: Range of shift values
        d_prime: Distance for auxiliary peak resolution (optional)

    Returns:
        Detected skew angle in degrees, or None if region is unsuitable
    """
    # Calculate both HCC and VCC
    R_V = calculate_vertical_cross_correlation(region, d, s_range)
    R_H = calculate_horizontal_cross_correlation(region, d, s_range)

    # Calculate total variations
    delta_V = calculate_total_variation(R_V)
    delta_H = calculate_total_variation(R_H)

    # Check if region has distinct peaks (suitable for skew detection)
    # Since we're using normalized correlation, the variation scale is different
    # A minimum variation of 10 is reasonable for normalized values
    min_variation_threshold = 10.0

    if delta_V < min_variation_threshold and delta_H < min_variation_threshold:
        logger.debug(f"Region unsuitable: delta_V={delta_V:.2f}, delta_H={delta_H:.2f}")
        return None

    # Choose the correlation function with larger variation
    if delta_V > delta_H:
        # Horizontal text layout (use VCC)
        R_selected = R_V
        is_horizontal = True
    else:
        # Vertical text layout (use HCC)
        R_selected = R_H
        is_horizontal = False

    # Find peaks in selected correlation function
    peaks = find_peaks(R_selected, s_range)

    if not peaks:
        logger.debug("No peaks found in correlation function")
        return None

    # Get the primary peak
    s_p, _ = peaks[0]

    # Calculate skew angle
    if s_p == 0:
        angle = 0.0
    else:
        angle = np.arctan(s_p / d) * 180.0 / np.pi

    # If there are multiple peaks (auxiliary peaks), resolve using different d
    if len(peaks) > 1 and d_prime is not None and d_prime != d:
        # Recalculate with different d
        if is_horizontal:
            R_prime = calculate_vertical_cross_correlation(region, d_prime, s_range)
        else:
            R_prime = calculate_horizontal_cross_correlation(region, d_prime, s_range)

        peaks_prime = find_peaks(R_prime, s_range)

        if peaks_prime:
            # Find matching angle
            s_p_prime, _ = peaks_prime[0]
            angle_prime = np.arctan(s_p_prime / d_prime) * 180.0 / np.pi

            # If angles are close, use the average
            if abs(angle - angle_prime) < 1.0:
                angle = (angle + angle_prime) / 2.0

    logger.debug(f"Region angle: {angle:.2f}° (horizontal layout: {is_horizontal})")
    return angle


def detect_skew(image: np.ndarray,
                d: int = 75,
                s_range: int = 25,
                d_prime: int = 50,
                region_size: int = 150,
                num_regions: int = 9,
                max_attempts: int = 50) -> float:
    """
    Detect skew angle in document image using MCCSD algorithm.

    Args:
        image: Input image (grayscale or color, will be converted to grayscale)
        d: Distance between lines for correlation (default: 75)
        s_range: Range of shift values (default: 25, gives ±18.43° range)
        d_prime: Alternative distance for auxiliary peak resolution (default: 50)
        region_size: Size of randomly selected regions (default: 150x150)
        num_regions: Number of regions to analyze (default: 9)
        max_attempts: Maximum attempts to find suitable regions (default: 50)

    Returns:
        Detected skew angle in degrees
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape

    # Check if image is large enough
    if height < region_size or width < region_size:
        logger.warning(f"Image too small for region-based detection, using entire image")
        angle = detect_skew_in_region(gray, d, s_range, d_prime)
        return angle if angle is not None else 0.0

    # Collect angles from multiple regions
    detected_angles = []
    attempts = 0

    while len(detected_angles) < num_regions and attempts < max_attempts:
        attempts += 1

        # Randomly select a region
        x = random.randint(0, width - region_size)
        y = random.randint(0, height - region_size)
        region = gray[y:y+region_size, x:x+region_size]

        # Detect skew in this region
        angle = detect_skew_in_region(region, d, s_range, d_prime)

        if angle is not None:
            detected_angles.append(angle)
            logger.debug(f"Region {len(detected_angles)}/{num_regions}: angle = {angle:.2f}°")

    if not detected_angles:
        logger.info("No valid regions found, trying full image detection")
        full_image_angle = detect_skew_in_region(gray, d, s_range, d_prime)
        if full_image_angle is not None:
            logger.info(f"Detected skew angle (full image): {full_image_angle:.2f}°")
            print(f"Detected skew angle: {full_image_angle:.2f}° (from full image)")
            return full_image_angle
        else:
            logger.warning("No suitable regions found for skew detection")
            return 0.0

    # If we have very few valid regions, also try full image and compare
    if len(detected_angles) < max(3, num_regions // 2):
        logger.info(f"Only {len(detected_angles)} valid regions, also checking full image")
        full_image_angle = detect_skew_in_region(gray, d, s_range, d_prime)
        if full_image_angle is not None:
            # Add full image result to the mix
            detected_angles.append(full_image_angle)
            detected_angles.append(full_image_angle)  # Weight it more
            logger.info(f"Full image detection: {full_image_angle:.2f}°")

    # Vote: use median of detected angles
    final_angle = np.median(detected_angles)

    logger.info(f"Detected skew angle: {final_angle:.2f}° (from {len(detected_angles)} measurements)")
    print(f"Detected skew angle: {final_angle:.2f}° (from {len(detected_angles)} measurements)")

    return final_angle


def rotate_image(image: np.ndarray, angle: float, background_color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Rotate image to correct skew.

    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
        background_color: Background color for empty areas (default: white)

    Returns:
        Rotated image
    """
    if abs(angle) < 0.1:
        logger.info("Skew angle too small, no rotation needed")
        return image

    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image size to fit entire rotated image
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Rotate image
    if len(image.shape) == 3:
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 borderValue=background_color)
    else:
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 borderValue=background_color[0])

    logger.info(f"Image rotated by {angle:.2f}° (original: {width}x{height}, new: {new_width}x{new_height})")

    return rotated


def detect_and_correct_skew(image: np.ndarray,
                           d: int = 75,
                           s_range: int = 25,
                           **kwargs) -> Tuple[np.ndarray, float]:
    """
    Detect and correct skew in document image.

    Args:
        image: Input image (grayscale or color)
        d: Distance between lines for correlation (default: 75)
        s_range: Range of shift values (default: 25)
        **kwargs: Additional arguments for detect_skew()

    Returns:
        Tuple of (corrected_image, detected_angle)
    """
    # Detect skew angle
    angle = detect_skew(image, d=d, s_range=s_range, **kwargs)

    # Correct skew
    if abs(angle) > 0.1:
        corrected = rotate_image(image, angle)
        return corrected, angle
    else:
        return image, angle
