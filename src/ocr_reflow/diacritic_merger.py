#!/usr/bin/env python3
"""
Diacritic detection and merging algorithm based on:
"Detection and Recognition of Diacritical and Punctuation Marks in Real-World Images"
by Jan Hadáček (2014)

This module implements diacritic-letter pairing and merging for Swedish and other languages
with diacritical marks (ö, å, ä, etc.)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def get_diacritic_search_window(letter_bbox, text_height):
    """
    Define a diacritic search window above a letter.

    Based on Section 4.1 of the paper:
    - t = 1.725 * H (distance from baseline to top of search window)
    - b = 0.776 * H (distance from baseline to bottom of search window)
    - w = 0.559 * H (half-width of search window)

    Args:
        letter_bbox: (xmin, ymin, xmax, ymax) of the letter
        text_height: Height H of the text line

    Returns:
        (xmin, ymin, xmax, ymax) of the search window
    """
    xmin, ymin, xmax, ymax = letter_bbox

    # Letter center x
    letter_center_x = (xmin + xmax) / 2

    # Parameters from the paper (adjusted for real-world data)
    t = 1.725 * text_height  # top distance from baseline
    b = 0.776 * text_height  # bottom distance from baseline
    w = 0.559 * text_height  # half-width

    # Baseline is at ymax (bottom of letter)
    baseline = ymax

    # Search window coordinates
    search_ymin = max(0, int(baseline - t))
    search_ymax = int(baseline - b)
    search_xmin = max(0, int(letter_center_x - w))
    search_xmax = int(letter_center_x + w)

    return (search_xmin, search_ymin, search_xmax, search_ymax)


def compute_anchor_points(region_mask):
    """
    Compute anchor points for diacritic and letter regions.

    Based on Rule 4.3.1 from the paper:
    - Diacritic anchor: center of mass of bottom 2/3 of the glyph
    - Letter anchor: center of mass of top 1/5 of the glyph

    Args:
        region_mask: Binary mask of the region

    Returns:
        (cx, cy) center of mass coordinates, or None if empty
    """
    if region_mask.sum() == 0:
        return None

    # Get all non-zero points
    points = np.argwhere(region_mask > 0)

    if len(points) == 0:
        return None

    # Compute center of mass
    cy, cx = points.mean(axis=0)

    return (int(cx), int(cy))


def compute_diacritic_anchor(region_mask):
    """
    Compute diacritic anchor point (center of mass of bottom 2/3).

    Args:
        region_mask: Binary mask of the diacritic region

    Returns:
        (cx, cy) anchor point coordinates
    """
    if region_mask.sum() == 0:
        return None

    h, w = region_mask.shape

    # Bottom 2/3 of scan lines
    bottom_third_start = int(h / 3)
    bottom_region = region_mask[bottom_third_start:, :]

    return compute_anchor_points(bottom_region)


def compute_letter_anchor(region_mask):
    """
    Compute letter anchor point (center of mass of top 1/5).

    Args:
        region_mask: Binary mask of the letter region

    Returns:
        (cx, cy) anchor point coordinates
    """
    if region_mask.sum() == 0:
        return None

    h, w = region_mask.shape

    # Top 1/5 of scan lines
    top_fifth_end = int(h / 5)
    top_region = region_mask[:top_fifth_end, :]

    return compute_anchor_points(top_region)


def compute_pair_features(letter_bbox, letter_mask, diacritic_bbox, diacritic_mask, text_height):
    """
    Compute geometric features for letter-diacritic pair classification.

    Based on Section 4.3 of the paper, features are:
    1. Angle of line segment connecting anchor points (relative to Y axis)
    2. Magnitude of vector between anchor points (relative to line height)
    3. Vertical distance between regions (relative to line height)
    4. Ratio of areas of both regions

    Args:
        letter_bbox: (xmin, ymin, xmax, ymax)
        letter_mask: Binary mask of letter
        diacritic_bbox: (xmin, ymin, xmax, ymax)
        diacritic_mask: Binary mask of diacritic
        text_height: Line height H

    Returns:
        Feature vector [angle, magnitude, vertical_dist, area_ratio] or None
    """
    # Compute anchor points
    letter_anchor = compute_letter_anchor(letter_mask)
    diacritic_anchor = compute_diacritic_anchor(diacritic_mask)

    if letter_anchor is None or diacritic_anchor is None:
        return None

    # Convert to absolute coordinates
    lx, ly = letter_anchor
    lx += letter_bbox[0]
    ly += letter_bbox[1]

    dx, dy = diacritic_anchor
    dx += diacritic_bbox[0]
    dy += diacritic_bbox[1]

    # Feature 1: Angle relative to Y axis (in degrees)
    delta_x = dx - lx
    delta_y = ly - dy  # Note: y increases downward
    angle = np.arctan2(delta_x, delta_y) * 180 / np.pi

    # Feature 2: Magnitude relative to line height
    magnitude = np.sqrt(delta_x**2 + delta_y**2) / text_height

    # Feature 3: Vertical distance (relative to line height)
    vertical_dist = (letter_bbox[1] - diacritic_bbox[3]) / text_height

    # Feature 4: Area ratio
    letter_area = letter_mask.sum()
    diacritic_area = diacritic_mask.sum()
    area_ratio = diacritic_area / (letter_area + 1e-6)

    return np.array([angle, magnitude, vertical_dist, area_ratio])


def is_diacritic_candidate(features, threshold=0.5):
    """
    Simple heuristic classifier to determine if a component is a diacritic.

    Based on expected properties from the paper:
    - Should be roughly centered above letter (small angle)
    - Should be relatively close (small magnitude)
    - Should be above the letter (positive vertical distance)
    - Should be smaller than the letter (area ratio < 1)

    Args:
        features: [angle, magnitude, vertical_dist, area_ratio]
        threshold: Classification threshold

    Returns:
        True if likely a diacritic, False otherwise
    """
    if features is None:
        return False

    angle, magnitude, vertical_dist, area_ratio = features

    # Heuristic rules based on expected diacritic properties
    # 1. Should be roughly centered (angle close to 0)
    if abs(angle) > 30:  # More than 30 degrees off center
        return False

    # 2. Should be relatively close
    if magnitude > 0.5:  # More than half line height away
        return False

    # 3. Should be above the letter
    if vertical_dist < 0:  # Below the letter
        return False

    # 4. Should be smaller than the letter
    if area_ratio > 0.5:  # Too large relative to letter
        return False

    # 5. Should not be too far above
    if vertical_dist > 0.4:  # More than 40% of line height above
        return False

    return True


def find_diacritics_for_letter(letter_bbox, binary_img, text_height):
    """
    Find diacritic candidates above a letter.

    Args:
        letter_bbox: (xmin, ymin, xmax, ymax) of the letter
        binary_img: Binarized image (white text on black background)
        text_height: Height of the text line

    Returns:
        List of (diacritic_bbox, score) tuples
    """

    # Define search window
    search_window = get_diacritic_search_window(letter_bbox, text_height)
    sx1, sy1, sx2, sy2 = search_window

    # Extract search region
    h, w = binary_img.shape[:2]
    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(w, sx2)
    sy2 = min(h, sy2)

    if sx2 <= sx1 or sy2 <= sy1:
        return []

    search_region = binary_img[sy1:sy2, sx1:sx2]

    if len(search_region.shape) == 3:
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    else:
        search_gray = search_region

    # Find connected components in search window
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        search_gray, connectivity=8
    )

    candidates = []

    # Extract letter mask for feature computation
    lx1, ly1, lx2, ly2 = letter_bbox
    lx1 = max(0, lx1)
    ly1 = max(0, ly1)
    lx2 = min(w, lx2)
    ly2 = min(h, ly2)

    if lx2 <= lx1 or ly2 <= ly1:
        return []

    letter_region = binary_img[ly1:ly2, lx1:lx2]
    if len(letter_region.shape) == 3:
        letter_mask = cv2.cvtColor(letter_region, cv2.COLOR_BGR2GRAY)
    else:
        letter_mask = letter_region

    # Check each component (skip background)
    for i in range(1, num_labels):
        x, y, w_comp, h_comp, area = stats[i]

        # Filter out very small components (noise)
        if area < 5:
            continue

        # Filter out components touching the border of search window
        if x <= 0 or y <= 0 or (x + w_comp) >= (sx2 - sx1) or (y + h_comp) >= (sy2 - sy1):
            continue

        # Get diacritic mask
        diacritic_mask = (labels == i).astype(np.uint8) * 255

        # Absolute coordinates
        abs_bbox = (sx1 + x, sy1 + y, sx1 + x + w_comp, sy1 + y + h_comp)

        # Compute features
        features = compute_pair_features(
            letter_bbox, letter_mask,
            abs_bbox, diacritic_mask,
            text_height
        )

        # Classify
        if is_diacritic_candidate(features):
            # Use vertical distance as score (closer is better)
            score = 1.0 / (features[2] + 0.1)  # Avoid division by zero
            candidates.append((abs_bbox, score))

    return candidates


def merge_letter_with_diacritic(letter_bbox, diacritic_bbox, binary_img):
    """
    Merge a letter with its diacritic by expanding the letter bounding box.

    Args:
        letter_bbox: (xmin, ymin, xmax, ymax) of the letter
        diacritic_bbox: (xmin, ymin, xmax, ymax) of the diacritic
        binary_img: Binarized image

    Returns:
        Merged bounding box (xmin, ymin, xmax, ymax)
    """
    lx1, ly1, lx2, ly2 = letter_bbox
    dx1, dy1, dx2, dy2 = diacritic_bbox

    # Union of both bounding boxes
    merged_xmin = min(lx1, dx1)
    merged_ymin = min(ly1, dy1)
    merged_xmax = max(lx2, dx2)
    merged_ymax = max(ly2, dy2)

    return (merged_xmin, merged_ymin, merged_xmax, merged_ymax)


def process_word_diacritics(word_letters, binary_img, text_height):
    """
    Process all letters in a word to find and merge diacritics.

    Args:
        word_letters: List of letter bounding boxes [(xmin, ymin, xmax, ymax), ...]
        binary_img: Binarized image (white text on black background)
        text_height: Height of the text line

    Returns:
        List of merged letter bounding boxes (some may be unchanged)
    """
    merged_letters = []

    for letter_bbox in word_letters:
        # Find diacritic candidates
        candidates = find_diacritics_for_letter(letter_bbox, binary_img, text_height)

        if candidates:
            # Choose the best candidate (highest score)
            best_candidate = max(candidates, key=lambda x: x[1])
            diacritic_bbox, score = best_candidate

            # Merge letter with diacritic
            merged_bbox = merge_letter_with_diacritic(letter_bbox, diacritic_bbox, binary_img)
            merged_letters.append(merged_bbox)
        else:
            # No diacritic found, keep original letter
            merged_letters.append(letter_bbox)

    return merged_letters
