"""
Table of Contents Detection and Analysis Module

Based on the paper: "Detection and Analysis of Table of Contents Based on Content Association"
by Xiaofan Lin and Yan Xiong (HPL-2005-105)

This module implements:
1. TOC page detection using page number patterns
2. Article reference extraction
3. Page number extraction and alignment detection
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageNumber:
    """Represents a detected page number on a TOC page"""
    text: str
    value: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    line_index: int


@dataclass
class TOCEntry:
    """Represents a single entry (article reference) in the table of contents"""
    page_number: Optional[PageNumber]
    text_words: List[Tuple[int, int, int, int]]  # List of word bounding boxes
    ymin: int
    ymax: int


def is_numeric_string(text: str) -> bool:
    """Check if a string is a numeric string (possibly with roman numerals)"""
    # Remove common separators and whitespace
    cleaned = text.replace('-', '').replace('.', '').replace(',', '').replace(' ', '').strip()

    if not cleaned:
        return False

    # Check for arabic numerals
    if cleaned.isdigit():
        return True

    # Check for roman numerals (case-insensitive)
    # Roman numerals use: I(1), V(5), X(10), L(50), C(100), D(500), M(1000)
    roman_pattern = r'^[ivxlcdmIVXLCDM]+$'
    if re.match(roman_pattern, cleaned):
        # Additional validation: must contain at least one roman numeral character
        # and follow basic roman numeral rules
        roman_chars = set('ivxlcdmIVXLCDM')
        if all(c in roman_chars for c in cleaned):
            return True

    return False


def extract_page_number_candidates(words: List[Tuple[int, int, int, int]],
                                   texts: List[str],
                                   page_width: int,
                                   page_height: int) -> List[PageNumber]:
    """
    Extract potential page numbers from a page.

    Page numbers are typically:
    - Numerical strings (Arabic or Roman)
    - Located at the beginning or end of lines
    - Often right-aligned in TOCs

    Args:
        words: List of word bounding boxes (xmin, ymin, xmax, ymax)
        texts: List of corresponding text strings
        page_width: Width of the page
        page_height: Height of the page

    Returns:
        List of PageNumber objects
    """
    candidates = []

    if not words or not texts:
        return candidates

    # First, group words into lines based on vertical position
    line_threshold = 20  # pixels
    lines = []
    current_line = []
    current_y = words[0][1] if words else 0

    for i, (word, text) in enumerate(zip(words, texts)):
        xmin, ymin, xmax, ymax = word

        if abs(ymin - current_y) > line_threshold and current_line:
            lines.append(current_line)
            current_line = []
            current_y = ymin

        current_line.append({'index': i, 'word': word, 'text': text})

    if current_line:
        lines.append(current_line)

    # Now check the last word of each line for page numbers
    # TOC entries typically have the page number as the last element
    for line in lines:
        if not line:
            continue

        # Check the last word in the line
        last_word_data = line[-1]
        text = last_word_data['text']
        word = last_word_data['word']
        xmin, ymin, xmax, ymax = word

        # Check if it's a numeric string
        if not is_numeric_string(text):
            continue

        # Extract numeric value
        # For Roman numerals, use a conversion function
        try:
            if text.replace('-', '').replace('.', '').strip().isdigit():
                # Arabic numeral
                numeric_text = ''.join(c for c in text if c.isdigit())
                value = int(numeric_text) if numeric_text else 0
            else:
                # Roman numeral - convert to numeric value
                value = roman_to_int(text.replace('-', '').replace('.', '').strip())
        except (ValueError, AttributeError):
            continue

        if value > 0:
            candidates.append(PageNumber(
                text=text,
                value=value,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                line_index=-1  # Will be assigned later
            ))

    return candidates


def roman_to_int(s: str) -> int:
    """Convert Roman numeral string to integer"""
    if not s:
        return 0

    s = s.upper()
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }

    total = 0
    prev_value = 0

    for char in reversed(s):
        if char not in roman_values:
            return 0  # Invalid roman numeral

        value = roman_values[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total


def assign_line_indices(page_numbers: List[PageNumber]) -> List[PageNumber]:
    """
    Assign line indices to page numbers based on vertical position.
    Page numbers on the same horizontal line get the same line index.
    """
    if not page_numbers:
        return page_numbers

    # Sort by vertical position
    sorted_pns = sorted(page_numbers, key=lambda pn: pn.ymin)

    line_index = 0
    current_line_y = sorted_pns[0].ymin
    line_threshold = 20  # Pixels - words within this range are on the same line

    for pn in sorted_pns:
        if abs(pn.ymin - current_line_y) > line_threshold:
            line_index += 1
            current_line_y = pn.ymin
        pn.line_index = line_index

    return page_numbers


def check_incremental_pattern(page_numbers: List[PageNumber]) -> Tuple[bool, float]:
    """
    Check if page numbers follow an incremental pattern.

    For TOC pages, numbers don't have to be strictly sequential, but they should
    generally be increasing (allowing for some exceptions like chapter markers).

    Returns:
        Tuple of (is_incremental, score)
        - is_incremental: True if the pattern is mostly incremental
        - score: Confidence score (0.0 to 1.0)
    """
    if len(page_numbers) < 2:
        return False, 0.0

    # Sort by vertical position (line_index), then by horizontal position
    sorted_pns = sorted(page_numbers, key=lambda pn: (pn.line_index, pn.ymin))

    # Count incremental pairs
    incremental_count = 0
    non_decreasing_count = 0  # Includes equal (for repeated page numbers)
    total_pairs = len(sorted_pns) - 1

    for i in range(len(sorted_pns) - 1):
        curr_val = sorted_pns[i].value
        next_val = sorted_pns[i+1].value

        if next_val > curr_val:
            incremental_count += 1
            non_decreasing_count += 1
        elif next_val == curr_val:
            # Allow equal values (e.g., same page number for multiple entries)
            non_decreasing_count += 1

    # Use non-decreasing pattern (more forgiving)
    score = non_decreasing_count / total_pairs if total_pairs > 0 else 0.0

    # For TOCs, we expect at least 60% non-decreasing pattern (relaxed threshold)
    is_incremental = score >= 0.6

    logger.debug(f"Incremental pattern: {incremental_count}/{total_pairs} strictly increasing, "
                f"{non_decreasing_count}/{total_pairs} non-decreasing, score={score:.2f}")

    return is_incremental, score


def check_vertical_alignment(page_numbers: List[PageNumber],
                             alignment_threshold: int = 50) -> Tuple[bool, float]:
    """
    Check if page numbers are vertically aligned.

    For TOC pages, we check alignment of the RIGHT edge (xmax) of page numbers
    since they are typically right-aligned.

    Args:
        page_numbers: List of page numbers to check
        alignment_threshold: Maximum horizontal deviation allowed (in pixels)

    Returns:
        Tuple of (is_aligned, score)
    """
    if len(page_numbers) < 2:
        return False, 0.0

    # For TOC, check right-edge alignment (xmax) since numbers are right-aligned
    x_positions_right = [pn.xmax for pn in page_numbers]
    median_x_right = np.median(x_positions_right)

    # Also check left-edge alignment (xmin) for comparison
    x_positions_left = [pn.xmin for pn in page_numbers]
    median_x_left = np.median(x_positions_left)

    # Count how many are aligned on the right edge
    aligned_count_right = sum(1 for x in x_positions_right if abs(x - median_x_right) <= alignment_threshold)
    score_right = aligned_count_right / len(page_numbers)

    # Count how many are aligned on the left edge
    aligned_count_left = sum(1 for x in x_positions_left if abs(x - median_x_left) <= alignment_threshold)
    score_left = aligned_count_left / len(page_numbers)

    # Use the better alignment score (TOCs typically have right-aligned numbers)
    score = max(score_right, score_left)
    is_aligned = score >= 0.65  # At least 65% should be aligned (relaxed from 70%)

    logger.debug(f"Alignment: right_edge_score={score_right:.2f}, left_edge_score={score_left:.2f}, using={score:.2f}")

    return is_aligned, score


def detect_toc_page(words: List[Tuple[int, int, int, int]],
                   texts: List[str],
                   page_width: int,
                   page_height: int,
                   min_page_numbers: int = 3) -> Tuple[bool, float, List[PageNumber]]:
    """
    Detect if a page is a table of contents page.

    A page is likely a TOC if it contains:
    1. Multiple page numbers (numerical strings)
    2. Page numbers are vertically aligned
    3. Page numbers follow an incremental pattern

    Args:
        words: List of word bounding boxes
        texts: List of corresponding text strings
        page_width: Width of the page
        page_height: Height of the page
        min_page_numbers: Minimum number of page numbers required

    Returns:
        Tuple of (is_toc, confidence_score, page_numbers)
    """
    # Extract page number candidates
    candidates = extract_page_number_candidates(words, texts, page_width, page_height)

    if len(candidates) < min_page_numbers:
        return False, 0.0, []

    # Assign line indices
    candidates = assign_line_indices(candidates)

    # Check for incremental pattern
    is_incremental, inc_score = check_incremental_pattern(candidates)

    # Check for vertical alignment
    is_aligned, align_score = check_vertical_alignment(candidates)

    # Calculate overall confidence score
    quantity_score = min(len(candidates) / 15.0, 1.0)  # More page numbers = higher score (adjusted for 18 lines)

    # Weighted combination of scores
    # Alignment is most important for TOC, then incremental pattern, then quantity
    confidence_score = (
        0.45 * align_score +    # Alignment is most important for TOC
        0.35 * inc_score +      # Incremental pattern is important
        0.20 * quantity_score   # Quantity matters but less critical
    )

    # Lower threshold to 0.5 to catch more TOC pages
    is_toc = confidence_score >= 0.5 and len(candidates) >= min_page_numbers

    if is_toc:
        logger.info(f"TOC detected: {len(candidates)} page numbers, "
                   f"incremental={inc_score:.2f}, aligned={align_score:.2f}, "
                   f"confidence={confidence_score:.2f}")

    return is_toc, confidence_score, candidates


def extract_toc_entries(words: List[Tuple[int, int, int, int]],
                       texts: List[str],
                       page_numbers: List[PageNumber],
                       page_width: int) -> List[TOCEntry]:
    """
    Extract individual TOC entries (article references) from a TOC page.

    Each entry consists of:
    - Text (title, author, etc.)
    - Page number

    Args:
        words: List of word bounding boxes
        texts: List of corresponding text strings
        page_numbers: Detected page numbers
        page_width: Width of the page

    Returns:
        List of TOCEntry objects
    """
    if not page_numbers:
        return []

    # Sort page numbers by vertical position
    sorted_pns = sorted(page_numbers, key=lambda pn: pn.ymin)

    entries = []
    line_threshold = 20  # Pixels - vertical threshold for same line

    # Create a set of page number positions for quick lookup
    pn_positions = {(pn.xmin, pn.ymin, pn.xmax, pn.ymax) for pn in page_numbers}

    for pn in sorted_pns:
        # Find words on the same line as this page number
        line_words = []
        for word, text in zip(words, texts):
            xmin, ymin, xmax, ymax = word

            # Skip the page number itself
            if (xmin, ymin, xmax, ymax) in pn_positions:
                continue

            # Check if word is on the same line
            if abs(ymin - pn.ymin) <= line_threshold:
                line_words.append(word)

        if line_words:
            # Calculate entry bounds
            all_y = [pn.ymin, pn.ymax] + [y for _, y, _, _ in line_words] + [y for _, _, _, y in line_words]
            entry_ymin = min(all_y)
            entry_ymax = max(all_y)

            entries.append(TOCEntry(
                page_number=pn,
                text_words=line_words,
                ymin=entry_ymin,
                ymax=entry_ymax
            ))

    return entries


def is_toc_block(words: List[Tuple[int, int, int, int]],
                texts: List[str],
                page_width: int,
                page_height: int) -> bool:
    """
    Simple check if a text block is likely a table of contents.

    This is a simplified version for use in layout analysis.

    Args:
        words: List of word bounding boxes
        texts: List of corresponding text strings
        page_width: Width of the page
        page_height: Height of the page

    Returns:
        True if the block is likely a TOC
    """
    is_toc, confidence, _ = detect_toc_page(words, texts, page_width, page_height, min_page_numbers=3)
    return is_toc and confidence >= 0.6


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)

    # Example: Simulated TOC page with right-aligned page numbers
    page_width = 600
    page_height = 800

    # Simulated words: (xmin, ymin, xmax, ymax)
    words = [
        (50, 100, 200, 120),   # "Chapter 1: Introduction"
        (520, 100, 550, 120),  # "1"
        (50, 140, 250, 160),   # "Chapter 2: Background"
        (520, 140, 550, 160),  # "5"
        (50, 180, 280, 200),   # "Chapter 3: Methodology"
        (510, 180, 550, 200),  # "12"
    ]

    texts = [
        "Chapter 1: Introduction",
        "1",
        "Chapter 2: Background",
        "5",
        "Chapter 3: Methodology",
        "12"
    ]

    is_toc, confidence, page_numbers = detect_toc_page(words, texts, page_width, page_height)

    print(f"Is TOC: {is_toc}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Page numbers found: {len(page_numbers)}")
    for pn in page_numbers:
        print(f"  - Page {pn.value} at line {pn.line_index}")

    if is_toc:
        entries = extract_toc_entries(words, texts, page_numbers, page_width)
        print(f"\nTOC Entries: {len(entries)}")
        for i, entry in enumerate(entries):
            print(f"  Entry {i+1}: {len(entry.text_words)} words, page number: {entry.page_number.value}")
