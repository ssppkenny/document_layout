"""
Table of Contents Detection using Multimodal Tree Decoder (MTD) approach

Based on the paper: "Multimodal Tree Decoder for Table of Contents Extraction in Document Images"
by Pengfei Hu, Zhenrong Zhang, Jianshu Zhang, Jun Du, Jiajia Wu

This module implements a simplified version of the MTD algorithm:
1. Extract multimodal features (visual, textual, layout) for each line
2. Classify lines as heading or normal
3. Build tree structure by predicting relationships between headings

Key concepts from the paper:
- Multimodal features: vision + text + layout
- Tree decoder with attention mechanism
- Three types of relationships: parent, sibling, identity
- Reference entity concept for building hierarchy
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LineFeatures:
    """Features extracted for a single line (entity) in the document"""
    # Layout features
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: float
    height: float

    # Normalized position features
    norm_x_left: float  # xmin / page_width
    norm_y_top: float   # ymin / page_height
    norm_x_right: float # xmax / page_width
    norm_y_bottom: float # ymax / page_height

    # Relative size features
    rel_width: float   # width / avg_width
    rel_height: float  # height / avg_height

    # Spacing features
    spacing_above: float  # Distance to previous line / avg_height
    spacing_below: float  # Distance to next line / avg_height

    # Text content
    text: str
    word_count: int

    # Derived features for TOC detection
    ends_with_number: bool
    has_dots: bool  # Contains leader dots (.....) common in TOC
    alignment_score: float  # How right-aligned is this line

    # Classification
    is_heading: bool = False
    confidence: float = 0.0

    line_index: int = 0


@dataclass
class TOCRelationship:
    """Represents relationship between two heading entities"""
    from_line_idx: int
    to_line_idx: int  # Reference entity
    relationship: str  # 'parent', 'sibling', 'identity'
    confidence: float


def extract_line_features(words: List[Tuple[int, int, int, int]],
                         texts: List[str],
                         page_width: int,
                         page_height: int) -> List[LineFeatures]:
    """
    Extract multimodal features for each line in the document.

    This is inspired by the MTD encoder which combines:
    - Vision features (position, size)
    - Text features (content, structure)
    - Layout features (spacing, alignment)

    Args:
        words: List of word bounding boxes (xmin, ymin, xmax, ymax)
        texts: List of corresponding text strings
        page_width: Width of the page
        page_height: Height of the page

    Returns:
        List of LineFeatures for each detected line
    """
    if not words or not texts:
        return []

    # Group words into lines based on vertical position
    line_threshold = 20  # pixels
    lines = []
    current_line = {'words': [], 'texts': []}
    current_y = words[0][1]

    for word, text in zip(words, texts):
        xmin, ymin, xmax, ymax = word

        if abs(ymin - current_y) > line_threshold and current_line['words']:
            lines.append(current_line)
            current_line = {'words': [], 'texts': []}
            current_y = ymin

        current_line['words'].append(word)
        current_line['texts'].append(text)

    if current_line['words']:
        lines.append(current_line)

    # Calculate average dimensions for normalization
    all_heights = []
    all_widths = []
    all_word_widths = []  # For detecting page numbers

    for line in lines:
        if line['words']:
            line_ymin = min(w[1] for w in line['words'])
            line_ymax = max(w[3] for w in line['words'])
            line_xmin = min(w[0] for w in line['words'])
            line_xmax = max(w[2] for w in line['words'])
            all_heights.append(line_ymax - line_ymin)
            all_widths.append(line_xmax - line_xmin)
            # Track individual word widths
            for word in line['words']:
                all_word_widths.append(word[2] - word[0])

    avg_height = np.mean(all_heights) if all_heights else 1.0
    avg_width = np.mean(all_widths) if all_widths else 1.0
    avg_word_width = np.mean(all_word_widths) if all_word_widths else 1.0

    # Extract features for each line
    line_features_list = []

    for i, line in enumerate(lines):
        if not line['words']:
            continue

        # Calculate line bounding box
        line_xmin = min(w[0] for w in line['words'])
        line_ymin = min(w[1] for w in line['words'])
        line_xmax = max(w[2] for w in line['words'])
        line_ymax = max(w[3] for w in line['words'])

        line_width = line_xmax - line_xmin
        line_height = line_ymax - line_ymin

        # Normalized position features (from MTD paper)
        norm_x_left = line_xmin / page_width if page_width > 0 else 0
        norm_y_top = line_ymin / page_height if page_height > 0 else 0
        norm_x_right = line_xmax / page_width if page_width > 0 else 0
        norm_y_bottom = line_ymax / page_height if page_height > 0 else 0

        # Relative size features (from MTD paper)
        rel_width = line_width / avg_width if avg_width > 0 else 1.0
        rel_height = line_height / avg_height if avg_height > 0 else 1.0

        # Spacing features (from MTD paper)
        spacing_above = 0.0
        if i > 0 and line_features_list:
            prev_line = line_features_list[-1]
            spacing_above = (line_ymin - prev_line.ymax) / avg_height if avg_height > 0 else 0

        spacing_below = 0.0  # Will be updated later

        # Text features
        line_text = ' '.join(line['texts'])
        word_count = len(line['texts'])

        # TOC-specific features
        # For detecting numbers, check the last word's width
        last_word_width = 0
        if line['words']:
            last_word = line['words'][-1]
            last_word_width = last_word[2] - last_word[0]

        ends_with_number = _ends_with_number(line_text, last_word_width, avg_word_width)
        has_dots = _has_leader_dots(line_text)
        alignment_score = _calculate_alignment_score(line_xmax, page_width)

        features = LineFeatures(
            xmin=line_xmin,
            ymin=line_ymin,
            xmax=line_xmax,
            ymax=line_ymax,
            width=line_width,
            height=line_height,
            norm_x_left=norm_x_left,
            norm_y_top=norm_y_top,
            norm_x_right=norm_x_right,
            norm_y_bottom=norm_y_bottom,
            rel_width=rel_width,
            rel_height=rel_height,
            spacing_above=spacing_above,
            spacing_below=0.0,
            text=line_text,
            word_count=word_count,
            ends_with_number=ends_with_number,
            has_dots=has_dots,
            alignment_score=alignment_score,
            line_index=i
        )

        line_features_list.append(features)

    # Update spacing_below for each line
    for i in range(len(line_features_list) - 1):
        curr_line = line_features_list[i]
        next_line = line_features_list[i + 1]
        curr_line.spacing_below = (next_line.ymin - curr_line.ymax) / avg_height if avg_height > 0 else 0

    return line_features_list


def _ends_with_number(text: str, word_width: float = 0, avg_word_width: float = 0) -> bool:
    """
    Check if text ends with a number (arabic or roman).
    If text is a placeholder (like 'w123'), use geometric heuristics instead.
    """
    if not text:
        return False

    # Check if this is a placeholder text pattern (w0, w1, etc.)
    if text.startswith('w') and text[1:].isdigit():
        # Use geometric heuristic: page numbers are typically narrow (1-3 digits)
        # Check if word is much narrower than average
        if word_width > 0 and avg_word_width > 0:
            ratio = word_width / avg_word_width
            # Numbers are typically 30-80% of average word width
            return 0.3 <= ratio <= 0.8
        # Can't determine without width info
        return False

    # Get the last word
    words = text.strip().split()
    if not words:
        return False

    last_word = words[-1]

    # Remove common punctuation
    cleaned = last_word.replace('.', '').replace(',', '').replace('-', '').strip()

    if not cleaned:
        return False

    # Check for arabic numerals
    if cleaned.isdigit():
        return True

    # Check for roman numerals
    roman_pattern = r'^[ivxlcdmIVXLCDM]+$'
    return bool(re.match(roman_pattern, cleaned))



def _has_leader_dots(text: str) -> bool:
    """Check if text contains leader dots (.....) typical in TOC"""
    # Look for 3 or more consecutive dots or periods
    return bool(re.search(r'\.{3,}', text))


def _calculate_alignment_score(xmax: float, page_width: int) -> float:
    """
    Calculate how right-aligned a line is.
    Returns a score from 0.0 (left-aligned) to 1.0 (right-aligned)
    """
    if page_width <= 0:
        return 0.0

    # Calculate distance from right margin as percentage
    distance_from_right = page_width - xmax
    normalized_distance = distance_from_right / page_width

    # Convert to alignment score (0 = far from right, 1 = close to right)
    alignment_score = 1.0 - normalized_distance

    return max(0.0, min(1.0, alignment_score))


def classify_heading_lines(line_features: List[LineFeatures]) -> List[LineFeatures]:
    """
    Classify lines as heading or normal entities.

    This is inspired by the MTD classifier module which uses BiGRU + fully connected layer.
    In this simplified version, we use rule-based heuristics based on the features.

    For TOC detection, we look for lines that:
    - End with numbers (page references)
    - Have leader dots
    - Are well-aligned vertically (right-aligned)
    - Have consistent spacing

    Args:
        line_features: List of LineFeatures

    Returns:
        Updated LineFeatures with is_heading and confidence set
    """
    if not line_features:
        return line_features

    # Count lines with TOC-like features
    lines_with_numbers = sum(1 for lf in line_features if lf.ends_with_number)
    lines_with_dots = sum(1 for lf in line_features if lf.has_dots)

    # Check if enough lines have TOC characteristics
    total_lines = len(line_features)
    number_ratio = lines_with_numbers / total_lines if total_lines > 0 else 0
    dots_ratio = lines_with_dots / total_lines if total_lines > 0 else 0

    # Calculate alignment consistency
    if lines_with_numbers >= 4:
        number_lines = [lf for lf in line_features if lf.ends_with_number]
        alignments = [lf.alignment_score for lf in number_lines]
        avg_alignment = np.mean(alignments)
        alignment_std = np.std(alignments)
    else:
        avg_alignment = 0.0
        alignment_std = 1.0

    # Classify each line
    for lf in line_features:
        # Calculate confidence score based on multiple factors
        confidence = 0.0

        # Factor 1: Ends with number
        if lf.ends_with_number:
            confidence += 0.4

        # Factor 2: Has leader dots
        if lf.has_dots:
            confidence += 0.2

        # Factor 3: Alignment score (high = likely TOC entry)
        if avg_alignment > 0.8:  # Well-aligned numbers
            confidence += 0.3 * lf.alignment_score

        # Factor 4: Consistent with other lines
        if number_ratio >= 0.4:  # At least 40% of lines have numbers
            confidence += 0.1

        # Threshold for classification
        lf.is_heading = confidence >= 0.5
        lf.confidence = min(1.0, confidence)

    return line_features


def build_toc_tree(heading_features: List[LineFeatures]) -> List[TOCRelationship]:
    """
    Build tree structure by predicting relationships between headings.

    This is inspired by the MTD decoder which uses:
    - GRU to track state
    - Attention mechanism to find reference entity
    - FFN to predict relationship type

    In this simplified version, we use heuristics based on:
    - Indentation (xmin position)
    - Text patterns (numbering like 1, 1.1, 1.1.1)
    - Spacing and size

    Args:
        heading_features: List of LineFeatures classified as headings

    Returns:
        List of TOCRelationship objects defining the tree structure
    """
    if not heading_features:
        return []

    relationships = []

    # Extract heading levels based on indentation and numbering
    for i, curr_heading in enumerate(heading_features):
        if i == 0:
            # First heading is root
            continue

        # Find reference entity (previous heading)
        # Check indentation to determine relationship
        ref_idx = i - 1
        ref_heading = heading_features[ref_idx]

        # Determine relationship based on indentation
        indent_diff = curr_heading.xmin - ref_heading.xmin

        # Check for hierarchical numbering patterns (e.g., 1.1 vs 1.2 vs 2.1)
        curr_numbers = _extract_heading_numbers(curr_heading.text)
        ref_numbers = _extract_heading_numbers(ref_heading.text)

        # Determine relationship type
        relationship_type = 'sibling'  # Default
        confidence = 0.7

        if len(curr_numbers) > len(ref_numbers):
            # More nesting levels -> likely a child (parent relationship)
            relationship_type = 'parent'
            confidence = 0.8
        elif len(curr_numbers) == len(ref_numbers):
            # Same nesting level -> likely a sibling
            if curr_numbers[:-1] == ref_numbers[:-1]:
                # Same parent prefix -> sibling
                relationship_type = 'sibling'
                confidence = 0.9
            else:
                # Different parent -> need to search back
                # For simplicity, mark as sibling
                relationship_type = 'sibling'
                confidence = 0.6
        elif indent_diff < -10:  # Moved left -> going up in hierarchy
            # Find actual parent by searching backwards
            relationship_type = 'sibling'
            confidence = 0.5

        # Check for identity relationship (continuation of same heading)
        if abs(indent_diff) < 5 and not curr_heading.ends_with_number and i > 0:
            # Very similar position and no page number -> might be continuation
            if not _looks_like_new_heading(curr_heading.text):
                relationship_type = 'identity'
                confidence = 0.7

        relationships.append(TOCRelationship(
            from_line_idx=i,
            to_line_idx=ref_idx,
            relationship=relationship_type,
            confidence=confidence
        ))

    return relationships


def _extract_heading_numbers(text: str) -> List[int]:
    """
    Extract hierarchical numbers from heading text.
    E.g., "2.1.3 Introduction" -> [2, 1, 3]
    """
    # Look for patterns like "1.2.3" at the start of the text
    match = re.match(r'^(\d+(?:\.\d+)*)', text.strip())
    if match:
        number_str = match.group(1)
        return [int(n) for n in number_str.split('.')]

    return []


def _looks_like_new_heading(text: str) -> bool:
    """Check if text looks like a new heading (vs continuation)"""
    # New headings typically:
    # - Start with numbers/letters
    # - Have capitalized first word
    # - Are not just lowercase continuation text

    text = text.strip()
    if not text:
        return False

    # Check if starts with number
    if text[0].isdigit():
        return True

    # Check if starts with capital letter and is not all caps
    if text[0].isupper():
        # Not all caps (which might be continuation of title)
        if not text.isupper():
            return True

    return False


def detect_toc_page_mtd(words: List[Tuple[int, int, int, int]],
                       texts: List[str],
                       page_width: int,
                       page_height: int,
                       min_toc_entries: int = 4,
                       min_confidence: float = 0.5) -> Tuple[bool, float, Dict]:
    """
    Detect if a page is a Table of Contents using MTD-inspired approach.

    This implements a simplified version of the MTD algorithm:
    1. Extract multimodal features (encoder)
    2. Classify lines as TOC entries (classifier)
    3. Build tree structure (decoder)

    Args:
        words: List of word bounding boxes
        texts: List of text strings
        page_width: Width of the page
        page_height: Height of the page
        min_toc_entries: Minimum number of TOC entries required
        min_confidence: Minimum average confidence score

    Returns:
        Tuple of (is_toc, confidence, metadata)
        - is_toc: True if page is detected as TOC
        - confidence: Overall confidence score (0.0 to 1.0)
        - metadata: Dictionary with detection details
    """
    # Step 1: Extract features (MTD Encoder)
    logger.debug("[MTD] Extracting line features...")
    line_features = extract_line_features(words, texts, page_width, page_height)

    if not line_features:
        return False, 0.0, {'reason': 'No lines detected'}

    logger.debug(f"[MTD] Extracted {len(line_features)} lines")

    # Step 2: Classify lines (MTD Classifier)
    logger.debug("[MTD] Classifying heading lines...")
    line_features = classify_heading_lines(line_features)

    # Count detected TOC entries (headings)
    toc_entries = [lf for lf in line_features if lf.is_heading]
    num_entries = len(toc_entries)

    logger.debug(f"[MTD] Detected {num_entries} potential TOC entries")

    # Check minimum entries threshold
    if num_entries < min_toc_entries:
        return False, 0.0, {
            'reason': f'Too few TOC entries: {num_entries} < {min_toc_entries}',
            'num_entries': num_entries
        }

    # Calculate average confidence
    avg_confidence = np.mean([lf.confidence for lf in toc_entries]) if toc_entries else 0.0

    logger.debug(f"[MTD] Average confidence: {avg_confidence:.2f}")

    # Check confidence threshold
    if avg_confidence < min_confidence:
        return False, avg_confidence, {
            'reason': f'Low confidence: {avg_confidence:.2f} < {min_confidence}',
            'num_entries': num_entries,
            'avg_confidence': avg_confidence
        }

    # Step 3: Build tree structure (MTD Decoder)
    logger.debug("[MTD] Building TOC tree structure...")
    relationships = build_toc_tree(toc_entries)

    # Additional validation: Check for right alignment of page numbers
    entries_with_numbers = [lf for lf in toc_entries if lf.ends_with_number]
    avg_alignment = 0.0  # Initialize default
    alignment_std = 1.0

    if entries_with_numbers:
        alignments = [lf.alignment_score for lf in entries_with_numbers]
        avg_alignment = np.mean(alignments)
        alignment_std = np.std(alignments)

        # Strong alignment suggests TOC
        if avg_alignment > 0.85 and alignment_std < 0.15:
            avg_confidence = min(1.0, avg_confidence + 0.1)

    # Metadata for debugging and analysis
    metadata = {
        'num_entries': num_entries,
        'avg_confidence': avg_confidence,
        'num_relationships': len(relationships),
        'entries_with_numbers': len(entries_with_numbers),
        'avg_alignment': avg_alignment,
        'lines_analyzed': len(line_features)
    }

    is_toc = True

    logger.debug(f"[MTD] TOC detection result: {is_toc}, confidence: {avg_confidence:.2f}")

    return is_toc, avg_confidence, metadata


def get_toc_entry_lines(words: List[Tuple[int, int, int, int]],
                       texts: List[str],
                       page_width: int,
                       page_height: int) -> List[Tuple[int, int, int, int, str]]:
    """
    Extract TOC entry lines with their bounding boxes and text.

    Returns:
        List of tuples (xmin, ymin, xmax, ymax, text) for each TOC entry
    """
    line_features = extract_line_features(words, texts, page_width, page_height)
    line_features = classify_heading_lines(line_features)

    toc_entries = []
    for lf in line_features:
        if lf.is_heading:
            toc_entries.append((
                int(lf.xmin),
                int(lf.ymin),
                int(lf.xmax),
                int(lf.ymax),
                lf.text
            ))

    return toc_entries
