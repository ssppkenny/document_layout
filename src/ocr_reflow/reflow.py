import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class Letter:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    bl: int

def detect_paragraphs_and_spacing_from_lines(lines: List[List[Letter]], original_width: int) -> Tuple[List[int], float]:
    """
    Detect paragraph breaks from pre-grouped lines by looking at horizontal indentation (xmin).
    
    Args:
        lines: List of lines, where each line is a list of Letter objects
        original_width: Width of the original image
        
    Returns:
        Tuple of (list of paragraph start indices, average line xmin)
    """
    if not lines:
        return [], 0.0
    
    # Flatten the lines to get all letters in order
    all_letters = []
    for line in lines:
        for letter in line:
            all_letters.append(letter)
    
    # Calculate the xmin of the first letter in each line
    line_first_xmins = []
    for line in lines:
        if line:
            # Find the leftmost letter in this line
            leftmost_letter = min(line, key=lambda l: l.xmin)
            line_first_xmins.append(leftmost_letter.xmin)
    
    if not line_first_xmins:
        return [0], 0.0
    
    # Calculate average xmin of first letters in lines
    avg_first_xmin = sum(line_first_xmins) / len(line_first_xmins)
    
    # Build mapping from letters to their indices in the flattened list
    letter_to_idx = {id(letter): idx for idx, letter in enumerate(all_letters)}
    
    # Detect paragraph breaks: when a line starts significantly to the right of average
    paragraph_starts = [0]  # First line always starts a paragraph
    
    # Calculate standard deviation of xmins to determine threshold
    if len(line_first_xmins) > 1:
        import math
        mean = avg_first_xmin
        variance = sum((x - mean) ** 2 for x in line_first_xmins) / len(line_first_xmins)
        std_dev = math.sqrt(variance)
        
        # Paragraph threshold: if xmin is more than 1.5 std deviations above mean
        threshold = mean + 1.5 * std_dev
        
        # Track cumulative index for flattened list
        cumulative_idx = 0
        for i, line in enumerate(lines):
            if i == 0:
                # Skip first line, but update cumulative index
                cumulative_idx += len(line)
                continue
            
            if line:
                # Find the leftmost letter in this line
                leftmost_letter = min(line, key=lambda l: l.xmin)
                
                if leftmost_letter.xmin > threshold:
                    # This line starts significantly to the right - likely a new paragraph
                    # Find the index of the first letter in this line in the flattened list
                    # First, sort letters in this line by x position to get reading order
                    sorted_line = sorted(line, key=lambda l: l.xmin)
                    if sorted_line:
                        first_letter_idx = letter_to_idx[id(sorted_line[0])]
                        if first_letter_idx not in paragraph_starts:
                            paragraph_starts.append(first_letter_idx)
            
            cumulative_idx += len(line)
    
    # Also look for significant jumps in xmin between consecutive lines
    # (alternative method for better detection)
    alternative_starts = [0]
    if len(lines) > 1:
        cumulative_idx = 0
        prev_line_first_xmin = None
        
        for i, line in enumerate(lines):
            if not line:
                cumulative_idx += len(line)
                continue
            
            # Find leftmost letter in current line
            leftmost_letter = min(line, key=lambda l: l.xmin)
            curr_xmin = leftmost_letter.xmin
            
            if i > 0 and prev_line_first_xmin is not None:
                # If current line starts at least 20 pixels to the right of previous line
                if curr_xmin > prev_line_first_xmin + 20:  # Absolute threshold
                    # Find the index of the first letter in this line
                    sorted_line = sorted(line, key=lambda l: l.xmin)
                    if sorted_line:
                        first_letter_idx = letter_to_idx[id(sorted_line[0])]
                        if first_letter_idx not in alternative_starts:
                            alternative_starts.append(first_letter_idx)
            
            prev_line_first_xmin = curr_xmin
            cumulative_idx += len(line)
    
    # Use the method that detects more paragraphs (usually alternative method is better)
    if len(alternative_starts) > len(paragraph_starts):
        paragraph_starts = alternative_starts
    
    # Sort paragraph starts and remove duplicates
    paragraph_starts = sorted(set(paragraph_starts))
    
    return paragraph_starts, avg_first_xmin

def is_letter_in_paragraph(letter_idx: int, paragraph_starts: List[int], total_letters: int) -> int:
    """
    Determine which paragraph a letter belongs to.
    
    Args:
        letter_idx: Index of the letter
        paragraph_starts: List of starting indices for each paragraph
        total_letters: Total number of letters
        
    Returns:
        Paragraph index (0-based)
    """
    for i in range(len(paragraph_starts) - 1):
        if paragraph_starts[i] <= letter_idx < paragraph_starts[i + 1]:
            return i
    
    # If we get here, it's in the last paragraph
    return len(paragraph_starts) - 1

def create_page_with_word_wrapping(lines: List[List[Letter]], original_image: np.ndarray,
                                 zoom_factor: float, new_page_width: int,
                                 left_margin: int = 50, right_margin: int = 50,
                                 top_margin: int = 50, bottom_margin: int = 50,
                                 line_spacing: int = 20, 
                                 paragraph_spacing_factor: float = 2.0,
                                 preserve_spacing: bool = True,
                                 background_color: tuple = (220, 220, 220),
                                 fixed_line_height: int = None,
                                 is_title: bool = False,
                                 debug: bool = False,
                                 preserve_line_breaks: bool = False,
                                 alignment: str = 'left') -> np.ndarray:
    """
    Create a new page image with letters reflowed with word wrapping.
    Letters are placed in original order, and new line begins when there's no space.
    Paragraph breaks are preserved with extra spacing.
    
    Args:
        lines: List of lines, where each line is a list of Letter objects in the exact order to be placed
        original_image: Source image containing the letters
        zoom_factor: Scaling factor for letters
        new_page_width: Width of the new page image
        left_margin: Left margin in pixels
        right_margin: Right Margin in pixels
        top_margin: Top margin in pixels
        bottom_margin: Bottom margin in pixels
        line_spacing: Vertical spacing between lines (used if fixed_line_height not provided)
        paragraph_spacing_factor: Multiplier for spacing between paragraphs
        preserve_spacing: Whether to preserve original spacing characteristics
        background_color: RGB tuple for page background
        fixed_line_height: If provided, use this constant height for all lines (overrides calculated heights)
        is_title: If True, treat as single paragraph with no indentation (for title blocks)
        preserve_line_breaks: If True, force a new output line at every original line boundary
            (used for code blocks, verse, lists — any content where line breaks are meaningful)
        alignment: Horizontal alignment for lines — 'left' (default), 'center', or 'right'.
            Only meaningful when preserve_line_breaks=True.
        left_margin, right_margin, top_margin, bottom_margin: Margins in pixels
        line_spacing: Additional spacing between lines within paragraph
        paragraph_spacing_factor: Factor to multiply line_spacing for paragraph spacing
        preserve_spacing: Whether to preserve original spacing between letters
        
    Returns:
        New page image with inserted letters
    """
    if not lines:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Flatten the lines to get all letters in order
    # Also track which letters start a new original line (for proper word spacing)
    all_letters = []
    line_start_indices = set()  # Indices where a new original line begins

    for line_idx, line in enumerate(lines):
        # Sort letters in each line by x position to get reading order
        sorted_line = sorted(line, key=lambda l: l.xmin)

        # Mark the index where this line starts (except the first line)
        if sorted_line and len(all_letters) > 0:
            line_start_indices.add(len(all_letters))

        all_letters.extend(sorted_line)
    
    if not all_letters:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Detect paragraph breaks by horizontal indentation using the original lines
    # Skip for titles - titles should be treated as single non-indented paragraph
    if is_title:
        # Title: no paragraph breaks, no indentation
        paragraph_starts = [0]  # Only one paragraph starting at index 0
        paragraph_spacing = 0   # No paragraph spacing
        paragraph_indentations = {}  # No indentations
        logger.debug("Title block: treating as single paragraph with no indentation")
    else:
        # Regular text: detect paragraphs
        paragraph_starts, avg_first_xmin = detect_paragraphs_and_spacing_from_lines(lines, original_image.shape[1])
        paragraph_spacing = int(line_spacing * paragraph_spacing_factor)

        logger.debug(f"Detected {len(paragraph_starts)} paragraphs")
        logger.debug(f"Paragraph starts at indices: {paragraph_starts}")
        logger.debug(f"Average first letter xmin: {avg_first_xmin}")

        # Calculate indentation for each paragraph from the original document
        # Map from paragraph index to indentation in the original page
        paragraph_indentations = {}
        cumulative_idx = 0
        for line_idx, line in enumerate(lines):
            if line:
                sorted_line = sorted(line, key=lambda l: l.xmin)
                first_letter_xmin = sorted_line[0].xmin

                # Check if this line starts a new paragraph
                for para_idx, para_start_idx in enumerate(paragraph_starts):
                    if cumulative_idx == para_start_idx:
                        # This line is the first line of a paragraph
                        indentation = first_letter_xmin  # Keep original indentation from page
                        paragraph_indentations[para_idx] = indentation
                        logger.debug(f"Paragraph {para_idx} starts at line {line_idx} with indentation (xmin): {indentation}")
                        break

                cumulative_idx += len(sorted_line)

    # Calculate available width for content
    available_width = new_page_width - left_margin - right_margin
    
    # Detect short lines in the original document
    # Calculate the width of each original line
    original_line_widths = []
    for line in lines:
        if line:
            sorted_line = sorted(line, key=lambda l: l.xmin)
            line_width = sorted_line[-1].xmax - sorted_line[0].xmin
            original_line_widths.append(line_width)

    # Calculate average line width to determine what is "short"
    if original_line_widths:
        avg_line_width = sum(original_line_widths) / len(original_line_widths)
        # A line is considered short if it's less than 70% of the average width
        short_line_threshold = avg_line_width * 0.7
        logger.debug(f"Average line width: {avg_line_width}, Short line threshold: {short_line_threshold}")
    else:
        short_line_threshold = float('inf')

    # Build a set of indices that are at the end of short lines
    short_line_ends = set()
    cumulative_idx = 0
    for line_idx, line in enumerate(lines):
        if line:
            sorted_line = sorted(line, key=lambda l: l.xmin)
            line_width = sorted_line[-1].xmax - sorted_line[0].xmin

            # Mark the last letter of this line if it's short
            if line_width < short_line_threshold:
                last_letter_idx = cumulative_idx + len(sorted_line) - 1
                short_line_ends.add(last_letter_idx)
                logger.debug(f"Line {line_idx} is short (width: {line_width}), marking letter {last_letter_idx} as end of short line")

            cumulative_idx += len(sorted_line)

    # Calculate average character width for word spacing
    avg_char_width = 0
    if all_letters:
        total_width = sum((letter.xmax - letter.xmin) for letter in all_letters)
        avg_char_width = int((total_width / len(all_letters)) * zoom_factor)

    logger.debug(f"Average character width (scaled): {avg_char_width}")
    logger.debug(f"Line start indices: {sorted(line_start_indices)}")

    # Debug placement records: list of dicts with x,y,w,h,status,letter
    debug_records = [] if debug else None

    # Store letter data in the exact order (as flattened from lines)
    letter_data = []
    for idx, letter in enumerate(all_letters):
        letter_region = original_image[letter.ymin:letter.ymax, letter.xmin:letter.xmax]
        if letter_region.size == 0:
            if debug:
                debug_records.append({'x': letter.xmin, 'y': letter.ymin,
                                      'w': letter.xmax - letter.xmin,
                                      'h': letter.ymax - letter.ymin,
                                      'status': 'zero_region', 'letter': letter})
            continue
            
        scaled_width = int((letter.xmax - letter.xmin) * zoom_factor)
        scaled_height = int((letter.ymax - letter.ymin) * zoom_factor)
        scaled_bl = int(letter.bl * zoom_factor)

        # Determine which paragraph this letter belongs to
        paragraph_idx = is_letter_in_paragraph(idx, paragraph_starts, len(all_letters))
        
        letter_data.append({
            'original_idx': idx,
            'letter': letter,
            'region': letter_region,
            'scaled_width': scaled_width,
            'scaled_height': scaled_height,
            'scaled_bl': scaled_bl,
            'paragraph_idx': paragraph_idx,
            'is_paragraph_start': idx in paragraph_starts,
            'is_end_of_short_line': idx in short_line_ends,
            'is_line_start': idx in line_start_indices  # New: track if this starts a new original line
        })
    
    if not letter_data:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Group letters into lines based on available width, preserving exact order
    lines_on_new_page = []
    current_line = []
    current_line_width = 0
    current_paragraph_idx = letter_data[0]['paragraph_idx'] if letter_data else -1
    current_line_paragraph_start = letter_data[0]['is_paragraph_start'] if letter_data else False
    current_line_indent = 0  # Track indentation for the current line

    for i, data in enumerate(letter_data):
        # Check if this is start of a new paragraph
        is_new_paragraph = data['paragraph_idx'] != current_paragraph_idx
        
        # Calculate space from previous letter if preserving spacing
        space = 0
        if i > 0 and not is_new_paragraph:
            # Check if this letter starts a new original line
            if data['is_line_start']:
                if preserve_line_breaks:
                    # Force a hard line break at every original line boundary
                    if current_line:
                        lines_on_new_page.append({
                            'letters': current_line,
                            'paragraph_idx': current_paragraph_idx,
                            'is_paragraph_start': current_line_paragraph_start
                        })
                        current_line = []
                        current_line_width = 0
                        current_line_paragraph_start = False
                    space = 0  # no leading space on the new line
                else:
                    # Add a word space (approximately one average character width)
                    space = int(avg_char_width * 0.35)
                    logger.debug(f"Adding word space before letter {i} (new original line)")
            elif preserve_spacing:
                # Get the actual previous letter in the provided order
                prev_data = letter_data[i-1]
                prev_letter = prev_data['letter']
                curr_letter = data['letter']

                # Calculate original space between these consecutive letters
                original_space = curr_letter.xmin - prev_letter.xmax
                if original_space > 0:
                    space = int(original_space * zoom_factor)

        # Calculate effective available width considering indentation
        effective_available_width = available_width - current_line_indent

        # Check if this letter would overflow the current line
        # Check if this letter would overflow the current line.
        # When preserve_line_breaks is True, never wrap mid-line — trust the original boundaries.
        would_overflow = (
            not preserve_line_breaks
            and current_line_width + space + data['scaled_width'] > effective_available_width
            and current_line
        )

        # Before wrapping, check if we're in the middle of a word and if splitting would leave only 1 letter on either side
        if would_overflow and current_line:
            # Check if we're splitting a word by looking at the space before current letter
            # Small space (< 0.5 avg char width) means we're in the middle of a word
            is_in_word = space < avg_char_width * 0.5
            if is_in_word:
                # Count letters in current word on current line (looking backward)
                letters_on_current_line = 0
                for j in range(len(current_line) - 1, -1, -1):
                    line_item = current_line[j]
                    space_before_this = line_item.get('space_before', 0)
                    # Stop counting if we hit a word boundary (large space before this letter)
                    if space_before_this >= avg_char_width * 0.5:
                        break
                    letters_on_current_line += 1

                # Count letters that would go on next line (looking forward from current position)
                letters_on_next_line = 1  # Current letter
                for j in range(i + 1, len(letter_data)):
                    next_data = letter_data[j]
                    # Stop if we hit a word boundary (line start)
                    if next_data.get('is_line_start', False):
                        break
                    # Calculate space before next letter
                    if j > 0:
                        next_prev_letter = letter_data[j-1]['letter']
                        next_curr_letter = next_data['letter']
                        next_space = next_curr_letter.xmin - next_prev_letter.xmax
                        if next_space * zoom_factor >= avg_char_width * 0.5:
                            break
                    letters_on_next_line += 1

                # Only prevent split if we have letters on current line AND either side has <=1 letter
                if letters_on_current_line > 0 and (letters_on_current_line <= 1 or letters_on_next_line <= 1):
                    # Remove letters from current line that are part of this word
                    word_letters = []
                    for _ in range(letters_on_current_line):
                        removed = current_line.pop()
                        current_line_width -= (removed['space_before'] + removed['scaled_width'])
                        word_letters.insert(0, removed)

                    # Start new line if current line has content
                    if current_line:
                        lines_on_new_page.append({
                            'letters': current_line,
                            'paragraph_idx': current_paragraph_idx,
                            'is_paragraph_start': current_line_paragraph_start
                        })
                        # Start fresh line with word letters (without current letter yet)
                        current_line = word_letters
                        current_line_width = sum(item['space_before'] + item['scaled_width'] for item in word_letters)
                    else:
                        # No content on current line, just start with the word letters
                        current_line = word_letters
                        current_line_width = sum(item['space_before'] + item['scaled_width'] for item in word_letters)

                    current_line_paragraph_start = False
                    current_line_indent = 0
                    # Reset space to 0 for first letter moved to new line
                    if current_line:
                        current_line[0]['space_before'] = 0
                        current_line_width = sum(item['space_before'] + item['scaled_width'] for item in current_line)
                    space = 0  # No space before current letter since we moved word beginning
                    would_overflow = False  # Don't wrap again - current letter will be added below

        if would_overflow:
            # Start a new line with this letter
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False
            current_line_indent = 0  # Reset indent for new line
            space = 0  # No space at beginning of new line

        # If this is a new paragraph and we're not at the beginning of a line,
        # force a new line
        if is_new_paragraph and current_line:
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False
            current_line_indent = 0  # Reset indent for new line
            space = 0
        
        # Update current paragraph
        if is_new_paragraph:
            current_paragraph_idx = data['paragraph_idx']
            current_line_paragraph_start = data['is_paragraph_start']
        
        # If this is the first letter in the line and it's a paragraph start,
        # mark the line as starting a paragraph and calculate indentation
        if len(current_line) == 0 and data['is_paragraph_start']:
            current_line_paragraph_start = True
            # Calculate indentation for this paragraph
            if data['paragraph_idx'] in paragraph_indentations:
                # Use a reasonable book-style indent: about 3 character widths
                avg_char_width = data['scaled_width']
                book_indent = int(avg_char_width * 3.5)
                current_line_indent = book_indent
                logger.debug(f"Applying book-style indentation {book_indent} to paragraph {data['paragraph_idx']}")

        # Add to current line
        data_with_space = data.copy()
        data_with_space['space_before'] = space
        current_line.append(data_with_space)
        current_line_width += space + data['scaled_width']

    # Add the last line
    if current_line:
        lines_on_new_page.append({
            'letters': current_line,
            'paragraph_idx': current_paragraph_idx,
            'is_paragraph_start': current_line_paragraph_start
        })
    
    # Calculate line heights and baselines for consistent spacing
    # Calculate line spacing using a more robust approach that handles outliers
    # Instead of using global maximum (which can be affected by one bad letter),
    # we'll use the 95th percentile to ignore extreme outliers

    # Calculate ONE constant line height for ALL lines
    # If fixed_line_height is provided, use it directly
    if fixed_line_height is not None and fixed_line_height > 0:
        global_line_height = fixed_line_height
        print(f"  [reflow] Using provided fixed line height: {global_line_height}px")
        logger.debug(f"Using provided fixed line height: {global_line_height}px")
    else:
        # Calculate from letter heights using 95th percentile to ignore extreme outliers
        # (subscripts, superscripts, unusual diacritics)
        all_above = []
        all_below = []

        for line in lines_on_new_page:
            if line['letters']:
                for item in line['letters']:
                    above = item['scaled_height'] - item['scaled_bl']
                    below = item['scaled_bl']
                    all_above.append(above)
                    all_below.append(below)

        # Calculate ONE constant line height for the entire document
        if len(all_above) > 0:
            # Use 95th percentile to ignore extreme outliers
            # This provides better spacing for typical text while still accommodating most letters
            import numpy as np
            percentile_above = int(np.percentile(all_above, 95))
            percentile_below = int(np.percentile(all_below, 95)) if len(all_below) > 0 else 0

            # Calculate text height (font size) based on robust statistics
            # Note: percentile_below may be 0 if there are no descenders (g, p, y, j)
            text_height = percentile_above + percentile_below

            # Apply typography best practice: line-height = 1.5x font size for body text
            # This means adding 0.5x the text height as spacing
            # (line_height = text_height * 1.5 = text_height + text_height * 0.5)
            optimal_line_spacing = int(text_height * 0.5)
            global_line_height = text_height + optimal_line_spacing

            print(f"  [reflow] Text height: {text_height}px (95th percentile: above={percentile_above}px, below={percentile_below}px), spacing: {optimal_line_spacing}px")
        else:
            # Fallback
            global_line_height = 40

        print(f"  [reflow] Calculated line height: {global_line_height}px")
        logger.debug(f"Calculated constant line height: {global_line_height}px")

    # Use the same line height for ALL lines
    line_heights = [global_line_height] * len(lines_on_new_page)

    print(f"  [reflow] Created {len(line_heights)} line heights, all set to {global_line_height}px")
    if len(line_heights) > 0:
        print(f"  [reflow] First 3 line heights: {line_heights[:3]}")

    logger.debug(f"Using constant line height: {global_line_height}px for all {len(lines_on_new_page)} lines")

    # Also collect global statistics for baseline calculations
    # Use MAXIMUM values to ensure NO letters are clipped (including tall merged i, j and descenders g, p, y)
    all_above_baseline = []
    all_below_baseline = []

    for line in lines_on_new_page:
        if line['letters']:
            for item in line['letters']:
                above = item['scaled_height'] - item['scaled_bl']
                below = item['scaled_bl']
                all_above_baseline.append(above)
                all_below_baseline.append(below)

    if all_above_baseline and all_below_baseline:
        # Use MAXIMUM values to prevent any clipping
        # Do NOT use percentiles - this causes tall letters and descenders to be clipped
        max_above_baseline = max(all_above_baseline)
        max_below_baseline = max(all_below_baseline)

        print(f"  [reflow] Baseline stats: max_above={max_above_baseline}px, max_below={max_below_baseline}px")
    else:
        # Fallback if no letters
        max_above_baseline = 20
        max_below_baseline = 5

    # Calculate total height needed using per-line heights
    total_height = top_margin
    previous_paragraph_idx = -1
    
    for line_idx, line in enumerate(lines_on_new_page):
        if not line['letters']:
            continue
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            total_height += paragraph_spacing
        
        # Use per-line height for appropriate spacing based on actual letter sizes
        if line_idx < len(line_heights):
            total_height += line_heights[line_idx]
        else:
            total_height += 40  # Fallback

        previous_paragraph_idx = line['paragraph_idx']

    # Add bottom margin
    total_height += bottom_margin
    
    # Create blank page with detected background color
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8)
    new_page[:] = background_color

    # Place letters line by line in the order they appear
    current_y = top_margin
    previous_paragraph_idx = -1
    
    for line_idx, line in enumerate(lines_on_new_page):
        if not line['letters']:
            continue
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            current_y += paragraph_spacing
            # Draw a faint line to mark paragraph break (for debugging)
            cv2.line(new_page, 
                    (left_margin, current_y - paragraph_spacing // 2),
                    (new_page_width - right_margin, current_y - paragraph_spacing // 2),
                    (200, 200, 200), 1, cv2.LINE_AA)
        
        # Calculate baseline for this line: use the maximum baseline shift among all letters
        # This ensures all letters on the line share the same baseline
        max_baseline = max(item['scaled_bl'] for item in line['letters'])

        # The baseline position on the page: we need to ensure there's enough space above
        # the baseline for the tallest letter. Use the global max_above_baseline to ensure
        # consistent spacing across all lines
        baseline_y = current_y + max_above_baseline

        # Place letters in this line in the order they appear
        # For preserve_line_breaks mode, apply alignment by computing the total line width first
        if preserve_line_breaks and alignment in ('center', 'right'):
            line_total_width = sum(item['space_before'] + item['scaled_width'] for item in line['letters'])
            if alignment == 'center':
                current_x = max(left_margin, left_margin + (available_width - line_total_width) // 2)
            else:  # right
                current_x = max(left_margin, new_page_width - right_margin - line_total_width)
        else:
            current_x = left_margin
        
        # If this line starts a paragraph, apply the original indentation from that paragraph
        # (only for non-preserve-lines mode — preserve-lines uses alignment instead)
        if not preserve_line_breaks and line['is_paragraph_start'] and line['paragraph_idx'] in paragraph_indentations:
            # Scale the indentation based on zoom factor
            original_indent = paragraph_indentations[line['paragraph_idx']]
            # Use a reasonable book-style indent: about 3 character widths
            # Calculate approximate character width from the first letter in this line if available
            if line['letters']:
                avg_char_width = line['letters'][0]['scaled_width']
                # Reduce original indent to about 3-4 character widths (standard book indentation)
                book_indent = int(avg_char_width * 3.5)
            else:
                book_indent = 20
            current_x += book_indent
            logger.debug(f"Applying book-style indentation {book_indent} to paragraph {line['paragraph_idx']}")

        for item in line['letters']:
            # Add space before letter if not first in line
            if current_x > left_margin:
                current_x += item['space_before']
            
            # Resize letter
            if item['scaled_width'] > 0 and item['scaled_height'] > 0:
                resized_letter = cv2.resize(item['region'], 
                                          (item['scaled_width'], item['scaled_height']))
            else:
                if debug:
                    debug_records.append({'x': current_x, 'y': current_y,
                                          'w': max(1, item['scaled_width']),
                                          'h': max(1, item['scaled_height']),
                                          'status': 'zero_size', 'letter': item['letter']})
                continue
            
            # Calculate vertical position: place letter so its baseline aligns with the line's baseline
            # The baseline in a letter is at: bottom - bl = ymin + (height - bl)
            # So the top of letter should be at: baseline_y - (scaled_height - scaled_bl)
            y_offset = baseline_y - item['scaled_height'] + item['scaled_bl']

            # Ensure coordinates are within bounds
            # Respect the right margin when checking horizontal bounds
            max_x = new_page_width - right_margin
            y_start = max(0, y_offset)
            y_end = min(y_offset + item['scaled_height'], total_height)
            x_start = current_x
            x_end = min(current_x + item['scaled_width'], max_x)

            # Warn if letter is being clipped
            if y_offset < 0:
                above_bl = item['scaled_height'] - item['scaled_bl']
                print(f"⚠️  WARNING: Letter clipped at top! y_offset={y_offset}, needs {above_bl}px above baseline, have {max_above_baseline}px")
                logger.warning(f"Letter clipped: y_offset={y_offset}, height={item['scaled_height']}, bl={item['scaled_bl']}, baseline_y={baseline_y}")

            # Place letter if it fits
            if x_end > x_start and y_end > y_start:
                # Adjust if letter would go out of bounds
                if (y_end - y_start) != item['scaled_height'] or (x_end - x_start) != item['scaled_width']:
                    # Crop the resized letter to fit
                    crop_height = y_end - y_start
                    crop_width = x_end - x_start
                    resized_letter = resized_letter[:crop_height, :crop_width]
                
                new_page[y_start:y_end, x_start:x_end] = resized_letter
                if debug:
                    debug_records.append({'x': x_start, 'y': y_start,
                                          'w': x_end - x_start, 'h': y_end - y_start,
                                          'status': 'placed', 'letter': item['letter']})
            else:
                if debug:
                    debug_records.append({'x': x_start, 'y': y_start,
                                          'w': item['scaled_width'], 'h': item['scaled_height'],
                                          'status': 'clipped', 'letter': item['letter']})

            current_x += item['scaled_width']
        
        # Move to next line using the appropriate height for this line
        if line_idx < len(line_heights):
            current_y += line_heights[line_idx]
        else:
            current_y += 40  # Fallback

        previous_paragraph_idx = line['paragraph_idx']
    
    if debug:
        return new_page, debug_records
    return new_page

def create_page_with_bounding_boxes_wrapping(lines: List[List[Letter]], original_image: np.ndarray,
                                           zoom_factor: float, new_page_width: int,
                                           left_margin: int = 50, right_margin: int = 50,
                                           top_margin: int = 50, bottom_margin: int = 50,
                                           line_spacing: int = 20,
                                           paragraph_spacing_factor: float = 2.0,
                                           preserve_spacing: bool = True,
                                           box_color=(0, 0, 255), 
                                           baseline_color=(0, 255, 0),
                                           paragraph_color=(255, 0, 0),
                                           background_color: tuple = (220, 220, 220),
                                           is_title: bool = False) -> np.ndarray:
    """
    Create a visualization with bounding boxes, arranged with word wrapping.
    
    Args:
        lines: List of lines, where each line is a list of Letter objects in the exact order to be placed
        original_image: Source image for paragraph detection
        zoom_factor: Scaling factor for letters
        new_page_width: Width of the new page image
        margins: Margins in pixels
        line_spacing: Additional spacing between lines within paragraph
        paragraph_spacing_factor: Factor to multiply line_spacing for paragraph spacing
        preserve_spacing: Whether to preserve original spacing between letters
        box_color: Color for bounding boxes (BGR)
        baseline_color: Color for baseline (BGR)
        paragraph_color: Color for paragraph markers (BGR)
        background_color: RGB tuple for page background

    Returns:
        New page image with drawn bounding boxes and baselines
    """
    if not lines:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Flatten the lines to get all letters in order
    # Also track which letters start a new original line (for proper word spacing)
    all_letters = []
    line_start_indices = set()  # Indices where a new original line begins

    for line_idx, line in enumerate(lines):
        # Sort letters in each line by x position to get reading order
        sorted_line = sorted(line, key=lambda l: l.xmin)

        # Mark the index where this line starts (except the first line)
        if sorted_line and len(all_letters) > 0:
            line_start_indices.add(len(all_letters))

        all_letters.extend(sorted_line)
    
    if not all_letters:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Detect paragraph breaks by horizontal indentation using the original lines
    # Skip for titles - titles should be treated as single non-indented paragraph
    if is_title:
        # Title: no paragraph breaks, no indentation
        paragraph_starts = [0]  # Only one paragraph starting at index 0
        paragraph_spacing = 0   # No paragraph spacing
        paragraph_indentations = {}  # No indentations
        logger.debug("Title block: treating as single paragraph with no indentation")
    else:
        # Regular text: detect paragraphs
        paragraph_starts, avg_first_xmin = detect_paragraphs_and_spacing_from_lines(lines, original_image.shape[1])
        paragraph_spacing = int(line_spacing * paragraph_spacing_factor)

        logger.debug(f"Detected {len(paragraph_starts)} paragraphs")
        logger.debug(f"Paragraph starts at indices: {paragraph_starts}")
        logger.debug(f"Average first letter xmin: {avg_first_xmin}")

        # Calculate indentation for each paragraph from the original document
        # Map from paragraph index to indentation in the original page
        paragraph_indentations = {}
        cumulative_idx = 0
        for line_idx, line in enumerate(lines):
            if line:
                sorted_line = sorted(line, key=lambda l: l.xmin)
                first_letter_xmin = sorted_line[0].xmin

                # Check if this line starts a new paragraph
                for para_idx, para_start_idx in enumerate(paragraph_starts):
                    if cumulative_idx == para_start_idx:
                        # This line is the first line of a paragraph
                        indentation = first_letter_xmin  # Keep original indentation from page
                        paragraph_indentations[para_idx] = indentation
                        logger.debug(f"Paragraph {para_idx} starts at line {line_idx} with indentation (xmin): {indentation}")
                        break

                cumulative_idx += len(sorted_line)

    # Calculate available width for content
    available_width = new_page_width - left_margin - right_margin
    
    # Detect short lines in the original document
    original_line_widths = []
    for line in lines:
        if line:
            sorted_line = sorted(line, key=lambda l: l.xmin)
            line_width = sorted_line[-1].xmax - sorted_line[0].xmin
            original_line_widths.append(line_width)

    if original_line_widths:
        avg_line_width = sum(original_line_widths) / len(original_line_widths)
        short_line_threshold = avg_line_width * 0.7
    else:
        short_line_threshold = float('inf')

    short_line_ends = set()
    cumulative_idx = 0
    for line_idx, line in enumerate(lines):
        if line:
            sorted_line = sorted(line, key=lambda l: l.xmin)
            line_width = sorted_line[-1].xmax - sorted_line[0].xmin

            if line_width < short_line_threshold:
                last_letter_idx = cumulative_idx + len(sorted_line) - 1
                short_line_ends.add(last_letter_idx)

            cumulative_idx += len(sorted_line)

    # Calculate average character width for word spacing
    avg_char_width = 0
    if all_letters:
        total_width = sum((letter.xmax - letter.xmin) for letter in all_letters)
        avg_char_width = int((total_width / len(all_letters)) * zoom_factor)

    # Store letter data in the exact order (as flattened from lines)
    letter_data = []
    for idx, letter in enumerate(all_letters):
        scaled_width = int((letter.xmax - letter.xmin) * zoom_factor)
        scaled_height = int((letter.ymax - letter.ymin) * zoom_factor)
        scaled_bl = int(letter.bl * zoom_factor)
        
        # Determine which paragraph this letter belongs to
        paragraph_idx = is_letter_in_paragraph(idx, paragraph_starts, len(all_letters))
        
        letter_data.append({
            'original_idx': idx,
            'letter': letter,
            'scaled_width': scaled_width,
            'scaled_height': scaled_height,
            'scaled_bl': scaled_bl,
            'paragraph_idx': paragraph_idx,
            'is_paragraph_start': idx in paragraph_starts,
            'is_end_of_short_line': idx in short_line_ends,
            'is_line_start': idx in line_start_indices  # New: track if this starts a new original line
        })
    
    if not letter_data:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Group letters into lines based on available width, preserving exact order
    lines_on_new_page = []
    current_line = []
    current_line_width = 0
    current_paragraph_idx = letter_data[0]['paragraph_idx'] if letter_data else -1
    current_line_paragraph_start = letter_data[0]['is_paragraph_start'] if letter_data else False
    current_line_indent = 0  # Track indentation for the current line

    for i, data in enumerate(letter_data):
        # Check if this is start of a new paragraph
        is_new_paragraph = data['paragraph_idx'] != current_paragraph_idx
        
        # Calculate space from previous letter if preserving spacing
        space = 0
        if i > 0 and not is_new_paragraph:
            # Check if this letter starts a new original line
            if data['is_line_start']:
                # Add a word space (approximately one average character width)
                space = int(avg_char_width * 0.35)
                logger.debug(f"Adding word space before letter {i} (new original line)")
            elif preserve_spacing:
                # Get the actual previous letter in the provided order
                prev_data = letter_data[i-1]
                prev_letter = prev_data['letter']
                curr_letter = data['letter']

                # Calculate original space between these consecutive letters
                original_space = curr_letter.xmin - prev_letter.xmax
                if original_space > 0:
                    space = int(original_space * zoom_factor)

        # Calculate effective available width considering indentation
        effective_available_width = available_width - current_line_indent

        # Check if this letter would overflow the current line
        # Check if this letter would overflow the current line.
        # When preserve_line_breaks is True, never wrap mid-line — trust the original boundaries.
        would_overflow = (
            not preserve_line_breaks
            and current_line_width + space + data['scaled_width'] > effective_available_width
            and current_line
        )

        # Before wrapping, check if we're in the middle of a word and if splitting would leave only 1 letter on either side
        if would_overflow and current_line:
            # Check if we're splitting a word by looking at the space before current letter
            # Small space (< 0.5 avg char width) means we're in the middle of a word
            is_in_word = space < avg_char_width * 0.5
            if is_in_word:
                # Count letters in current word on current line (looking backward)
                letters_on_current_line = 0
                for j in range(len(current_line) - 1, -1, -1):
                    line_item = current_line[j]
                    space_before_this = line_item.get('space_before', 0)
                    # Stop counting if we hit a word boundary (large space before this letter)
                    if space_before_this >= avg_char_width * 0.5:
                        break
                    letters_on_current_line += 1

                # Count letters that would go on next line (looking forward from current position)
                letters_on_next_line = 1  # Current letter
                for j in range(i + 1, len(letter_data)):
                    next_data = letter_data[j]
                    # Stop if we hit a word boundary (line start)
                    if next_data.get('is_line_start', False):
                        break
                    # Calculate space before next letter
                    if j > 0:
                        next_prev_letter = letter_data[j-1]['letter']
                        next_curr_letter = next_data['letter']
                        next_space = next_curr_letter.xmin - next_prev_letter.xmax
                        if next_space * zoom_factor >= avg_char_width * 0.5:
                            break
                    letters_on_next_line += 1

                # Only prevent split if we have letters on current line AND either side has <=1 letter
                if letters_on_current_line > 0 and (letters_on_current_line <= 1 or letters_on_next_line <= 1):
                    # Remove letters from current line that are part of this word
                    word_letters = []
                    for _ in range(letters_on_current_line):
                        removed = current_line.pop()
                        current_line_width -= (removed['space_before'] + removed['scaled_width'])
                        word_letters.insert(0, removed)

                    # Start new line if current line has content
                    if current_line:
                        lines_on_new_page.append({
                            'letters': current_line,
                            'paragraph_idx': current_paragraph_idx,
                            'is_paragraph_start': current_line_paragraph_start
                        })
                        # Start fresh line with word letters (without current letter yet)
                        current_line = word_letters
                        current_line_width = sum(item['space_before'] + item['scaled_width'] for item in word_letters)
                    else:
                        # No content on current line, just start with the word letters
                        current_line = word_letters
                        current_line_width = sum(item['space_before'] + item['scaled_width'] for item in word_letters)

                    current_line_paragraph_start = False
                    current_line_indent = 0
                    # Reset space to 0 for first letter moved to new line
                    if current_line:
                        current_line[0]['space_before'] = 0
                        current_line_width = sum(item['space_before'] + item['scaled_width'] for item in current_line)
                    space = 0  # No space before current letter since we moved word beginning
                    would_overflow = False  # Don't wrap again - current letter will be added below

        if would_overflow:
            # Start a new line with this letter
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False
            current_line_indent = 0  # Reset indent for new line
            space = 0  # No space at beginning of new line

        # If this is a new paragraph and we're not at the beginning of a line,
        # force a new line
        if is_new_paragraph and current_line:
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False
            current_line_indent = 0  # Reset indent for new line
            space = 0
        
        # Update current paragraph
        if is_new_paragraph:
            current_paragraph_idx = data['paragraph_idx']
            current_line_paragraph_start = data['is_paragraph_start']
        
        # If this is the first letter in the line and it's a paragraph start,
        # mark the line as starting a paragraph and calculate indentation
        if len(current_line) == 0 and data['is_paragraph_start']:
            current_line_paragraph_start = True
            # Calculate indentation for this paragraph
            if data['paragraph_idx'] in paragraph_indentations:
                # Use a reasonable book-style indent: about 3 character widths
                avg_char_width = data['scaled_width']
                book_indent = int(avg_char_width * 3.5)
                current_line_indent = book_indent
                logger.debug(f"Applying book-style indentation {book_indent} to paragraph {data['paragraph_idx']} in visualization")

        # Add to current line
        data_with_space = data.copy()
        data_with_space['space_before'] = space
        current_line.append(data_with_space)
        current_line_width += space + data['scaled_width']

        # Check if this letter is at the end of a short line in the original
        # If so, force a new line after it (unless it's the last letter)
#         if data['is_end_of_short_line'] and i < len(letter_data) - 1:
#             lines_on_new_page.append({
#                 'letters': current_line,
#                 'paragraph_idx': current_paragraph_idx,
#                 'is_paragraph_start': current_line_paragraph_start
#             })
#             current_line = []
#             current_line_width = 0
#             current_line_paragraph_start = False

    # Add the last line
    if current_line:
        lines_on_new_page.append({
            'letters': current_line,
            'paragraph_idx': current_paragraph_idx,
            'is_paragraph_start': current_line_paragraph_start
        })
    
    # Calculate ONE constant line height for entire document (same as main reflow function)
    all_heights = []

    for line in lines_on_new_page:
        if line['letters']:
            for item in line['letters']:
                all_heights.append(item['scaled_height'])

    if len(all_heights) > 0:
        # Use 95th percentile to ignore extreme outliers
        import numpy as np
        percentile_height = int(np.percentile(all_heights, 95))

        # Apply typography best practice: line-height = 1.5x font size for body text
        # This means adding 0.5x the text height as spacing
        optimal_line_spacing = int(percentile_height * 0.5)
        global_line_height = percentile_height + optimal_line_spacing
    else:
        global_line_height = 40

    # Calculate total height needed
    total_height = top_margin
    previous_paragraph_idx = -1
    
    for line in lines_on_new_page:
        if not line['letters']:
            continue
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            total_height += paragraph_spacing
        
        total_height += global_line_height
        previous_paragraph_idx = line['paragraph_idx']

    # Add bottom margin
    total_height += bottom_margin
    
    # Create blank page with detected background color
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8)
    new_page[:] = background_color

    # Draw bounding boxes line by line in the exact order
    current_y = top_margin
    previous_paragraph_idx = -1
    
    for line_idx, line in enumerate(lines_on_new_page):
        if not line['letters']:
            continue
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            current_y += paragraph_spacing
            # Draw paragraph separator line
            cv2.line(new_page, 
                    (left_margin, current_y - paragraph_spacing // 2),
                    (new_page_width - right_margin, current_y - paragraph_spacing // 2),
                    paragraph_color, 2, cv2.LINE_AA)
            
            # Add paragraph marker text
            marker_text = f"Paragraph {line['paragraph_idx'] + 1} Start (Indented)"
            cv2.putText(new_page, marker_text, 
                       (left_margin, current_y - paragraph_spacing // 2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, paragraph_color, 1)
        
        # Calculate max height for this line
        max_line_height = max(item['scaled_height'] for item in line['letters'])
        
        # Calculate baseline for this line: use the maximum baseline shift among all letters
        max_baseline = max(item['scaled_bl'] for item in line['letters'])

        # The baseline position on the page
        baseline_y = current_y + max_baseline

        # Place letters in this line in order
        current_x = left_margin
        
        # If this line starts a paragraph, apply the original indentation from that paragraph
        if line['is_paragraph_start'] and line['paragraph_idx'] in paragraph_indentations:
            # Scale the indentation based on zoom factor
            original_indent = paragraph_indentations[line['paragraph_idx']]
            # Use a reasonable book-style indent: about 3 character widths
            # Calculate approximate character width from the first letter in this line if available
            if line['letters']:
                avg_char_width = line['letters'][0]['scaled_width']
                # Reduce original indent to about 3-4 character widths (standard book indentation)
                book_indent = int(avg_char_width * 3.5)
            else:
                book_indent = 20
            current_x += book_indent
            logger.debug(f"Applying book-style indentation {book_indent} to paragraph {line['paragraph_idx']} in visualization")

        for item in line['letters']:
            # Add space before letter if not first in line
            if current_x > left_margin:
                current_x += item['space_before']
            
            # Calculate vertical position based on baseline
            # The baseline is at: baseline_y (shared for all letters on the line)
            # The top of the letter is at: baseline_y - (scaled_height - scaled_bl)
            y_offset = baseline_y - item['scaled_height'] + item['scaled_bl']

            # Draw bounding box
            x1 = current_x
            y1 = y_offset
            x2 = current_x + item['scaled_width']
            y2 = y_offset + item['scaled_height']
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(new_page, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw baseline
                cv2.line(new_page, (x1, baseline_y), (x2, baseline_y), baseline_color, 1)
                
                # Add original index and paragraph info
                text = f"{item['original_idx']}"
                if item['original_idx'] in paragraph_starts:
                    text += " (P)"
                cv2.putText(new_page, text, 
                           (x1 + 2, y1 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            current_x += item['scaled_width']
        
        # Move to next line using global constant height
        current_y += global_line_height

        previous_paragraph_idx = line['paragraph_idx']
    
    # Draw margin lines for visualization
    cv2.line(new_page, (left_margin, 0), (left_margin, total_height), (200, 200, 200), 1)
    cv2.line(new_page, (new_page_width - right_margin, 0), 
             (new_page_width - right_margin, total_height), (200, 200, 200), 1)
    cv2.line(new_page, (0, top_margin), (new_page_width, top_margin), (200, 200, 200), 1)
    cv2.line(new_page, (0, total_height - bottom_margin), 
             (new_page_width, total_height - bottom_margin), (200, 200, 200), 1)
    
    # Add info text
    info_text = f"Letters: {len(all_letters)} | Paragraphs: {len(paragraph_starts)} | Width: {new_page_width}"
    cv2.putText(new_page, info_text, (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return new_page

# Test with indented paragraphs using lines as input
if __name__ == "__main__":
    # Create test lines with indented paragraphs
    lines = []
    
    # Paragraph 1: Not indented (starts at x=10)
    line1 = [
        Letter(xmin=10, ymin=30, xmax=30, ymax=55, bl=5),   # Index 0
        Letter(xmin=40, ymin=30, xmax=60, ymax=55, bl=5),   # Index 1
        Letter(xmin=70, ymin=30, xmax=90, ymax=55, bl=5),   # Index 2
    ]
    lines.append(line1)
    
    # Line 2 of paragraph 1 (same x position)
    line2 = [
        Letter(xmin=10, ymin=65, xmax=30, ymax=90, bl=5),   # Index 3
        Letter(xmin=40, ymin=65, xmax=60, ymax=90, bl=5),   # Index 4
    ]
    lines.append(line2)
    
    # Line 3 of paragraph 1
    line3 = [
        Letter(xmin=10, ymin=100, xmax=30, ymax=125, bl=5), # Index 5
    ]
    lines.append(line3)
    
    # Paragraph 2: INDENTED (starts at x=50 instead of x=10)
    line4 = [
        Letter(xmin=50, ymin=130, xmax=70, ymax=155, bl=5), # Index 6 - Paragraph start!
        Letter(xmin=80, ymin=130, xmax=100, ymax=155, bl=5), # Index 7
    ]
    lines.append(line4)
    
    # Line 2 of paragraph 2 (also indented)
    line5 = [
        Letter(xmin=50, ymin=165, xmax=70, ymax=190, bl=5), # Index 8
        Letter(xmin=80, ymin=165, xmax=100, ymax=190, bl=5), # Index 9
    ]
    lines.append(line5)
    
    # Paragraph 3: NOT indented (back to x=10)
    line6 = [
        Letter(xmin=10, ymin=200, xmax=30, ymax=225, bl=5), # Index 10 - Paragraph start!
        Letter(xmin=40, ymin=200, xmax=60, ymax=225, bl=5), # Index 11
    ]
    lines.append(line6)
    
    # Create original image
    original_image = np.ones((250, 120, 3), dtype=np.uint8) * 255
    
    # Put text for visualization
    cv2.putText(original_image, "P1L1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(original_image, "P1L2", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(original_image, "P1L3", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.putText(original_image, "P2L1", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(original_image, "P2L2", (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.putText(original_image, "P3L1", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Test with narrow page to force reflow
    zoom_factor = 1.5
    new_page_width = 180  # Narrow to force wrapping
    
    logger.debug("Testing paragraph detection by horizontal indentation with lines input...")
    logger.debug(f"Total lines: {len(lines)}")
    logger.debug("Expected: Paragraph 1 (lines 0-2), Paragraph 2 (lines 3-4), Paragraph 3 (line 5)")
    logger.debug("Paragraph 2 should be indented (starts at x=50 vs x=10)")

    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )
    
    # Create visualization with bounding boxes
    page_boxes = create_page_with_bounding_boxes_wrapping(
        lines, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )
    
    # Display
    cv2.imshow("Original with indented paragraphs", original_image)
    cv2.imshow("Reflowed with paragraph spacing", page_reflowed)
    cv2.imshow("Bounding Boxes with paragraph markers", page_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save for reference
    cv2.imwrite("reflowed_lines_input.jpg", page_reflowed)
    cv2.imwrite("boxes_lines_input.jpg", page_boxes)


def create_toc_page_with_right_alignment(lines: List[List[Letter]], original_image: np.ndarray,
                                         zoom_factor: float, new_page_width: int,
                                         left_margin: int = 50, right_margin: int = 50,
                                         top_margin: int = 50, bottom_margin: int = 50,
                                         background_color: tuple = (220, 220, 220),
                                         fixed_line_height: int = None) -> np.ndarray:
    """
    Create a reflowed TOC page with right-aligned page numbers.

    Simplified approach: preserve original line structure, scale uniformly,
    and maintain right alignment by preserving relative positions.

    Args:
        lines: List of lines, where each line is a list of Letter objects
        original_image: Source image containing the letters
        zoom_factor: Scaling factor for letters
        new_page_width: Width of the new page
        left_margin: Left margin in pixels
        right_margin: Right margin in pixels
        top_margin: Top margin in pixels
        bottom_margin: Bottom margin in pixels
        background_color: RGB tuple for page background
        fixed_line_height: If provided, use this constant height for all lines

    Returns:
        Reflowed page image with TOC structure preserved
    """
    if not lines:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    print("="*80)
    print(f"  ★★★ TOC REFLOW FUNCTION CALLED ★★★")
    print(f"  [TOC] Processing {len(lines)} lines with TOC-specific layout")
    print("="*80)

    # Calculate original page width from letters
    all_x = []
    for line in lines:
        for letter in line:
            all_x.extend([letter.xmin, letter.xmax])

    if not all_x:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    original_content_width = max(all_x) - min(all_x)
    available_width = new_page_width - left_margin - right_margin

    # Calculate scaling factor to fit content
    content_scale = min(zoom_factor, available_width / original_content_width if original_content_width > 0 else zoom_factor)
    print(f"  [TOC] Content scale: {content_scale:.2f}")

    # Process each line independently to preserve structure
    processed_lines = []
    for line_idx, line in enumerate(lines):
        if not line:
            continue

        sorted_line = sorted(line, key=lambda l: l.xmin)
        line_data = []

        for letter in sorted_line:
            height = letter.ymax - letter.ymin
            width = letter.xmax - letter.xmin

            scaled_height = int(height * content_scale)
            scaled_width = int(width * content_scale)
            scaled_bl = int(letter.bl * content_scale)

            # Extract and scale letter image
            letter_img = original_image[letter.ymin:letter.ymax, letter.xmin:letter.xmax]
            if letter_img.size > 0:
                # For binarized images, extract only the actual letter pixels (not gaps)
                # This prevents horizontally-merged split letters (like ö) from showing gaps
                # Check if this looks like a binarized image (mostly white/black, little gray)
                if len(letter_img.shape) == 3:
                    gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = letter_img

                # If most pixels are near 0 or 255, it's likely binarized
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                total_pixels = gray.shape[0] * gray.shape[1]
                edge_pixels = hist[0:10].sum() + hist[246:256].sum()  # pixels near 0 or 255
                is_binarized = (edge_pixels / total_pixels) > 0.9

                if is_binarized:
                    # Find connected components in this letter region
                    # Invert for findContours (needs white text on black background)
                    inverted = cv2.bitwise_not(gray)

                    # Find contours (connected components)
                    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) > 0:
                        # Create mask with all contours filled (merges split parts)
                        mask = np.zeros_like(gray)
                        cv2.drawContours(mask, contours, -1, 255, -1)  # Fill all contours

                        # Apply mask to keep only letter pixels, set background to white
                        if len(letter_img.shape) == 3:
                            letter_img_masked = letter_img.copy()
                            letter_img_masked[mask == 0] = 255  # White background
                            letter_img = letter_img_masked
                        else:
                            letter_img[mask == 0] = 255

                letter_img = cv2.resize(letter_img, (scaled_width, scaled_height))
            else:
                letter_img = np.ones((scaled_height, scaled_width, 3), dtype=np.uint8) * 255

            line_data.append({
                'img': letter_img,
                'width': scaled_width,
                'height': scaled_height,
                'bl': scaled_bl,
                'original_x': letter.xmin
            })

        if line_data:
            processed_lines.append(line_data)

    if not processed_lines:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Calculate line heights - use actual height of each line for proper spacing
    # Use TIGHTER spacing for TOC (1.2x instead of 1.5x) to match original compact layout
    line_heights = []
    for line_data in processed_lines:
        max_height = max(item['height'] for item in line_data)
        # Apply 1.2x spacing for compact TOC layout
        line_height = int(max_height * 1.2)
        line_heights.append(line_height)

    print(f"  [TOC] Line heights range: {min(line_heights)}-{max(line_heights)}px (using 1.2x spacing)")

    # Calculate total page height
    total_height = top_margin + sum(line_heights) + bottom_margin
    total_height = max(total_height, 800)

    # Create page
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8)
    new_page[:] = background_color

    # Place each line - detect and right-align page numbers
    current_y = top_margin

    # First, find the right alignment position by looking at the rightmost elements
    # In TOC, page numbers are typically the last 1-3 letters on each line
    right_align_x = new_page_width - right_margin - 10  # Default position

    for line_idx, (line_data, line_height) in enumerate(zip(processed_lines, line_heights)):
        if not line_data:
            continue

        # Calculate baseline position for this line
        max_above = max(item['height'] - item['bl'] for item in line_data)
        baseline_y = current_y + max_above

        # Detect page numbers: find large gaps in X positions
        # Page numbers are separated from main text by a gap
        if len(line_data) > 2:
            # Calculate gaps between consecutive letters in ORIGINAL coordinates
            gaps = []
            for i in range(len(line_data) - 1):
                # Get the right edge of current letter in original coordinates
                # original_x is already in original coordinates
                curr_right = line_data[i]['original_x'] + (line_data[i]['width'] / content_scale)
                next_left = line_data[i + 1]['original_x']
                gap = next_left - curr_right
                gaps.append((i, gap))

            # Find the largest gap - this separates text from page number
            if gaps:
                max_gap_idx, max_gap = max(gaps, key=lambda x: x[1])

                # For TOC, use a fixed threshold (18px) because:
                # 1. Gap size is consistent across different font sizes in real TOCs
                # 2. We want to catch gaps of ~20px even in large fonts
                # 3. Scaling with letter width made big fonts harder to split
                gap_threshold = 18

                # If gap is significant, split there
                if max_gap > gap_threshold:
                    text_letters = line_data[:max_gap_idx + 1]
                    number_letters = line_data[max_gap_idx + 1:]
                else:
                    # No significant gap, treat all as text
                    text_letters = line_data
                    number_letters = []
            else:
                text_letters = line_data
                number_letters = []
        else:
            text_letters = line_data
            number_letters = []

        # Place text letters (left-aligned)
        current_x = left_margin
        for item in text_letters:
            # Map original X position to new page (scaled)
            original_rel_x = item['original_x'] - min(all_x)
            scaled_rel_x = int(original_rel_x * content_scale)
            x_pos = left_margin + scaled_rel_x

            # Calculate vertical position based on baseline
            y_pos = baseline_y - (item['height'] - item['bl'])

            # Place letter if it fits
            if (x_pos >= left_margin and
                x_pos + item['width'] <= new_page_width - right_margin and
                y_pos >= 0 and
                y_pos + item['height'] <= total_height):
                try:
                    new_page[y_pos:y_pos + item['height'], x_pos:x_pos + item['width']] = item['img']
                    current_x = max(current_x, x_pos + item['width'])
                except Exception as e:
                    logger.warning(f"Failed to place letter: {e}")

        # Place page number letters (right-aligned)
        if number_letters:
            # Calculate total width of page numbers
            total_number_width = sum(item['width'] for item in number_letters)
            # Start from right alignment position
            x_pos = right_align_x - total_number_width

            for item in number_letters:
                # Calculate vertical position based on baseline
                y_pos = baseline_y - (item['height'] - item['bl'])

                # Place letter if it fits
                if (x_pos >= left_margin and
                    x_pos + item['width'] <= new_page_width - right_margin and
                    y_pos >= 0 and
                    y_pos + item['height'] <= total_height):
                    try:
                        new_page[y_pos:y_pos + item['height'], x_pos:x_pos + item['width']] = item['img']
                        x_pos += item['width']
                    except Exception as e:
                        logger.warning(f"Failed to place page number: {e}")

        current_y += line_height

    # Crop to actual content
    if current_y < total_height:
        new_page = new_page[:current_y, :, :]

    print(f"  [TOC] Created page {new_page.shape[1]}x{new_page.shape[0]}px with right-aligned numbers")

    return new_page

