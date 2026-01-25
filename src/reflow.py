import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

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
                                 background_color: tuple = (220, 220, 220)) -> np.ndarray:
    """
    Create a new page image with letters reflowed with word wrapping.
    Letters are placed in original order, and new line begins when there's no space.
    Paragraph breaks are preserved with extra spacing.
    
    Args:
        lines: List of lines, where each line is a list of Letter objects in the exact order to be placed
        original_image: Source image containing the letters
        zoom_factor: Scaling factor for letters
        new_page_width: Width of the new page image
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
    all_letters = []
    for line in lines:
        # Sort letters in each line by x position to get reading order
        sorted_line = sorted(line, key=lambda l: l.xmin)
        all_letters.extend(sorted_line)
    
    if not all_letters:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Detect paragraph breaks by horizontal indentation using the original lines
    paragraph_starts, avg_first_xmin = detect_paragraphs_and_spacing_from_lines(lines, original_image.shape[1])
    paragraph_spacing = int(line_spacing * paragraph_spacing_factor)
    
    print(f"Detected {len(paragraph_starts)} paragraphs")
    print(f"Paragraph starts at indices: {paragraph_starts}")
    print(f"Average first letter xmin: {avg_first_xmin}")
    
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
                    print(f"Paragraph {para_idx} starts at line {line_idx} with indentation (xmin): {indentation}")
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
        print(f"Average line width: {avg_line_width}, Short line threshold: {short_line_threshold}")
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
                print(f"Line {line_idx} is short (width: {line_width}), marking letter {last_letter_idx} as end of short line")

            cumulative_idx += len(sorted_line)

    # Store letter data in the exact order (as flattened from lines)
    letter_data = []
    for idx, letter in enumerate(all_letters):
        letter_region = original_image[letter.ymin:letter.ymax, letter.xmin:letter.xmax]
        if letter_region.size == 0:
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
            'is_end_of_short_line': idx in short_line_ends
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
    
    for i, data in enumerate(letter_data):
        # Check if this is start of a new paragraph
        is_new_paragraph = data['paragraph_idx'] != current_paragraph_idx
        
        # Calculate space from previous letter if preserving spacing
        space = 0
        if preserve_spacing and i > 0 and not is_new_paragraph:
            # Get the actual previous letter in the provided order
            prev_data = letter_data[i-1]
            prev_letter = prev_data['letter']
            curr_letter = data['letter']
            
            # Calculate original space between these consecutive letters
            original_space = curr_letter.xmin - prev_letter.xmax
            if original_space > 0:
                space = int(original_space * zoom_factor)
        
        # Check if this letter would overflow the current line
        if current_line_width + space + data['scaled_width'] > available_width and current_line:
            # Start a new line with this letter
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False
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
            space = 0
        
        # Update current paragraph
        if is_new_paragraph:
            current_paragraph_idx = data['paragraph_idx']
            current_line_paragraph_start = data['is_paragraph_start']
        
        # Add to current line
        data_with_space = data.copy()
        data_with_space['space_before'] = space
        current_line.append(data_with_space)
        current_line_width += space + data['scaled_width']
        
        # If this is the first letter in the line and it's a paragraph start,
        # mark the line as starting a paragraph
        if len(current_line) == 1 and data['is_paragraph_start']:
            current_line_paragraph_start = True

        # Check if this letter is at the end of a short line in the original
        # If so, force a new line after it (unless it's the last letter)
        if data['is_end_of_short_line'] and i < len(letter_data) - 1:
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False

    # Add the last line
    if current_line:
        lines_on_new_page.append({
            'letters': current_line,
            'paragraph_idx': current_paragraph_idx,
            'is_paragraph_start': current_line_paragraph_start
        })
    
    # Calculate line heights and baselines for consistent spacing
    # For equal line spacing, we need to define a fixed line height based on the max height across all lines
    max_height_any_line = max(
        (max(item['scaled_height'] for item in line['letters']) if line['letters'] else 0)
        for line in lines_on_new_page
    )

    # Also calculate the maximum space needed above baseline across all lines
    # This is the maximum of (scaled_height - scaled_bl) for all letters
    max_above_baseline = max(
        (max(item['scaled_height'] - item['scaled_bl'] for item in line['letters']) if line['letters'] else 0)
        for line in lines_on_new_page
    )

    # And the maximum space needed below baseline across all lines
    # This is the maximum of scaled_bl for all letters
    max_below_baseline = max(
        (max(item['scaled_bl'] for item in line['letters']) if line['letters'] else 0)
        for line in lines_on_new_page
    )

    # Fixed line height should accommodate both the space above and below the baseline
    fixed_line_height = max_above_baseline + max_below_baseline + line_spacing

    # Calculate total height needed with equal line spacing
    total_height = top_margin
    previous_paragraph_idx = -1
    
    for line_idx, line in enumerate(lines_on_new_page):
        if not line['letters']:
            continue
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            total_height += paragraph_spacing
        
        # Use fixed line height for equal spacing
        total_height += fixed_line_height
        previous_paragraph_idx = line['paragraph_idx']

    # Add bottom margin
    total_height += bottom_margin
    
    # Create blank page with detected background color
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8)
    new_page[:] = background_color

    # Place letters line by line in exact order
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
            print(f"Applying book-style indentation {book_indent} to paragraph {line['paragraph_idx']}")

        for item in line['letters']:
            # Add space before letter if not first in line
            if current_x > left_margin:
                current_x += item['space_before']
            
            # Resize letter
            if item['scaled_width'] > 0 and item['scaled_height'] > 0:
                resized_letter = cv2.resize(item['region'], 
                                          (item['scaled_width'], item['scaled_height']))
            else:
                continue
            
            # Calculate vertical position: place letter so its baseline aligns with the line's baseline
            # The baseline in a letter is at: bottom - bl = ymin + (height - bl)
            # So the top of letter should be at: baseline_y - (scaled_height - scaled_bl)
            y_offset = baseline_y - item['scaled_height'] + item['scaled_bl']

            # Ensure coordinates are within bounds
            y_start = max(0, y_offset)
            y_end = min(y_offset + item['scaled_height'], total_height)
            x_start = current_x
            x_end = min(current_x + item['scaled_width'], new_page_width)
            
            # Place letter if it fits
            if x_end > x_start and y_end > y_start:
                # Adjust if letter would go out of bounds
                if (y_end - y_start) != item['scaled_height'] or (x_end - x_start) != item['scaled_width']:
                    # Crop the resized letter to fit
                    crop_height = y_end - y_start
                    crop_width = x_end - x_start
                    resized_letter = resized_letter[:crop_height, :crop_width]
                
                new_page[y_start:y_end, x_start:x_end] = resized_letter
            
            current_x += item['scaled_width']
        
        # Move to next line with fixed line height
        current_y += fixed_line_height

        previous_paragraph_idx = line['paragraph_idx']
    
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
                                           background_color: tuple = (220, 220, 220)) -> np.ndarray:
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
        
    Returns:
        New page image with drawn bounding boxes and baselines
    """
    if not lines:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Flatten the lines to get all letters in order
    all_letters = []
    for line in lines:
        # Sort letters in each line by x position to get reading order
        sorted_line = sorted(line, key=lambda l: l.xmin)
        all_letters.extend(sorted_line)
    
    if not all_letters:
        page = np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # Detect paragraph breaks by horizontal indentation using the original lines
    paragraph_starts, avg_first_xmin = detect_paragraphs_and_spacing_from_lines(lines, original_image.shape[1])
    paragraph_spacing = int(line_spacing * paragraph_spacing_factor)
    
    print(f"Detected {len(paragraph_starts)} paragraphs")
    print(f"Paragraph starts at indices: {paragraph_starts}")
    print(f"Average first letter xmin: {avg_first_xmin}")
    
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
                    print(f"Paragraph {para_idx} starts at line {line_idx} with indentation (xmin): {indentation}")
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
            'is_end_of_short_line': idx in short_line_ends
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
    
    for i, data in enumerate(letter_data):
        # Check if this is start of a new paragraph
        is_new_paragraph = data['paragraph_idx'] != current_paragraph_idx
        
        # Calculate space from previous letter if preserving spacing
        space = 0
        if preserve_spacing and i > 0 and not is_new_paragraph:
            prev_data = letter_data[i-1]
            prev_letter = prev_data['letter']
            curr_letter = data['letter']
            
            original_space = curr_letter.xmin - prev_letter.xmax
            if original_space > 0:
                space = int(original_space * zoom_factor)
        
        # Check if this letter would overflow the current line
        if current_line_width + space + data['scaled_width'] > available_width and current_line:
            # Start a new line with this letter
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False
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
            space = 0
        
        # Update current paragraph
        if is_new_paragraph:
            current_paragraph_idx = data['paragraph_idx']
            current_line_paragraph_start = data['is_paragraph_start']
        
        # Add to current line
        data_with_space = data.copy()
        data_with_space['space_before'] = space
        current_line.append(data_with_space)
        current_line_width += space + data['scaled_width']
        
        # If this is the first letter in the line and it's a paragraph start,
        # mark the line as starting a paragraph
        if len(current_line) == 1 and data['is_paragraph_start']:
            current_line_paragraph_start = True

        # Check if this letter is at the end of a short line in the original
        # If so, force a new line after it (unless it's the last letter)
        if data['is_end_of_short_line'] and i < len(letter_data) - 1:
            lines_on_new_page.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False

    # Add the last line
    if current_line:
        lines_on_new_page.append({
            'letters': current_line,
            'paragraph_idx': current_paragraph_idx,
            'is_paragraph_start': current_line_paragraph_start
        })
    
    # Calculate total height needed
    total_height = top_margin
    previous_paragraph_idx = -1
    
    for line in lines_on_new_page:
        if not line['letters']:
            continue
        
        line_height = max((item['scaled_height'] for item in line['letters']), default=0)
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            total_height += paragraph_spacing
        
        total_height += line_height
        previous_paragraph_idx = line['paragraph_idx']
        
        # Add line spacing (except after last line)
        if line != lines_on_new_page[-1]:
            total_height += line_spacing

    # Add bottom margin
    total_height += bottom_margin
    
    # Create blank page with detected background color
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8)
    new_page[:] = background_color

    # Draw bounding boxes line by line in exact order
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
            print(f"Applying book-style indentation {book_indent} to paragraph {line['paragraph_idx']} in visualization")

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
        
        # Move to next line
        current_y += max_line_height
        
        # Add line spacing
        if line_idx < len(lines_on_new_page) - 1:
            current_y += line_spacing

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
    
    print("Testing paragraph detection by horizontal indentation with lines input...")
    print(f"Total lines: {len(lines)}")
    print("Expected: Paragraph 1 (lines 0-2), Paragraph 2 (lines 3-4), Paragraph 3 (line 5)")
    print("Paragraph 2 should be indented (starts at x=50 vs x=10)")
    
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
