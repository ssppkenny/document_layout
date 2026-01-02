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

def detect_paragraphs_and_spacing(letters: List[Letter], original_height: int) -> Tuple[List[int], float]:
    """
    Detect paragraph breaks and calculate average line spacing.
    
    Args:
        letters: List of Letter objects
        original_height: Height of the original image
        
    Returns:
        Tuple of (list of paragraph start indices, average line spacing in original)
    """
    if not letters:
        return [], 0.0
    
    # First, group letters into lines in the original image
    lines = []
    
    # Sort by y to group into lines
    sorted_letters = sorted(letters, key=lambda l: (l.ymin, l.xmin))
    
    # Group into lines with y_threshold
    y_threshold = original_height * 0.02  # 2% of image height
    
    if not sorted_letters:
        return [0], 0.0
    
    current_line = [sorted_letters[0]]
    for i in range(1, len(sorted_letters)):
        if abs(sorted_letters[i].ymin - current_line[0].ymin) <= y_threshold:
            current_line.append(sorted_letters[i])
        else:
            # Sort line by x
            current_line.sort(key=lambda l: l.xmin)
            lines.append(current_line)
            current_line = [sorted_letters[i]]
    
    if current_line:
        current_line.sort(key=lambda l: l.xmin)
        lines.append(current_line)
    
    # Calculate average line spacing
    avg_line_spacing = 0.0
    if len(lines) > 1:
        spacings = []
        for i in range(1, len(lines)):
            # Find min y of current line and max y of previous line
            min_y_current = min(letter.ymin for letter in lines[i])
            max_y_prev = max(letter.ymax for letter in lines[i-1])
            spacing = min_y_current - max_y_prev
            if spacing > 0:
                spacings.append(spacing)
        
        if spacings:
            avg_line_spacing = sum(spacings) / len(spacings)
    
    # Build mapping from letters to their indices in the original list
    letter_to_idx = {id(letter): idx for idx, letter in enumerate(letters)}
    
    # Detect paragraph breaks (where spacing is significantly larger than average)
    paragraph_starts = [0]  # First line always starts a paragraph
    if len(lines) > 1 and avg_line_spacing > 0:
        for i in range(1, len(lines)):
            min_y_current = min(letter.ymin for letter in lines[i])
            max_y_prev = max(letter.ymax for letter in lines[i-1])
            spacing = min_y_current - max_y_prev
            
            # If spacing is more than 2x average, it's likely a paragraph break
            if spacing > avg_line_spacing * 2.0:
                # Get the first letter index in this line
                first_letter_idx = letter_to_idx[id(lines[i][0])]
                paragraph_starts.append(first_letter_idx)
    
    return paragraph_starts, avg_line_spacing

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

def create_page_with_word_wrapping(letters: List[Letter], original_image: np.ndarray, 
                                 zoom_factor: float, new_page_width: int,
                                 left_margin: int = 50, right_margin: int = 50,
                                 top_margin: int = 50, bottom_margin: int = 50,
                                 line_spacing: int = 20, 
                                 paragraph_spacing_factor: float = 2.0,
                                 preserve_spacing: bool = True) -> np.ndarray:
    """
    Create a new page image with letters reflowed with word wrapping.
    Letters are placed in original order, and new line begins when there's no space.
    Paragraph breaks are preserved with extra spacing.
    
    Args:
        letters: List of Letter objects in the exact order to be placed
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
    if not letters:
        return np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8) * 255
    
    # Detect paragraph breaks in original layout
    paragraph_starts, avg_original_spacing = detect_paragraphs_and_spacing(letters, original_image.shape[0])
    paragraph_spacing = int(line_spacing * paragraph_spacing_factor)
    
    print(f"Detected {len(paragraph_starts)} paragraphs")
    print(f"Paragraph starts at indices: {paragraph_starts}")
    
    # Calculate available width for content
    available_width = new_page_width - left_margin - right_margin
    
    # Store letter data in the exact order provided
    letter_data = []
    for idx, letter in enumerate(letters):
        letter_region = original_image[letter.ymin:letter.ymax, letter.xmin:letter.xmax]
        if letter_region.size == 0:
            continue
            
        scaled_width = int((letter.xmax - letter.xmin) * zoom_factor)
        scaled_height = int((letter.ymax - letter.ymin) * zoom_factor)
        
        # Determine which paragraph this letter belongs to
        paragraph_idx = is_letter_in_paragraph(idx, paragraph_starts, len(letters))
        
        letter_data.append({
            'original_idx': idx,
            'letter': letter,
            'region': letter_region,
            'scaled_width': scaled_width,
            'scaled_height': scaled_height,
            'paragraph_idx': paragraph_idx,
            'is_paragraph_start': idx in paragraph_starts
        })
    
    if not letter_data:
        return np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8) * 255
    
    # Group letters into lines based on available width, preserving exact order
    lines = []
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
            lines.append({
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
            lines.append({
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
    
    # Add the last line
    if current_line:
        lines.append({
            'letters': current_line,
            'paragraph_idx': current_paragraph_idx,
            'is_paragraph_start': current_line_paragraph_start
        })
    
    # Calculate total height needed
    total_height = top_margin
    previous_paragraph_idx = -1
    
    for line in lines:
        if not line['letters']:
            continue
        
        # Find max height in this line
        line_height = max(item['scaled_height'] for item in line['letters'])
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            total_height += paragraph_spacing
        
        total_height += line_height
        previous_paragraph_idx = line['paragraph_idx']
        
        # Add line spacing (except after last line)
        if line != lines[-1]:
            next_line = lines[lines.index(line) + 1]
            # Only add line spacing if next line is in same paragraph
            if next_line['paragraph_idx'] == line['paragraph_idx']:
                total_height += line_spacing
    
    # Add bottom margin
    total_height += bottom_margin
    
    # Create blank white page
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8) * 255
    
    # Place letters line by line in exact order
    current_y = top_margin
    previous_paragraph_idx = -1
    
    for line_idx, line in enumerate(lines):
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
        
        # Calculate max height for this line
        max_line_height = max(item['scaled_height'] for item in line['letters'])
        
        # Place letters in this line in the order they appear
        current_x = left_margin
        
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
            
            # Calculate vertical position (center in line)
            y_offset = current_y + (max_line_height - item['scaled_height']) // 2
            
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
        
        # Move to next line
        current_y += max_line_height
        
        # Add line spacing if not last line and next line is in same paragraph
        if line_idx < len(lines) - 1:
            next_line = lines[line_idx + 1]
            if next_line['paragraph_idx'] == line['paragraph_idx']:
                current_y += line_spacing
        
        previous_paragraph_idx = line['paragraph_idx']
    
    return new_page

def create_page_with_bounding_boxes_wrapping(letters: List[Letter], original_image: np.ndarray,
                                           zoom_factor: float, new_page_width: int,
                                           left_margin: int = 50, right_margin: int = 50,
                                           top_margin: int = 50, bottom_margin: int = 50,
                                           line_spacing: int = 20,
                                           paragraph_spacing_factor: float = 2.0,
                                           preserve_spacing: bool = True,
                                           box_color=(0, 0, 255), 
                                           baseline_color=(0, 255, 0),
                                           paragraph_color=(255, 0, 0)) -> np.ndarray:
    """
    Create a visualization with bounding boxes, arranged with word wrapping.
    
    Args:
        letters: List of Letter objects in the exact order to be placed
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
    if not letters:
        return np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8) * 255
    
    # Detect paragraph breaks in original layout
    paragraph_starts, avg_original_spacing = detect_paragraphs_and_spacing(letters, original_image.shape[0])
    paragraph_spacing = int(line_spacing * paragraph_spacing_factor)
    
    print(f"Detected {len(paragraph_starts)} paragraphs")
    print(f"Paragraph starts at indices: {paragraph_starts}")
    
    # Calculate available width for content
    available_width = new_page_width - left_margin - right_margin
    
    # Store letter data in the exact order provided
    letter_data = []
    for idx, letter in enumerate(letters):
        scaled_width = int((letter.xmax - letter.xmin) * zoom_factor)
        scaled_height = int((letter.ymax - letter.ymin) * zoom_factor)
        scaled_bl = int(letter.bl * zoom_factor)
        
        # Determine which paragraph this letter belongs to
        paragraph_idx = is_letter_in_paragraph(idx, paragraph_starts, len(letters))
        
        letter_data.append({
            'original_idx': idx,
            'letter': letter,
            'scaled_width': scaled_width,
            'scaled_height': scaled_height,
            'scaled_bl': scaled_bl,
            'paragraph_idx': paragraph_idx,
            'is_paragraph_start': idx in paragraph_starts
        })
    
    # Group letters into lines based on available width, preserving exact order
    lines = []
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
            lines.append({
                'letters': current_line,
                'paragraph_idx': current_paragraph_idx,
                'is_paragraph_start': current_line_paragraph_start
            })
            current_line = []
            current_line_width = 0
            current_line_paragraph_start = False
            space = 0
        
        # If this is a new paragraph and we're not at the beginning of a line,
        # force a new line
        if is_new_paragraph and current_line:
            lines.append({
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
    
    # Add the last line
    if current_line:
        lines.append({
            'letters': current_line,
            'paragraph_idx': current_paragraph_idx,
            'is_paragraph_start': current_line_paragraph_start
        })
    
    # Calculate total height needed
    total_height = top_margin
    previous_paragraph_idx = -1
    
    for line in lines:
        if not line['letters']:
            continue
        
        line_height = max((item['scaled_height'] for item in line['letters']), default=0)
        
        # Add paragraph spacing if this line starts a new paragraph (not first paragraph)
        if line['is_paragraph_start'] and previous_paragraph_idx != -1:
            total_height += paragraph_spacing
        
        total_height += line_height
        previous_paragraph_idx = line['paragraph_idx']
        
        # Add line spacing (except after last line)
        if line != lines[-1]:
            next_line = lines[lines.index(line) + 1]
            # Only add line spacing if next line is in same paragraph
            if next_line['paragraph_idx'] == line['paragraph_idx']:
                total_height += line_spacing
    
    # Add bottom margin
    total_height += bottom_margin
    
    # Create blank white page
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8) * 255
    
    # Draw bounding boxes line by line in exact order
    current_y = top_margin
    previous_paragraph_idx = -1
    
    for line_idx, line in enumerate(lines):
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
            marker_text = f"Paragraph {line['paragraph_idx'] + 1} Start"
            cv2.putText(new_page, marker_text, 
                       (left_margin, current_y - paragraph_spacing // 2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, paragraph_color, 1)
        
        # Calculate max height for this line
        max_line_height = max(item['scaled_height'] for item in line['letters'])
        
        # Place letters in this line in order
        current_x = left_margin
        
        for item in line['letters']:
            # Add space before letter if not first in line
            if current_x > left_margin:
                current_x += item['space_before']
            
            # Calculate vertical position (center in line)
            y_offset = current_y + (max_line_height - item['scaled_height']) // 2
            
            # Draw bounding box
            x1 = current_x
            y1 = y_offset
            x2 = current_x + item['scaled_width']
            y2 = y_offset + item['scaled_height']
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(new_page, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw baseline
                baseline_y = y2 - item['scaled_bl']
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
        
        # Add line spacing if not last line and next line is in same paragraph
        if line_idx < len(lines) - 1:
            next_line = lines[line_idx + 1]
            if next_line['paragraph_idx'] == line['paragraph_idx']:
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
    info_text = f"Letters: {len(letters)} | Paragraphs: {len(paragraph_starts)} | Width: {new_page_width}"
    cv2.putText(new_page, info_text, (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return new_page

# Test with clear paragraphs
if __name__ == "__main__":
    # Create test letters with two clear paragraphs
    letters = []
    
    # Paragraph 1: 3 lines
    # Line 1 of paragraph 1 (y=30)
    letters.append(Letter(xmin=10, ymin=30, xmax=30, ymax=55, bl=5))   # Index 0
    letters.append(Letter(xmin=40, ymin=30, xmax=60, ymax=55, bl=5))   # Index 1
    letters.append(Letter(xmin=70, ymin=30, xmax=90, ymax=55, bl=5))   # Index 2
    
    # Line 2 of paragraph 1 (y=65 - small gap)
    letters.append(Letter(xmin=10, ymin=65, xmax=30, ymax=90, bl=5))   # Index 3
    letters.append(Letter(xmin=40, ymin=65, xmax=60, ymax=90, bl=5))   # Index 4
    
    # Line 3 of paragraph 1 (y=100 - small gap)
    letters.append(Letter(xmin=10, ymin=100, xmax=30, ymax=125, bl=5)) # Index 5
    
    # LARGE GAP - paragraph break (from y=125 to y=180 is 55 pixels)
    
    # Paragraph 2: 2 lines
    # Line 1 of paragraph 2 (y=180 - large gap)
    letters.append(Letter(xmin=10, ymin=180, xmax=30, ymax=205, bl=5)) # Index 6 - Paragraph start!
    letters.append(Letter(xmin=40, ymin=180, xmax=60, ymax=205, bl=5)) # Index 7
    
    # Line 2 of paragraph 2 (y=215 - small gap)
    letters.append(Letter(xmin=10, ymin=215, xmax=30, ymax=240, bl=5)) # Index 8
    letters.append(Letter(xmin=40, ymin=215, xmax=60, ymax=240, bl=5)) # Index 9
    
    # Create original image
    original_image = np.ones((250, 100, 3), dtype=np.uint8) * 255
    
    # Put text for visualization
    cv2.putText(original_image, "P1L1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(original_image, "P1L2", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(original_image, "P1L3", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Large gap here
    
    cv2.putText(original_image, "P2L1", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(original_image, "P2L2", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Test with narrow page to force reflow
    zoom_factor = 1.5
    new_page_width = 200
    
    print("Testing paragraph preservation...")
    print(f"Total letters: {len(letters)}")
    print(f"Expected: Paragraph 1 (letters 0-5), Paragraph 2 (letters 6-9)")
    
    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        letters, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )
    
    # Create visualization with bounding boxes
    page_boxes = create_page_with_bounding_boxes_wrapping(
        letters, original_image, zoom_factor, new_page_width,
        left_margin=20, top_margin=20, right_margin=20, bottom_margin=20,
        line_spacing=15, paragraph_spacing_factor=2.0, preserve_spacing=True
    )
    
    # Display
    cv2.imshow("Original with paragraphs", original_image)
    cv2.imshow("Reflowed with paragraph spacing", page_reflowed)
    cv2.imshow("Bounding Boxes with paragraph markers", page_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save for reference
    cv2.imwrite("reflowed_with_paragraphs.jpg", page_reflowed)
    cv2.imwrite("boxes_with_paragraphs.jpg", page_boxes)
    
    print("\nCheck the output images:")
    print("1. 'Reflowed with paragraph spacing' - should show text with extra space between paragraphs")
    print("2. 'Bounding Boxes with paragraph markers' - should show paragraph breaks with red lines and markers")
