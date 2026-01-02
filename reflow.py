import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Letter:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    bl: int

def create_page_with_word_wrapping(letters: List[Letter], original_image: np.ndarray, 
                                 zoom_factor: float, new_page_width: int,
                                 left_margin: int = 50, right_margin: int = 50,
                                 top_margin: int = 50, bottom_margin: int = 50,
                                 line_spacing: int = 20, 
                                 preserve_spacing: bool = True) -> np.ndarray:
    """
    Create a new page image with letters reflowed with word wrapping.
    Letters are placed in original order, and new line begins when there's no space.
    
    Args:
        letters: List of Letter objects in the exact order to be placed
        original_image: Source image containing the letters
        zoom_factor: Scaling factor for letters
        new_page_width: Width of the new page image
        left_margin, right_margin, top_margin, bottom_margin: Margins in pixels
        line_spacing: Additional spacing between lines
        preserve_spacing: Whether to preserve original spacing between letters
        
    Returns:
        New page image with inserted letters
    """
    if not letters:
        return np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8) * 255
    
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
        
        letter_data.append({
            'original_idx': idx,
            'letter': letter,
            'region': letter_region,
            'scaled_width': scaled_width,
            'scaled_height': scaled_height
        })
    
    if not letter_data:
        return np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8) * 255
    
    # Group letters into lines based on available width, preserving exact order
    lines = []
    current_line = []
    current_line_width = 0
    
    for i, data in enumerate(letter_data):
        # Calculate space from previous letter if preserving spacing
        space = 0
        if preserve_spacing and i > 0:
            # Get the actual previous letter in the provided order
            prev_data = letter_data[i-1]
            prev_letter = prev_data['letter']
            curr_letter = data['letter']
            
            # Calculate original space between these consecutive letters
            original_space = curr_letter.xmin - prev_letter.xmax
            if original_space > 0:
                space = int(original_space * zoom_factor)
        
        # Check if this letter would overflow the current line
        # (Check if it would go beyond the right margin)
        if current_line_width + space + data['scaled_width'] > available_width and current_line:
            # Start a new line with this letter
            lines.append(current_line)
            current_line = []
            current_line_width = 0
            space = 0  # No space at beginning of new line
        
        # Add to current line
        data_with_space = data.copy()
        data_with_space['space_before'] = space
        current_line.append(data_with_space)
        current_line_width += space + data['scaled_width']
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    # Calculate total height needed
    total_height = top_margin
    for line in lines:
        if not line:
            continue
        # Find max height in this line
        line_height = max(item['scaled_height'] for item in line)
        total_height += line_height + line_spacing
    
    # Remove extra line spacing from last line and add bottom margin
    total_height = total_height - line_spacing + bottom_margin
    
    # Create blank white page
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8) * 255
    
    # Place letters line by line in exact order
    current_y = top_margin
    
    for line_idx, line in enumerate(lines):
        if not line:
            continue
            
        # Calculate max height for this line
        max_line_height = max(item['scaled_height'] for item in line)
        
        # Place letters in this line in the order they appear
        current_x = left_margin
        
        for item in line:
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
        current_y += max_line_height + line_spacing
    
    return new_page

def create_page_with_bounding_boxes_wrapping(letters: List[Letter], zoom_factor: float, 
                                           new_page_width: int,
                                           left_margin: int = 50, right_margin: int = 50,
                                           top_margin: int = 50, bottom_margin: int = 50,
                                           line_spacing: int = 20,
                                           preserve_spacing: bool = True,
                                           box_color=(0, 0, 255), 
                                           baseline_color=(0, 255, 0)) -> np.ndarray:
    """
    Create a visualization with bounding boxes, arranged with word wrapping.
    
    Args:
        letters: List of Letter objects in the exact order to be placed
        zoom_factor: Scaling factor for letters
        new_page_width: Width of the new page image
        margins: Margins in pixels
        line_spacing: Additional spacing between lines
        preserve_spacing: Whether to preserve original spacing between letters
        box_color: Color for bounding boxes (BGR)
        baseline_color: Color for baseline (BGR)
        
    Returns:
        New page image with drawn bounding boxes and baselines
    """
    if not letters:
        return np.ones((top_margin + bottom_margin + 100, new_page_width, 3), dtype=np.uint8) * 255
    
    # Calculate available width for content
    available_width = new_page_width - left_margin - right_margin
    
    # Store letter data in the exact order provided
    letter_data = []
    for idx, letter in enumerate(letters):
        scaled_width = int((letter.xmax - letter.xmin) * zoom_factor)
        scaled_height = int((letter.ymax - letter.ymin) * zoom_factor)
        scaled_bl = int(letter.bl * zoom_factor)
        
        letter_data.append({
            'original_idx': idx,
            'letter': letter,
            'scaled_width': scaled_width,
            'scaled_height': scaled_height,
            'scaled_bl': scaled_bl
        })
    
    # Group letters into lines based on available width, preserving exact order
    lines = []
    current_line = []
    current_line_width = 0
    
    for i, data in enumerate(letter_data):
        # Calculate space from previous letter if preserving spacing
        space = 0
        if preserve_spacing and i > 0:
            prev_data = letter_data[i-1]
            prev_letter = prev_data['letter']
            curr_letter = data['letter']
            
            original_space = curr_letter.xmin - prev_letter.xmax
            if original_space > 0:
                space = int(original_space * zoom_factor)
        
        # Check if this letter would overflow the current line
        if current_line_width + space + data['scaled_width'] > available_width and current_line:
            # Start a new line with this letter
            lines.append(current_line)
            current_line = []
            current_line_width = 0
            space = 0
        
        # Add to current line
        data_with_space = data.copy()
        data_with_space['space_before'] = space
        current_line.append(data_with_space)
        current_line_width += space + data['scaled_width']
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    # Calculate total height needed
    total_height = top_margin
    for line in lines:
        if not line:
            continue
        line_height = max((item['scaled_height'] for item in line), default=0)
        total_height += line_height + line_spacing
    
    # Remove extra line spacing from last line and add bottom margin
    total_height = total_height - line_spacing + bottom_margin
    
    # Create blank white page
    new_page = np.ones((total_height, new_page_width, 3), dtype=np.uint8) * 255
    
    # Draw bounding boxes line by line in exact order
    current_y = top_margin
    
    for line_idx, line in enumerate(lines):
        if not line:
            continue
            
        # Calculate max height for this line
        max_line_height = max(item['scaled_height'] for item in line)
        
        # Place letters in this line in order
        current_x = left_margin
        
        for item in line:
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
                
                # Add original index to verify order
                cv2.putText(new_page, f"{item['original_idx']}", 
                           (x1 + 2, y1 + 12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            current_x += item['scaled_width']
        
        # Move to next line
        current_y += max_line_height + line_spacing
    
    # Draw margin lines for visualization
    cv2.line(new_page, (left_margin, 0), (left_margin, total_height), (200, 200, 200), 1)
    cv2.line(new_page, (new_page_width - right_margin, 0), 
             (new_page_width - right_margin, total_height), (200, 200, 200), 1)
    cv2.line(new_page, (0, top_margin), (new_page_width, top_margin), (200, 200, 200), 1)
    cv2.line(new_page, (0, total_height - bottom_margin), 
             (new_page_width, total_height - bottom_margin), (200, 200, 200), 1)
    
    # Add info text
    info_text = f"Letters: {len(letters)} | Width: {new_page_width} | Zoom: {zoom_factor}"
    cv2.putText(new_page, info_text, (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return new_page

# Test to verify exact order preservation
if __name__ == "__main__":
    # Create test letters in a specific, non-sorted order
    letters = [
        Letter(xmin=100, ymin=90, xmax=120, ymax=115, bl=5),  # Index 0
        Letter(xmin=10, ymin=30, xmax=30, ymax=55, bl=5),     # Index 1
        Letter(xmin=70, ymin=30, xmax=90, ymax=55, bl=5),     # Index 2
        Letter(xmin=40, ymin=60, xmax=60, ymax=85, bl=5),     # Index 3
        Letter(xmin=10, ymin=60, xmax=30, ymax=85, bl=5),     # Index 4
        Letter(xmin=100, ymin=60, xmax=120, ymax=85, bl=5),   # Index 5
        Letter(xmin=40, ymin=30, xmax=60, ymax=55, bl=5),     # Index 6
        Letter(xmin=70, ymin=60, xmax=90, ymax=85, bl=5),     # Index 7
        Letter(xmin=10, ymin=90, xmax=30, ymax=115, bl=5),    # Index 8
        Letter(xmin=40, ymin=90, xmax=60, ymax=115, bl=5),    # Index 9
        Letter(xmin=70, ymin=90, xmax=90, ymax=115, bl=5),    # Index 10
        Letter(xmin=100, ymin=30, xmax=120, ymax=55, bl=5),   # Index 11
    ]
    
    print("Original order (as provided in list):")
    for i, letter in enumerate(letters):
        print(f"  Letter {i}: y={letter.ymin}, x={letter.xmin}")
    
    # Create dummy original image
    original_image = np.ones((130, 130, 3), dtype=np.uint8) * 255
    for idx, letter in enumerate(letters):
        cv2.putText(original_image, str(idx), (letter.xmin, letter.ymin + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Test with narrow page to force reflow
    zoom_factor = 1.2
    new_page_width = 150  # Very narrow to force many line breaks
    
    # Create reflowed page
    page_reflowed = create_page_with_word_wrapping(
        letters, original_image, zoom_factor, new_page_width,
        left_margin=10, top_margin=10, right_margin=10, bottom_margin=10,
        line_spacing=10, preserve_spacing=True
    )
    
    # Create visualization with bounding boxes (showing indices)
    page_boxes = create_page_with_bounding_boxes_wrapping(
        letters, zoom_factor, new_page_width,
        left_margin=10, top_margin=10, right_margin=10, bottom_margin=10,
        line_spacing=10, preserve_spacing=True
    )
    
    print(f"\nNew page width: {new_page_width}")
    print(f"Zoom factor: {zoom_factor}")
    print(f"Reflowed page dimensions: {page_reflowed.shape}")
    print("\nCheck the bounding box image - indices should be in order 0, 1, 2, 3, ...")
    
    # Display
    cv2.imshow("Reflowed with Exact Order Preservation", page_reflowed)
    cv2.imshow("Bounding Boxes with Indices (should be 0,1,2,3...)", page_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save for reference
    cv2.imwrite("reflowed_exact_order.jpg", page_reflowed)
    cv2.imwrite("boxes_exact_order.jpg", page_boxes)
