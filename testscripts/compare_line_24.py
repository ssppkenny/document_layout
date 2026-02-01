#!/usr/bin/env python3
"""
Compare line 24 from original (line_24_extracted.png) with its reflowed version in output_reflowed.png
"""
import cv2
import numpy as np

# Load both images
original_line = cv2.imread('line_24_extracted.png')
reflowed_page = cv2.imread('output_reflowed.png')

if original_line is None:
    print("ERROR: Could not load line_24_extracted.png")
    exit(1)

if reflowed_page is None:
    print("ERROR: Could not load output_reflowed.png")
    exit(1)

print(f'Original line 24 size: {original_line.shape[1]}x{original_line.shape[0]}')
print(f'Reflowed page size: {reflowed_page.shape[1]}x{reflowed_page.shape[0]}')

# The reflowed output should have this line somewhere
# Let's assume it's around line 24 position in the reflowed page
# Since we reflow, the position might change, but let's look at a reasonable range

# For visualization, let's create a side-by-side comparison
# We'll need to find the actual position in the reflowed page
# For now, let's just extract a region from the reflowed page that should contain this line

# Line 24 should be somewhere in the middle-lower part of the page
# Let's extract a few lines from the reflowed page for manual inspection

# Extract regions from reflowed page at different Y positions to find our line
print('\nSearching for line 24 in reflowed output...')

# The reflowed page likely has this content starting around y=1000-1500
# Let's extract regions every 100 pixels and save them for inspection
for y_start in range(800, min(2000, reflowed_page.shape[0]), 100):
    y_end = min(y_start + 100, reflowed_page.shape[0])
    region = reflowed_page[y_start:y_end, 50:1950].copy()

    # Check if this region has significant content
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    text_pixels = np.sum(gray < 200)

    if text_pixels > 100:  # If there's significant text
        filename = f'reflowed_region_y{y_start}.png'
        cv2.imwrite(filename, region)
        print(f'  Saved region at y={y_start}: {filename}')

print('\n✓ Regions extracted. Please check the images to find line 24.')
print('  Original line 24: line_24_extracted.png')
print('  Reflowed regions: reflowed_region_y*.png')
