#!/usr/bin/env python3
"""
Final check: Compare the last words from out13.png in reflowed output
"""
import cv2
import numpy as np

print("Loading images...")
original = cv2.imread('notebooks/out13.png')
reflowed_line = cv2.imread('out13_reflowed_line1.png')

if original is None or reflowed_line is None:
    print("ERROR: Could not load images")
    exit(1)

# Scale original to match
original_scaled = cv2.resize(original, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

# Extract just the text region from both (remove margins)
# Original scaled should be around 1750x105
# Reflowed line is 1900x100

# Create side-by-side comparison focusing on the last few words
# Take rightmost 600 pixels from each

orig_crop = original_scaled[:, -600:] if original_scaled.shape[1] > 600 else original_scaled
refl_crop = reflowed_line[:, -600:] if reflowed_line.shape[1] > 600 else reflowed_line

# Make them same height for comparison
max_h = max(orig_crop.shape[0], refl_crop.shape[0])

orig_padded = np.ones((max_h, orig_crop.shape[1], 3), dtype=np.uint8) * 220
refl_padded = np.ones((max_h, refl_crop.shape[1], 3), dtype=np.uint8) * 220

orig_padded[:orig_crop.shape[0], :] = orig_crop
refl_padded[:refl_crop.shape[0], :] = refl_crop

# Combine side by side
comparison = np.hstack([orig_padded, np.ones((max_h, 20, 3), dtype=np.uint8) * 200, refl_padded])

# Add labels
cv2.putText(comparison, 'Original (last words)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(comparison, 'Reflowed (last words)', (orig_padded.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

output_path = 'final_comparison_last_words.png'
cv2.imwrite(output_path, comparison)

print(f'\n✓ Final comparison saved to: {output_path}')
print('\nPlease check if the last words in the reflowed output are:')
print('  1. Not clipped (all pixels visible)')
print('  2. Properly scaled')
print('  3. Baseline-aligned')
