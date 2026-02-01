#!/usr/bin/env python3
"""
Create a detailed comparison of out13.png original vs reflowed
Show both images side by side with annotations
"""
import cv2
import numpy as np

# Load images
original = cv2.imread('notebooks/out13.png')
reflowed = cv2.imread('out13_reflowed_test.png')

if original is None or reflowed is None:
    print("ERROR: Could not load images")
    exit(1)

orig_h, orig_w = original.shape[:2]
refl_h, refl_w = reflowed.shape[:2]

print(f'Original: {orig_w}x{orig_h}')
print(f'Reflowed: {refl_w}x{refl_h}')

# Zoom in on the reflowed to see more detail
# Extract just the first 400 pixels height to see the content better
reflowed_crop = reflowed[:min(400, refl_h), :].copy()

# Resize original to match scale (zoom 2.5x)
original_scaled = cv2.resize(original, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

# Create side-by-side comparison
max_height = max(original_scaled.shape[0], reflowed_crop.shape[0])
comparison = np.ones((max_height, original_scaled.shape[1] + reflowed_crop.shape[1] + 20, 3), dtype=np.uint8) * 200

# Place images
comparison[0:original_scaled.shape[0], 0:original_scaled.shape[1]] = original_scaled
comparison[0:reflowed_crop.shape[0], original_scaled.shape[1]+20:] = reflowed_crop

# Add labels
cv2.putText(comparison, 'Original (2.5x)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(comparison, 'Reflowed', (original_scaled.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Save comparison
output_path = 'out13_comparison.png'
cv2.imwrite(output_path, comparison)
print(f'\n✓ Comparison saved to: {output_path}')

# Also save a zoomed-in view of just the last few words from reflowed
# to check if they're clipped
if refl_h > 50:
    # Extract first line from reflowed (around y=50-150)
    line1 = reflowed[50:150, 50:1950].copy()
    cv2.imwrite('out13_reflowed_line1.png', line1)
    print(f'✓ First line saved to: out13_reflowed_line1.png')

    # Extract second line
    line2 = reflowed[150:250, 50:1950].copy()
    cv2.imwrite('out13_reflowed_line2.png', line2)
    print(f'✓ Second line saved to: out13_reflowed_line2.png')
