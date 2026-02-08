#!/usr/bin/env python3
"""
Check what text is actually in the title by looking at the original image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original
img = cv2.imread('../images/jtg_p033.png')
if img is None:
    print("Can't load image")
    exit(1)

# Show the top portion (where title likely is)
top_portion = img[:800, :, :]

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(top_portion, cv2.COLOR_BGR2RGB))
plt.title('Original Image - Top Portion (contains title)')
plt.axis('off')
plt.tight_layout()
plt.savefig('notebooks/original_title_area.png', dpi=150)
print("✓ Saved to notebooks/original_title_area.png")
print("\nPlease check this image to see what text is actually in the title.")
print("It might not be just 'Epilogue' - there could be additional text.")
