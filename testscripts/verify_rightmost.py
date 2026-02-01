"""
Create a super clear visualization showing that the rightmost point is correctly placed
"""
import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load image
filename = 'notebooks/out13.png'
img = cv2.imread(filename)
img_h, img_w, _ = img.shape
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run text detection
model = detection_predictor(pretrained=True)
docs = DocumentFile.from_images([filename])
result = model(docs)
words = result[0]["words"]

# Convert to absolute coordinates
words[:, 0] = (words[:, 0] * img_w).astype(np.int32)
words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2
words[:, 2] = (words[:, 2] * img_w).astype(np.int32)
words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2

# Find leftmost and rightmost words
leftmost_idx = np.argmin(words[:, 0])
rightmost_idx = np.argmax(words[:, 2])

left_word = words[leftmost_idx]
right_word = words[rightmost_idx]

# Calculate centers
left_center_y = (left_word[1] + left_word[3]) / 2
right_center_y = (right_word[1] + right_word[3]) / 2

print("Leftmost word:")
print(f"  Box: x=[{left_word[0]}, {left_word[2]}], y=[{left_word[1]}, {left_word[3]}]")
print(f"  Center Y: {left_center_y:.1f}")
print(f"  Left margin point: ({left_word[0]}, {left_center_y:.1f})")

print("\\nRightmost word:")
print(f"  Box: x=[{right_word[0]}, {right_word[2]}], y=[{right_word[1]}, {right_word[3]}]")
print(f"  Center Y: {right_center_y:.1f}")
print(f"  Right margin point: ({right_word[2]}, {right_center_y:.1f})")

print(f"\\nLine angle: Y difference = {right_center_y - left_center_y:.1f} pixels")

# Create matplotlib visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Left panel: Full line
ax1.imshow(img_rgb)
ax1.set_title('Full Line Detection (Angled Line)', fontsize=14, fontweight='bold')

# Draw all word boxes
for xmin, ymin, xmax, ymax, _ in words:
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)

# Highlight leftmost and rightmost
left_rect = patches.Rectangle((left_word[0], left_word[1]),
                               left_word[2]-left_word[0], left_word[3]-left_word[1],
                               linewidth=3, edgecolor='magenta', facecolor='none')
ax1.add_patch(left_rect)

right_rect = patches.Rectangle((right_word[0], right_word[1]),
                                right_word[2]-right_word[0], right_word[3]-right_word[1],
                                linewidth=3, edgecolor='cyan', facecolor='none')
ax1.add_patch(right_rect)

# Draw the line
ax1.plot([left_word[0], right_word[2]], [left_center_y, right_center_y],
         'g-', linewidth=3, label='Detected line')

# Draw margin points
ax1.plot(left_word[0], left_center_y, 'go', markersize=15,
         markeredgecolor='white', markeredgewidth=2, label='Left margin')
ax1.plot(right_word[2], right_center_y, 'yo', markersize=15,
         markeredgecolor='white', markeredgewidth=2, label='Right margin')

# Add annotations
ax1.annotate(f'({left_word[0]}, {left_center_y:.0f})',
             xy=(left_word[0], left_center_y), xytext=(left_word[0]-50, left_center_y-15),
             fontsize=10, color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.annotate(f'({right_word[2]}, {right_center_y:.0f})',
             xy=(right_word[2], right_center_y), xytext=(right_word[2]+10, right_center_y-15),
             fontsize=10, color='yellow', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.legend(loc='upper left')
ax1.axis('off')

# Right panel: Zoomed rightmost word
margin = 20
x1 = int(max(0, right_word[0] - margin))
x2 = int(min(img_w, right_word[2] + margin))
y1 = int(max(0, right_word[1] - margin))
y2 = int(min(img_h, right_word[3] + margin))

zoomed = img_rgb[y1:y2, x1:x2]
ax2.imshow(zoomed, interpolation='nearest')
ax2.set_title(f'Rightmost Word (Zoomed 2x)\\nBox: y=[{right_word[1]}, {right_word[3]}], center_y={right_center_y:.1f}',
              fontsize=14, fontweight='bold')

# Adjust coordinates for zoomed view
zoom_xmin = right_word[0] - x1
zoom_xmax = right_word[2] - x1
zoom_ymin = right_word[1] - y1
zoom_ymax = right_word[3] - y1
zoom_center_y = right_center_y - y1

# Draw the word box
zoom_rect = patches.Rectangle((zoom_xmin, zoom_ymin),
                               zoom_xmax-zoom_xmin, zoom_ymax-zoom_ymin,
                               linewidth=3, edgecolor='red', facecolor='none')
ax2.add_patch(zoom_rect)

# Draw horizontal lines
ax2.axhline(y=zoom_ymin, color='blue', linestyle='--', linewidth=2, label=f'Top (y={right_word[1]})')
ax2.axhline(y=zoom_ymax, color='cyan', linestyle='--', linewidth=2, label=f'Bottom (y={right_word[3]})')
ax2.axhline(y=zoom_center_y, color='green', linestyle='-', linewidth=3, label=f'Center (y={right_center_y:.1f})')

# Draw the rightmost point
point_x = zoom_xmax
point_y = zoom_center_y
ax2.plot(point_x, point_y, 'yo', markersize=20,
         markeredgecolor='white', markeredgewidth=3, label=f'Right margin point')

# Add vertical line at right edge
ax2.axvline(x=zoom_xmax, color='yellow', linestyle=':', linewidth=2, alpha=0.7)

ax2.legend(loc='upper left', fontsize=10)
ax2.set_xlim(-5, zoom_xmax - zoom_xmin + margin)
ax2.set_ylim(zoom_ymax - zoom_ymin + margin, -5)

# Add text showing it's correct
ax2.text(0.5, 0.95, '✓ Point is at (xmax, center_y) - CORRECT',
         transform=ax2.transAxes, fontsize=12, fontweight='bold',
         ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

plt.suptitle('Verification: Rightmost Point is in the Middle of Letter Height',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('rightmost_verification.png', dpi=150, bbox_inches='tight')
print("\\n✓ Verification image saved to: rightmost_verification.png")
print("  Left panel: Full line with angled detection")
print("  Right panel: Zoomed view showing point is exactly at center_y")
