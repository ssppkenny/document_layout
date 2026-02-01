"""
Create a detailed visualization showing the rightmost letter and the detected point
"""
import cv2
import numpy as np
from doctr.models import detection_predictor
from doctr.io import DocumentFile

# Load image
filename = 'notebooks/out13.png'
img = cv2.imread(filename)
img_h, img_w, _ = img.shape

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

# Find rightmost word
rightmost_idx = np.argmax(words[:, 2])
xmin, ymin, xmax, ymax, conf = words[rightmost_idx]

print(f"Rightmost word:")
print(f"  Bounding box: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
print(f"  Height: {ymax - ymin}")
print(f"  Center Y: {(ymin + ymax) / 2}")

# Create enlarged visualization
scale = 5
enlarged = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

# Scale coordinates
xmin_s = int(xmin * scale)
xmax_s = int(xmax * scale)
ymin_s = int(ymin * scale)
ymax_s = int(ymax * scale)

# Calculate center
center_y = (ymin + ymax) / 2
center_y_s = int(center_y * scale)

# Draw the word box in RED
cv2.rectangle(enlarged, (xmin_s, ymin_s), (xmax_s, ymax_s), (0, 0, 255), 3)

# Draw horizontal lines at ymin, ymax, and center
cv2.line(enlarged, (xmin_s, ymin_s), (xmax_s, ymin_s), (255, 0, 0), 2)  # Top - Blue
cv2.line(enlarged, (xmin_s, ymax_s), (xmax_s, ymax_s), (255, 255, 0), 2)  # Bottom - Cyan
cv2.line(enlarged, (xmin_s, center_y_s), (xmax_s, center_y_s), (0, 255, 0), 3)  # Center - Green

# Draw the rightmost point (should be at xmax, center_y)
point_x = xmax_s
point_y = center_y_s
cv2.circle(enlarged, (point_x, point_y), 15, (0, 255, 0), -1)  # Green circle
cv2.circle(enlarged, (point_x, point_y), 17, (255, 255, 255), 3)  # White outline

# Add text labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(enlarged, f'ymin={ymin}', (xmin_s - 150, ymin_s), font, 1, (255, 0, 0), 2)
cv2.putText(enlarged, f'ymax={ymax}', (xmin_s - 150, ymax_s), font, 1, (255, 255, 0), 2)
cv2.putText(enlarged, f'center_y={center_y:.1f}', (xmin_s - 200, center_y_s), font, 1, (0, 255, 0), 2)
cv2.putText(enlarged, f'Point: ({int(xmax)}, {int(center_y)})', (point_x - 300, point_y - 30), font, 1, (255, 255, 255), 2)

# Save
cv2.imwrite('rightmost_point_detail.png', enlarged)
print(f"\\n✓ Detailed visualization saved to: rightmost_point_detail.png")
print(f"  Image is scaled {scale}x for clarity")
print(f"  GREEN line = horizontal line through center of letter")
print(f"  GREEN circle = detected rightmost point")
print(f"  The point should be exactly on the green line")
