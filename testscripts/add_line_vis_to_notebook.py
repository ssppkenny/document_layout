#!/usr/bin/env python3
"""
Add line detection visualization to reflow_layout_analysis.ipynb
"""

import json
import sys

notebook_path = "notebooks/reflow_layout_analysis.ipynb"

# Load notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Check if already added
for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown' and cell.get('source'):
        content = ''.join(cell['source'])
        if 'Step 3a' in content and 'Visualize Detected Text Lines' in content:
            print("Line detection visualization already exists in notebook")
            sys.exit(0)

# New markdown cell
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 3a: Visualize Detected Text Lines\n",
        "\n",
        "For plain text boxes, we detect text lines by finding leftmost and rightmost points. This visualization shows:\n",
        "- **Blue circles**: Leftmost points of each line\n",
        "- **Yellow circles**: Rightmost points of each line\n",
        "- **Colored lines**: Detected text baselines\n",
        "- **Gray rectangles**: Individual detected words"
    ]
}

# New code cell - simplified version that works with the notebook context
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Import visualization function\n",
        "from ocr_reflow import margins, visualize_detected_lines\n",
        "from doctr.models import detection_predictor\n",
        "from doctr.io import DocumentFile\n",
        "\n",
        "print(\"Detecting text lines...\")\n",
        "\n",
        "# Run word detection on the full image\n",
        "model = detection_predictor(pretrained=True)\n",
        "docs = DocumentFile.from_images([image_path])\n",
        "result = model(docs)\n",
        "words = result[0][\"words\"]\n",
        "\n",
        "# Convert normalized coordinates to absolute\n",
        "words[:, 0] = (words[:, 0] * img_w).astype(np.int32)\n",
        "words[:, 1] = (words[:, 1] * img_h).astype(np.int32) + 2\n",
        "words[:, 2] = (words[:, 2] * img_w).astype(np.int32)\n",
        "words[:, 3] = (words[:, 3] * img_h).astype(np.int32) - 2\n",
        "words = words.astype(np.int32)\n",
        "\n",
        "# Detect margins (lines)\n",
        "left_margins, right_margins = margins(words)\n",
        "print(f\"Detected {len(left_margins)} text lines\")\n",
        "\n",
        "# Create visualization\n",
        "line_vis_img = visualize_detected_lines(original_img, words, left_margins, right_margins)\n",
        "\n",
        "# Display\n",
        "plt.figure(figsize=(15, 20))\n",
        "plt.imshow(cv2.cvtColor(line_vis_img, cv2.COLOR_BGR2RGB))\n",
        "plt.title(f'Detected Text Lines ({len(left_margins)} lines)', fontsize=16)\n",
        "plt.axis('off')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Print line details\n",
        "print(f\"\\nLine details:\")\n",
        "for i, (l, r) in enumerate(zip(left_margins, right_margins)):\n",
        "    width = r[0] - l[0]\n",
        "    print(f\"  Line {i+1:2d}: ({l[0]:4d}, {l[1]:3d}) → ({r[0]:4d}, {r[1]:3d})  width={width:4d}px\")"
    ]
}

# Insert the new cells after cell 10 (layout visualization)
# This will become cells 11 and 12, pushing the rest down
insert_position = 11

notebook['cells'].insert(insert_position, markdown_cell)
notebook['cells'].insert(insert_position + 1, code_cell)

# Save the modified notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✓ Added line detection visualization to {notebook_path}")
print(f"  - Inserted markdown cell at position {insert_position}")
print(f"  - Inserted code cell at position {insert_position + 1}")
print(f"  - Total cells now: {len(notebook['cells'])}")
