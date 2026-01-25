# Text Segmentation and Reflow

A Python project for extracting text from scanned document images and reflowing it onto a new page with improved formatting, proper line wrapping, and consistent spacing.

## Features

- 📄 **Text Detection & Extraction**: Uses doctr (Document Text Recognition) for detecting text regions and characters
- 🔤 **Character-Level Segmentation**: Extracts individual letter bounding boxes with baseline information
- 📐 **Smart Line Detection**: Groups characters into lines using spatial analysis and clustering
- ✂️ **Intelligent Word Wrapping**: Reflows text to a new page width with proper word boundaries
- 🚫 **Single-Letter Split Prevention**: Prevents awkward word splits like "f"+"act" or "195"+"0"
- 📏 **Consistent Line Spacing**: Robust spacing calculation that handles outliers and maintains readability
- 📑 **Paragraph Detection**: Automatically detects and preserves paragraph breaks with proper indentation
- 🎨 **Background Color Preservation**: Maintains the original page's background color

## Project Structure

```
segmentation/
├── src/
│   ├── main.py                    # Main script for processing images
│   ├── reflow.py                  # Text reflow and page layout logic
│   └── divide_conquer_4d.py       # 4D spatial algorithms for rectangle analysis
├── tests/
│   ├── test_1950.py              # Test for number splitting
│   ├── test_outlier_spacing.py  # Test for line spacing with outliers
│   └── test_*.py                 # Additional test cases
├── pixi.toml                     # Pixi package manager configuration
├── pixi.lock                     # Locked dependency versions
└── README.md                     # This file
```

## Installation

### Prerequisites

- Linux x64 system (as specified in pixi.toml)
- CUDA 13.0 (for GPU acceleration)
- [Pixi](https://pixi.sh) package manager

### Setting Up the Environment with Pixi

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd segmentation
   ```

3. **Create and activate the environment**:
   ```bash
   # Pixi will automatically create the environment from pixi.toml
   pixi install
   
   # Activate the environment
   pixi shell
   ```

   This will install all dependencies including:
   - Python 3.12
   - PyTorch 2.9.1+
   - NumPy, SciPy, Matplotlib
   - OpenCV, Shapely
   - python-doctr (Document OCR)
   - JupyterLab (for notebooks)

4. **Verify installation**:
   ```bash
   python --version
   python -c "import cv2; import doctr; print('All imports successful!')"
   ```

### Alternative: Manual Environment Setup

If you prefer not to use Pixi:

```bash
# Create a virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch>=2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Install other dependencies
pip install numpy>=2.3.5 scipy>=1.16.3 matplotlib>=3.10.8
pip install opencv-python>=4.11.0.86 shapely>=2.1.2
pip install python-doctr>=1.0.0
pip install jupyterlab>=4.5.0 ipywidgets>=8.1.8
```

## Usage

### Basic Usage

Process a scanned document image and reflow the text:

```bash
cd src
python main.py <path_to_image.png>
```

**Example:**
```bash
python main.py ../test_economist_january_original.png
```

**Output files:**
- `out.png` - Reflowed page with text
- `out1.png` - Original image with detected letter bounding boxes (debug)
- `out2.png` - Detected lines visualization (debug)

### Python API Usage

```python
from main import process_document
import cv2

# Load your image
image = cv2.imread('your_document.png')

# Process and reflow
reflowed_page = process_document(
    image,
    zoom_factor=1.5,          # Scale factor for text
    new_page_width=800,       # Width of new page in pixels
    left_margin=50,           # Left margin
    right_margin=50,          # Right margin
    top_margin=50,            # Top margin
    bottom_margin=50,         # Bottom margin
    line_spacing=20,          # Extra spacing between lines
    paragraph_spacing=40      # Extra spacing between paragraphs
)

# Save result
cv2.imwrite('output.png', reflowed_page)
```

### Using the Reflow Module Directly

```python
from reflow import create_page_with_word_wrapping, Letter
import cv2
import numpy as np

# Define your lines (list of lists of Letter objects)
lines = [
    [
        Letter(xmin=10, ymin=10, xmax=30, ymax=30, bl=5),
        Letter(xmin=35, ymin=10, xmax=55, ymax=30, bl=5),
        # ... more letters
    ],
    # ... more lines
]

# Create original image
original_image = cv2.imread('source.png')

# Reflow text
new_page = create_page_with_word_wrapping(
    lines=lines,
    original_image=original_image,
    zoom_factor=1.5,
    new_page_width=800,
    left_margin=50,
    right_margin=50,
    top_margin=50,
    bottom_margin=50,
    line_spacing=20,
    paragraph_spacing_factor=2.0,
    preserve_spacing=True
)

cv2.imwrite('reflowed.png', new_page)
```

## Features in Detail

### 1. Word Split Prevention

The system prevents awkward word splits where only 1 character remains on a line:

- ❌ Before: `"fact"` splits as `"f"` on one line, `"act"` on next
- ✅ After: `"fact"` moves entirely to new line

This applies to both text and numbers (e.g., `"1950"` won't split as `"195"` + `"0"`).

### 2. Robust Line Spacing

Uses percentile-based calculations to handle outlier letters with incorrect baseline values:

- Calculates 95th percentile instead of maximum
- Applies safety cap at 2.5× typical letter height
- Prevents one bad character from ruining all line spacing

**Example output when outlier detected:**
```
[Line Spacing] Capping line height from 338 to 90 (detected outlier)
```

### 3. Paragraph Detection

Automatically detects paragraph breaks by analyzing:
- Horizontal indentation of first letters
- Short lines (lines significantly shorter than average)
- Preserves paragraph structure in reflowed output
- Applies book-style indentation (~3.5 character widths)

### 4. Baseline-Aware Placement

Each letter is placed with its baseline aligned correctly:
- Maintains proper vertical alignment
- Handles descenders (g, j, p, q, y)
- Handles ascenders (b, d, f, h, k, l, t)
- Ensures consistent text appearance

## Running Tests

The project includes several test scripts to verify functionality:

```bash
# Test number splitting prevention
python test_1950.py

# Test line spacing with outliers
python test_outlier_spacing.py

# Test word split prevention
python test_midword_fact.py

# Run all tests
for test in test_*.py; do python "$test"; done
```

## Jupyter Notebook

Explore the functionality interactively:

```bash
pixi shell
jupyter lab segmentation.ipynb
```

## Development

### Code Formatting

The project uses Black for code formatting:

```bash
pixi run black src/
```

### Adding Dependencies

**With Pixi:**
```bash
# Add conda dependency
pixi add package-name

# Add PyPI dependency
pixi add --pypi package-name
```

**Manual:**
Edit `pixi.toml` and run:
```bash
pixi install
```

## Troubleshooting

### Line Spacing Too Large

If you see huge gaps between lines, the system should automatically detect and fix this. Look for:
```
[Line Spacing] Capping line height from X to Y (detected outlier)
```

This indicates some letters have incorrect baseline values, but the spacing was corrected.

### Word Splits Still Occurring

Ensure `preserve_spacing=True` in the reflow function. The word split prevention requires analyzing letter spacing.

### CUDA Errors

If you see CUDA-related errors:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU instead (modify main.py to not use GPU)
```

### Import Errors

Make sure you're in the Pixi environment:
```bash
pixi shell
cd src
python main.py <image>
```

## Technical Details

### Letter Data Structure

Each letter contains:
- `xmin, ymin, xmax, ymax`: Bounding box coordinates
- `bl`: Baseline offset from bottom of bounding box

### Algorithms Used

1. **Text Detection**: doctr's detection_predictor
2. **Line Grouping**: Spatial clustering based on y-coordinates
3. **Word Boundary Detection**: Space analysis (threshold at 0.5× avg character width)
4. **Paragraph Detection**: Horizontal indentation analysis + short line detection
5. **Spacing Calculation**: 95th percentile with safety cap

## License

[Specify your license here]

## Author

Sergey Mikhno <sergey.mikhno@gmail.com>

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 0.1.0 (January 2026)
- Initial release
- Text detection and character extraction
- Smart word wrapping with split prevention
- Robust line spacing calculation
- Paragraph detection and preservation
- Background color preservation

---

**Need help?** Open an issue or contact the author.
