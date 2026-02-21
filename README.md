# Text Segmentation and Reflow

A Python package for extracting text from scanned document images and reflowing it onto a new page with improved formatting, proper line wrapping, and consistent spacing.

## Quick Links

📦 **Installation**: See [INSTALL.md](docs/INSTALL.md) for detailed installation instructions  
🪟 **Windows WSL**: See [Step-by-Step Guide for Windows WSL](#step-by-step-guide-for-windows-wsl) below  
📓 **Jupyter Guide**: See [JUPYTER_GUIDE.md](docs/JUPYTER_GUIDE.md) for using in notebooks  
📝 **Example Notebook**: Open `notebooks/example_usage.ipynb` for interactive examples  
🧪 **Test Installation**: Run `python test_package.py` to verify setup

## Features

- 🔄 **Automatic Skew Detection & Correction**: Detects and corrects document skew using MCCSD algorithm (±18° range)
- 📄 **Text Detection & Extraction**: Uses doctr (Document Text Recognition) for detecting text regions and characters
- 🔤 **Character-Level Segmentation**: Extracts individual letter bounding boxes with baseline information
- 📐 **Smart Line Detection**: Groups characters into lines using spatial analysis and clustering
- ✂️ **Intelligent Word Wrapping**: Reflows text to a new page width with proper word boundaries
- 🚫 **Single-Letter Split Prevention**: Prevents awkward word splits like "f"+"act" or "195"+"0"
- 📏 **Consistent Line Spacing**: Robust spacing calculation that handles outliers and maintains readability
- 📑 **Paragraph Detection**: Automatically detects and preserves paragraph breaks with proper indentation
- 🎨 **Background Color Preservation**: Maintains the original page's background color
- 🏗️ **Layout Analysis**: Identifies figures, tables, formulas and preserves their layout
- 🚀 **GPU Acceleration**: Automatic CUDA support for neural network models (2-12x faster on GPU)
- 📚 **Table of Contents Detection**: Three algorithms for detecting TOC pages:
  - **Fine-tuned LayoutLMv3**: Deep learning model ⭐ **BEST** (88.2% accuracy, 3.1x faster)
    - Trained on balanced dataset (34 samples: 17 TOC + 17 non-TOC)
    - Better TOC detection: 82.4% vs 76.5%
    - Production ready ✅
  - **Original Algorithm**: Rule-based detection ✅ **Also Good** (85.3% accuracy)
    - Hand-crafted rules using alignment patterns and page number analysis
    - No training needed, works out-of-the-box
  - **MTD Algorithm**: Multimodal Tree Decoder-inspired approach (research prototype)
  
  **Latest Test Results** (34 pages: 17 TOC + 17 non-TOC):
  - **Fine-tuned LayoutLMv3**: 88.2% accuracy (30/34 correct) 🏆 **WINNER**
  - Original: 85.3% accuracy (29/34 correct) ✅ Still good
  
  **See**: 
  - `docs/TRAINING_34_PAGES_SUCCESS.md` - Final training results with 34 pages
  - `docs/BALANCED_TRAINING_SUCCESS.md` - 26-page training results
  - `docs/COMPARISON_RESULTS.md` - Detailed comparison

## Project Structure

```
segmentation/
├── src/
│   └── ocr_reflow/              # Main package
│       ├── __init__.py          # Package initialization
│       ├── main.py              # Main processing logic
│       ├── reflow.py            # Text reflow and page layout
│       ├── skew_detection.py    # Skew detection and correction
│       ├── layout.py            # Layout analysis integration
│       ├── divide_conquer_4d.py # 4D spatial algorithms
│       └── cli.py               # Command-line interface
├── docs/                        # Documentation
│   ├── CONTRIBUTING.md
│   ├── INSTALL.md               # Installation guide
│   ├── JUPYTER_GUIDE.md         # Jupyter usage guide
│   ├── JUPYTER_TEST.md          # Jupyter quick test
│   ├── PACKAGE_SUMMARY.md       # Package overview
│   ├── SKEW_DETECTION.md        # Skew detection documentation
│   ├── QUICKSTART.md
│   └── WORKFLOW.md
├── notebooks/
│   └── example_usage.ipynb      # Jupyter notebook example
├── tests/
│   ├── test_1950.py             # Test for number splitting
│   ├── test_outlier_spacing.py # Test for line spacing with outliers
│   └── test_*.py                # Additional test cases
├── testscripts/
│   └── test_skew_detection.py   # Skew detection test script
├── images/                      # Sample images for testing
├── examples/
│   └── basic_usage.py           # Example Python script
├── models/                      # ML model files
├── pyproject.toml               # Package metadata and dependencies
├── setup.py                     # Package setup configuration
├── pixi.toml                    # Pixi package manager configuration
├── pixi.lock                    # Locked dependency versions
├── skew_detection.tex           # Algorithm description (LaTeX)
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Installation

### Option 1: Install as a Python Package (Recommended for Jupyter)

Install the package in development mode to use it in Jupyter notebooks:

```bash
# Clone the repository
git clone <your-repo-url>
cd segmentation

# Install in editable mode with all dependencies
pip install -e .

# Or install with dev dependencies (includes Jupyter)
pip install -e ".[dev]"
```

After installation, you can use the package in Jupyter:

```python
from ocr_reflow import process_document
import cv2

result = process_document("document.png")
cv2.imwrite("output.png", result)
```

### Option 2: Using Pixi (For Development)

For development with a complete environment:

1. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd segmentation
   ```

3. **Install dependencies**:

   **For CPU-only (default, works on WSL without NVIDIA):**
   ```bash
   pixi install
   ```

   **For GPU with CUDA support:**
   ```bash
   pixi install -e gpu
   ```

   **Note for Windows WSL users:** The default CPU environment works perfectly on WSL without an NVIDIA GPU. Only use the GPU environment if you have CUDA drivers installed.

4. **Install the package in editable mode**:
   ```bash
   # Activate pixi environment
   pixi shell
   
   # Install the package
   pip install -e .
   ```

   This will install all dependencies including:
   - Python 3.12
   - PyTorch 2.9.1+ (CPU or GPU version depending on environment)
   - torchvision (CPU or GPU version depending on environment)
   - NumPy, SciPy, Matplotlib
   - OpenCV, Shapely
   - python-doctr (Document OCR)
   - JupyterLab (for notebooks)

5. **Verify installation**:
   ```bash
   python --version
   python -c "from ocr_reflow import process_document; print('Package imported successfully!')"
   ```

### Option 3: Manual Environment Setup

If you prefer manual setup:

```bash
# Create a virtual environment
python3.8 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Step-by-Step Guide for Windows WSL

This guide walks you through the complete setup process on Windows WSL (Windows Subsystem for Linux) from scratch.

### Prerequisites

1. **Install WSL** (if not already installed):
   - Open PowerShell as Administrator and run:
     ```powershell
     wsl --install
     ```
   - Restart your computer
   - Set up your Linux username and password when prompted

2. **Verify WSL is running**:
   - Open a new PowerShell or Command Prompt window
   - Type `wsl` and press Enter
   - You should see a Linux terminal prompt

### Step 1: Update WSL System

```bash
# Update package lists
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl git libopencv-dev
```

### Step 2: Install Pixi

```bash
# Download and install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Reload shell configuration to use pixi
source ~/.bashrc

# Verify installation
pixi --version
```

You should see output like: `pixi 0.62.2` or similar.

### Step 3: Clone the Repository

```bash
# Navigate to your preferred directory (e.g., home directory)
cd ~

# Create a code directory (optional but recommended)
mkdir -p ~/code
cd ~/code

# Clone the repository (replace with actual repo URL)
git clone <your-repo-url>
cd segmentation
```

**Note:** If you don't have git credentials set up, you may need to configure them:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 4: Install Dependencies with Pixi

The default CPU environment works perfectly on WSL without NVIDIA GPU:

```bash
# Install the CPU environment (default - works without GPU)
pixi install

# This will take a few minutes as it downloads and installs:
# - Python 3.12
# - PyTorch (CPU version)
# - torchvision (CPU version)
# - NumPy, SciPy, Matplotlib
# - JupyterLab and all other dependencies
```

**Expected output:**
```
✔ The default environment has been installed.
```

### Step 5: Install the Package

```bash
# Activate the pixi environment
pixi shell

# You should see your prompt change to indicate you're in the pixi environment

# Install the ocr-reflow package in editable mode
pip install -e .

# Expected output: "Successfully installed ocr-reflow-0.1.0"
```

### Step 6: Verify Installation

```bash
# Test that the package imports correctly
python -c "import ocr_reflow; print('✓ Package imported successfully!')"

# Run the comprehensive test script
python test_imports.py
```

**Expected output:**
```
✓ Package imported successfully!
```

### Step 7: Start Jupyter Lab

```bash
# Start Jupyter Lab (still in pixi shell)
jupyter lab

# Or run the example notebook directly
jupyter lab notebooks/example_usage.ipynb
```

**Expected behavior:**
- Jupyter Lab will start and show a URL like: `http://localhost:8888/lab?token=...`
- Your default Windows browser should automatically open
- If not, copy the URL from the terminal and paste it into your browser

### Step 8: Run the Example Notebook

Once Jupyter Lab opens in your browser:

1. Navigate to `notebooks/example_usage.ipynb` in the file browser
2. Click to open the notebook
3. Run cells one by one using `Shift+Enter`, or run all cells with `Run > Run All Cells`
4. The notebook will:
   - Import the package
   - Load a sample image
   - Process and reflow the text
   - Display the results

### Step 9: Process Your Own Documents

```bash
# Exit Jupyter Lab (Ctrl+C in the terminal)

# Process an image using the CLI
ocr-reflow your_document.png

# Or specify output filename
ocr-reflow your_document.png output.png
```

### Accessing Windows Files from WSL

You can access your Windows files from WSL:

```bash
# Windows C: drive is mounted at /mnt/c/
cd /mnt/c/Users/YourUsername/Documents

# Process a Windows file
ocr-reflow /mnt/c/Users/YourUsername/Documents/document.png

# Save output to Windows location
ocr-reflow input.png /mnt/c/Users/YourUsername/Documents/output.png
```

### Opening Jupyter Lab from Windows

If you want Jupyter Lab to automatically open in your Windows browser:

```bash
# In WSL, start Jupyter Lab
jupyter lab --no-browser

# Copy the URL shown (something like http://localhost:8888/lab?token=...)
# Paste it into your Windows browser
```

### Troubleshooting WSL-Specific Issues

#### Issue: "pixi: command not found"

**Solution:**
```bash
# Reload shell configuration
source ~/.bashrc

# Or restart WSL
exit
# Then open WSL again
```

#### Issue: Port 8888 Already in Use

**Solution:**
```bash
# Use a different port
jupyter lab --port=8889

# Or find and kill the process using port 8888
lsof -ti:8888 | xargs kill -9
```

#### Issue: Browser Doesn't Open Automatically

**Solution:**
```bash
# Start Jupyter without auto-opening browser
jupyter lab --no-browser

# Copy the URL and paste it into your Windows browser
```

#### Issue: Slow Performance

**Note:** CPU-based text processing is slower than GPU but still functional:
- Small images (< 2MB): Usually fast enough
- Large images (> 5MB): May take several minutes
- This is normal for CPU processing

#### Issue: Cannot Access Windows Files

**Solution:**
```bash
# Make sure you're using the /mnt/ prefix
ls /mnt/c/Users/

# Check if the Windows drive is mounted
mount | grep mnt
```

### Quick Command Reference for WSL

```bash
# Start WSL from Windows
wsl

# Exit WSL
exit

# Activate pixi environment
cd ~/code/segmentation
pixi shell

# Run Jupyter Lab
jupyter lab

# Process a document
ocr-reflow document.png

# Check package version
pip show ocr-reflow

# Update dependencies
pixi update

# Reinstall environment if needed
rm -rf .pixi/envs/default
pixi install
pip install -e .
```

### Next Steps

Now that you have everything set up, you can:

1. **Explore the example notebook**: `notebooks/example_usage.ipynb`
2. **Read the documentation**: See `docs/` folder for detailed guides
3. **Process your documents**: Use the CLI or Python API
4. **Experiment with parameters**: Try different zoom factors, margins, etc.

For more details, see:
- [Installation Guide](docs/INSTALL.md) - Detailed installation instructions
- [Jupyter Guide](docs/JUPYTER_GUIDE.md) - Using the package in notebooks
- [WSL Compatibility](docs/WSL_COMPATIBILITY.md) - Technical details about WSL support

## Usage

### Command-Line Interface

After installing the package, you can use the CLI:

```bash
# Process a document (output will be named <input>_reflowed.png)
ocr-reflow document.png

# Specify output filename
ocr-reflow document.png output.png

# Process with layout analysis (detects figures, tables, TOC pages)
python src/ocr_reflow/main.py document.png --layout

# Use MTD algorithm for Table of Contents detection
python src/ocr_reflow/main.py document.png --layout --toc-algorithm mtd

# Use original rule-based algorithm (default)
python src/ocr_reflow/main.py document.png --layout --toc-algorithm original

# Adjust page width and zoom factor
python src/ocr_reflow/main.py document.png --layout --page-width 1600 --zoom-factor 2.0
```

#### Table of Contents Detection

When using `--layout`, the system can automatically detect Table of Contents pages and apply special formatting to preserve the vertical alignment of page numbers. You can choose between two detection algorithms:

**Original Algorithm (default)** - Rule-based approach:
- Analyzes right-edge alignment of text lines
- Detects page numbers by width patterns
- Fast and reliable for traditional TOC layouts
- Use: `--toc-algorithm original` or omit the flag

**MTD Algorithm** - Multimodal Tree Decoder inspired approach:
- Based on research paper "Multimodal Tree Decoder for Table of Contents Extraction"
- Combines visual, textual, and layout features
- Uses attention mechanisms to build hierarchical structure
- Better for complex or non-traditional TOC layouts
- Use: `--toc-algorithm mtd`

Example:
```bash
# Detect and reflow a TOC page with MTD algorithm
python src/ocr_reflow/main.py images/mh_p005.png --layout --toc-algorithm mtd

# Process multiple pages with original algorithm
for img in images/book_p*.png; do
    python src/ocr_reflow/main.py "$img" --layout --toc-algorithm original
done
```

### GPU Acceleration (CUDA)

The neural network models (MTD and LayoutLMv3) automatically use GPU acceleration when available:

**Automatic Detection**:
- No configuration needed
- Automatically detects CUDA GPU
- Falls back to CPU if GPU not available
- Works transparently in all modes

**Performance**:
- GPU speedup: 2-12x faster than CPU
- Tested on: NVIDIA GeForce RTX 3050 4GB
- Memory usage: ~500-800 MB GPU RAM per model

**Verify CUDA is Working**:
```bash
# Run CUDA verification test
python test_cuda.py

# Quick check
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU during processing
watch -n 1 nvidia-smi
```

**System Requirements for GPU**:
- NVIDIA GPU with CUDA support
- CUDA Toolkit (installed automatically with PyTorch)
- 2GB+ GPU memory recommended
- No changes needed to use GPU - it's automatic!

See `docs/CUDA_INTEGRATION_COMPLETE.md` for details.

### Using in Jupyter Notebooks

The package is designed to work seamlessly in Jupyter notebooks. See the example notebook at `notebooks/example_usage.ipynb`.

```python
# Import the package
from ocr_reflow import process_document
import cv2
from matplotlib import pyplot as plt

# Process a document
result = process_document("document.png")

# Display the result
plt.figure(figsize=(12, 16))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the result
cv2.imwrite("output.png", result)
```

### Python Script Usage

You can also use the package in your Python scripts:

```python
from ocr_reflow import process_document
import cv2

# Process a single document
filename = "your_document.png"
reflowed_page = process_document(filename)

# Save the result
cv2.imwrite("reflowed_output.png", reflowed_page)
```

### Advanced Usage - Direct Module Access

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

## Using in Jupyter Notebooks

The package is designed to work seamlessly in Jupyter notebooks. After installation, you can use it directly:

### Quick Start

```python
from ocr_reflow import process_document
import cv2
from matplotlib import pyplot as plt

# Process a document
result = process_document("document.png")

# Display the result
plt.figure(figsize=(12, 16))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

### Running the Example Notebook

**Note**: The notebook has been recently updated (Feb 2026) to fix kernel crash issues. The new version is lightweight (7.4KB) and includes all latest features: title handling with extra spacing, horizontal baselines, and layout-aware processing.

```bash
# Start Jupyter Lab with the example notebook
pixi run jupyter lab notebooks/example_usage.ipynb

# Or start Jupyter Lab without opening a specific notebook
pixi run jupyter lab

# Or if using venv
jupyter lab notebooks/example_usage.ipynb
```

Then open `notebooks/example_usage.ipynb` for a complete tutorial with:
- Layout-aware document processing
- Title handling with proper spacing
- Smart skew detection (plain text only)
- Visualization examples
- Troubleshooting tips

For more details, see [JUPYTER_GUIDE.md](docs/JUPYTER_GUIDE.md).

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
```

**Solution:** Use the CPU environment instead:
```bash
# Reinstall with CPU environment (default)
pixi install

# Or if you already have the GPU environment
pixi install -e default
```

The CPU environment works on all systems including Windows WSL without NVIDIA GPU.

### Import Errors

Make sure you're in the Pixi environment:
```bash
pixi shell
cd src
python main.py <image>
```

## Pixi Environments

This project supports two Pixi environments to accommodate different hardware configurations:

### CPU Environment (Default)
- **Use case:** Systems without NVIDIA GPU, Windows WSL, or when CUDA is not needed
- **Installation:** `pixi install` or `pixi install -e default`
- **Features:** CPU-only PyTorch, no CUDA dependencies
- **Compatibility:** Works on all Linux systems, including WSL

### GPU Environment
- **Use case:** Systems with NVIDIA GPU and CUDA 12 drivers
- **Installation:** `pixi install -e gpu`
- **Features:** PyTorch with CUDA 12 support
- **Compatibility:** Requires NVIDIA GPU with CUDA drivers installed

### Switching Environments

```bash
# Switch to CPU environment
pixi install -e default
pixi shell

# Switch to GPU environment (requires CUDA)
pixi install -e gpu
pixi shell
```

### Windows WSL Notes

Pixi works seamlessly on Windows WSL:
- WSL is treated as a native Linux environment
- The default CPU environment works without any NVIDIA drivers
- If you have WSL with NVIDIA GPU support, you can use the GPU environment
- No special configuration needed for WSL compatibility

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
