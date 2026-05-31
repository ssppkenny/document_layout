# Text Segmentation and Reflow

A Python package for extracting text from scanned document images (PNG, JPEG, PDF, DjVu) and reflowing it onto a new page with improved formatting, proper line wrapping, and consistent spacing.

## Important: Model Download Required

After cloning this repository, the LayoutLMv3 TOC detection model (~484 MB) downloads automatically from HuggingFace on first use.

**Quick Setup:**
```bash
git clone <your-repo-url>
cd segmentation
pixi install

# Model downloads automatically on first run from HuggingFace:
pixi run python src/ocr_reflow/main.py images/mh_p005.png --layout
# Downloads: ssppkenny/layoutlmv3-toc-detector (~484 MB, one-time)
```

**Model**: [layoutlmv3-toc-detector](https://huggingface.co/ssppkenny/layoutlmv3-toc-detector) (HuggingFace Hub)

See [SETUP_AFTER_CLONE.md](docs/SETUP_AFTER_CLONE.md) for detailed setup instructions.

## Quick Links

- **Installation**: See [INSTALL.md](docs/INSTALL.md) for detailed installation instructions
- **Quick Setup**: See [SETUP_AFTER_CLONE.md](docs/SETUP_AFTER_CLONE.md) for setup after git clone
- **Windows WSL**: See [Step-by-Step Guide for Windows WSL](#step-by-step-guide-for-windows-wsl) below
- **Jupyter Guide**: See [JUPYTER_GUIDE.md](docs/JUPYTER_GUIDE.md) for using in notebooks
- **Example Notebook**: Open `notebooks/example_usage.ipynb` for interactive examples
- **Test Installation**: Run `pixi run python tests/test_package.py` to verify setup

## Features

- **Automatic Skew Detection & Correction**: Detects and corrects document skew using MCCSD algorithm (±18° range)
- **Text Detection & Extraction**: Uses doctr (Document Text Recognition) for detecting text regions and characters
- **Character-Level Segmentation**: Extracts individual letter bounding boxes with baseline information
- **Smart Line Detection**: Groups characters into lines using spatial analysis and clustering
- **Word-Level Reflow** (`--word-reflow`): Reflows text using whole-word crops as atomic units — more robust than letter-level for complex scripts
- **Hyphenation Support** (`--lang`): Grammatical word splitting at line breaks using pyphen dictionaries and Tesseract OCR; supports Russian, English, Swedish, and other languages
- **Binarization** (`--bin`): Otsu binarization pre-processing to improve detection of diacritics (Swedish, Czech, etc.)
- **Intelligent Word Wrapping**: Reflows text to a new page width with proper word boundaries
- **Single-Letter Split Prevention**: Prevents awkward word splits like "f"+"act" or "195"+"0"
- **Consistent Line Spacing**: Robust spacing calculation that handles outliers and maintains readability
- **Paragraph Detection**: Automatically detects and preserves paragraph breaks with proper indentation
- **Background Color Preservation**: Maintains the original page's background color
- **Layout Analysis**: Identifies figures, tables, formulas and preserves their layout
- **Multi-Format Input**: Processes PNG, JPEG, PDF, and DjVu files; PDF/DjVu rendered at 300 DPI via PyMuPDF
- **GPU Acceleration**: Automatic CUDA support for neural network models (2-12x faster on GPU)
- **Table of Contents Detection**: Three algorithms for detecting TOC pages:
  - **Fine-tuned LayoutLMv3**: Deep learning model — **BEST** (100% accuracy on 54-page test set)
  - **Original Algorithm**: Rule-based detection — fast, no model required
  - **MTD Algorithm**: Multimodal Tree Decoder-inspired approach (research prototype)

## Project Structure

```
segmentation/
├── src/
│   └── ocr_reflow/              # Main package
│       ├── __init__.py          # Package initialization
│       ├── main.py              # Main processing logic and CLI entry point
│       ├── reflow.py            # Letter-level text reflow and page layout
│       ├── reflow_words.py      # Word-level reflow with hyphenation support
│       ├── skew_detection.py    # Skew detection and correction (MCCSD)
│       ├── layout.py            # Layout analysis (doclayout-yolo + YOLOv26 ensemble)
│       ├── binarization.py      # Otsu binarization pre-processing
│       ├── document_loader.py   # PDF and DjVu loading via PyMuPDF
│       ├── divide_conquer_4d.py # 4D spatial algorithms
│       ├── toc_detection.py     # Rule-based TOC detection
│       ├── toc_detection_mtd.py # MTD TOC detection
│       ├── layoutlm_toc_detector.py  # LayoutLMv3 TOC detector
│       ├── mtd_toc_detector.py  # MTD model wrapper
│       ├── model_manager.py     # Model download and management
│       ├── device_utils.py      # CUDA/CPU device selection
│       ├── diacritic_merger.py  # Diacritic merging for complex scripts
│       ├── visualize_reflow.py  # Word segmentation visualization
│       ├── cli.py               # Command-line interface (ocr-reflow entry point)
│       ├── extractor.py         # OCR text extraction from layout regions
│       ├── language_detection.py # Auto-detect OCR language from first pages
│       ├── ocr_export_layout.py # OCR + HTML export for EPUB pipeline
│       ├── epub_export.py       # EPUB 3 generation from page results
│       ├── fix_epub_spelling.py # Hunspell-based spelling correction for EPUBs
│       ├── server.py            # FastAPI server for remote processing
│       └── ocr_export.py        # Legacy compatibility shim
├── scripts/                     # Utility scripts
│   ├── translate_epub.py        # EPUB translation (T5/NLLB/M2M100)
│   ├── translate_chapter.py     # Single-chapter translation
│   ├── train_layoutlmv3.py      # LayoutLMv3 TOC detector training
│   ├── test_sse.py              # SSE event stream test
│   └── patch_epub_*.py          # EPUB patching helpers
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
│   ├── test_outlier_spacing.py  # Test for line spacing with outliers
│   └── test_*.py                # Additional test cases
├── testscripts/
│   └── test_skew_detection.py   # Skew detection test script
├── images/                      # Sample images for testing
├── examples/
│   └── basic_usage.py           # Example Python script
├── models/                      # ML model files
├── pyproject.toml               # Package metadata and dependencies
├── pixi.toml                    # Pixi package manager configuration
├── pixi.lock                    # Locked dependency versions
├── skew_detection.tex           # Algorithm description (LaTeX)
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Installation

### Option 1: Using Pixi (Recommended for Development)

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

4. **Install Tesseract** (required for `--lang` hyphenation):
   ```bash
   sudo apt install tesseract-ocr
   # For additional languages:
   sudo apt install tesseract-ocr-rus tesseract-ocr-swe
   # Or copy .traineddata files to /usr/share/tessdata/
   ```

5. **Verify installation**:
   ```bash
   pixi run python src/ocr_reflow/main.py --help
   ```

### Option 2: Install as a Python Package

```bash
git clone <your-repo-url>
cd segmentation

# Install in editable mode with all dependencies
pip install -e .

# Or install with dev dependencies (includes Jupyter)
pip install -e ".[dev]"
```

After installation, you can use the package in Python:

```python
from ocr_reflow import process_document
import cv2

result = process_document("document.png")
cv2.imwrite("output.png", result)
```

### Option 3: Manual Environment Setup

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install -e .
```

## Models

This project uses several machine learning models stored in the `models/` directory:

### Model Files

1. **DocLayout-YOLO** (`models/doclayout_yolo_docstructbench_imgsz1024.pt`)
   - **Purpose**: Document layout analysis (detects titles, text blocks, figures, tables, formulas)
   - **Size**: ~39 MB
   - **Source**: [HuggingFace](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench)
   - **Auto-download**: Will download on first use if not present

2. **Fine-tuned LayoutLMv3** — [Available on HuggingFace](https://huggingface.co/YOUR_USERNAME/layoutlmv3-toc-detector)
   - **Purpose**: Table of Contents detection (TOC vs non-TOC classification)
   - **Size**: ~484 MB
   - **Performance**: **100.00% accuracy** on 54-page test set (27 TOC + 27 non-TOC)
   - **Hosted on**: [HuggingFace Hub](https://huggingface.co/YOUR_USERNAME/layoutlmv3-toc-detector)
   - **Auto-download**: Downloads automatically on first use

3. **DocTR** (auto-downloaded by library)
   - **Purpose**: Text detection and character-level OCR
   - **Location**: `~/.cache/doctr/models/`
   - **Auto-download**: Automatically downloaded by doctr library on first use

### Downloading the LayoutLMv3 Model

**Option 1: Automatic Download (Recommended)**

The model downloads automatically on first use:

```bash
pixi run python src/ocr_reflow/main.py images/mh_p005.png --layout
# Downloads LayoutLMv3 model from HuggingFace (~484 MB, one-time)
# Cached in ~/.cache/ocr_reflow/models/
```

**Option 2: Manual Download**

```bash
pixi run pip install huggingface-hub

pixi run python -c "
from huggingface_hub import snapshot_download
from pathlib import Path

model_path = Path('models/layoutlmv3_toc/best_model')
model_path.parent.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id='YOUR_USERNAME/layoutlmv3-toc-detector',
    local_dir=str(model_path)
)
print('Model downloaded successfully!')
"
```

**Option 3: Using HuggingFace CLI**

```bash
pixi run hf hub download YOUR_USERNAME/layoutlmv3-toc-detector \
    --local-dir models/layoutlmv3_toc/best_model/ \
    --repo-type model
```

### Model Management

Check installed models:
```bash
pixi run python src/ocr_reflow/model_manager.py info
```

Download missing models:
```bash
pixi run python src/ocr_reflow/model_manager.py download
```

### Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **100.00%** |
| **Training Dataset** | 54 pages (27 TOC + 27 non-TOC) |
| **Model Size** | 484 MB |
| **Speed** | 3.1s per page |

See `models/README.md` for detailed model management information.

### Training Your Own TOC Detection Model (Optional)

```bash
pixi run python scripts/train_layoutlmv3.py
```

Training takes ~2 minutes on GPU (NVIDIA RTX 3050 or better), ~10-15 minutes on CPU.

**Note**: Training is optional — the pre-trained model from HuggingFace is ready to use.

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
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential curl git libopencv-dev
```

### Step 2: Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
pixi --version
```

You should see output like: `pixi 0.62.2` or similar.

### Step 3: Clone the Repository

```bash
cd ~
mkdir -p ~/code
cd ~/code

git clone <your-repo-url>
cd segmentation
```

**Note:** If you don't have git credentials set up:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 4: Install Dependencies with Pixi

```bash
pixi install
```

**Expected output:**
```
The default environment has been installed.
```

### Step 5: Install the Package

```bash
pixi shell
pip install -e .
# Expected output: "Successfully installed ocr-reflow-0.1.0"
```

### Step 6: Verify Installation

```bash
python -c "import ocr_reflow; print('Package imported successfully!')"
python tests/test_imports.py
```

### Step 7: Start Jupyter Lab

```bash
jupyter lab
# Or open a specific notebook:
jupyter lab notebooks/example_usage.ipynb
```

Jupyter Lab will show a URL like `http://localhost:8888/lab?token=...`. Your default Windows browser should open automatically. If not, copy the URL and paste it into your browser.

### Step 8: Run the Example Notebook

Once Jupyter Lab opens:

1. Navigate to `notebooks/example_usage.ipynb`
2. Click to open the notebook
3. Run cells with `Shift+Enter`, or run all with `Run > Run All Cells`

### Step 9: Process Your Own Documents

```bash
# Process an image using the CLI
pixi run python src/ocr_reflow/main.py your_document.png --layout --word-reflow

# Or specify page width and zoom
pixi run python src/ocr_reflow/main.py your_document.png --layout --word-reflow --page-width 2000 --zoom-factor 2.5
```

### Accessing Windows Files from WSL

```bash
# Windows C: drive is mounted at /mnt/c/
cd /mnt/c/Users/YourUsername/Documents

# Process a Windows file
pixi run python src/ocr_reflow/main.py /mnt/c/Users/YourUsername/Documents/document.png --layout
```

### Opening Jupyter Lab from Windows

```bash
jupyter lab --no-browser
# Copy the URL shown and paste it into your Windows browser
```

### Troubleshooting WSL-Specific Issues

#### Issue: "pixi: command not found"

```bash
source ~/.bashrc
# Or restart WSL: exit, then open WSL again
```

#### Issue: Port 8888 Already in Use

```bash
jupyter lab --port=8889
# Or kill the process: lsof -ti:8888 | xargs kill -9
```

#### Issue: Browser Doesn't Open Automatically

```bash
jupyter lab --no-browser
# Copy the URL and paste it into your Windows browser
```

#### Issue: Slow Performance

CPU-based text processing is slower than GPU but still functional:
- Small images (< 2MB): Usually fast enough
- Large images (> 5MB): May take several minutes

#### Issue: Cannot Access Windows Files

```bash
ls /mnt/c/Users/
mount | grep mnt
```

### Quick Command Reference for WSL

```bash
# Start WSL from Windows
wsl

# Activate pixi environment
cd ~/code/segmentation
pixi shell

# Run Jupyter Lab
jupyter lab

# Process a document
pixi run python src/ocr_reflow/main.py document.png --layout --word-reflow

# Check package version
pip show ocr-reflow

# Update dependencies
pixi update

# Reinstall environment if needed
rm -rf .pixi/envs/default
pixi install
pip install -e .
```

## Usage

### Command-Line Interface

Two entry points are available:

#### Primary: `ocr-reflow` (installed entry point)

After `pip install -e .`, the `ocr-reflow` command provides:

```
usage: ocr-reflow [-h] [--page N] filename
```

Process a single page with optional page selection for multi-page files:

| Argument | Default | Description |
|---|---|---|
| `filename` | — | Input file: PNG, JPEG, PDF, or DjVu |
| `--page N` | 0 | 0-based page number for PDF and DjVu files |

#### Script mode: `main.py` (full flag set)

For more options, run `main.py` directly via pixi:

```
pixi run python src/ocr_reflow/main.py [-h] [--layout] [--bin] [--no-output]
               [--show-words] [--page-width N] [--zoom-factor F]
               [--toc-algorithm ALG] [--page N] [--word-reflow] [--lang LANG]
               filename
```

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `filename` | — | Input file: PNG, JPEG, PDF, or DjVu |
| `--layout` | off | Enable layout-aware processing (detects titles, figures, tables, TOC) |
| `--word-reflow` | off | Use word-level reflow instead of letter-level reflow |
| `--lang LANG` | none | Language code for hyphenation (e.g. `ru`, `en`, `sv`). Requires `--word-reflow`. Enables pyphen + Tesseract word splitting |
| `--bin` | off | Apply Otsu binarization before processing (helps with diacritics: Swedish, Czech, etc.) |
| `--page-width N` | 2000 | Width of the output page in pixels |
| `--zoom-factor F` | 2.5 | Scaling factor for letter/word crops (controls text size independently of page width) |
| `--page N` | 0 | 0-based page number for PDF and DjVu files |
| `--toc-algorithm ALG` | `layoutlm` | TOC detection algorithm: `layoutlm` (LayoutLMv3), `original` (rule-based), `mtd` (Multimodal Tree Decoder) |
| `--show-words` | off | Generate a word segmentation visualization image |
| `--no-output` | off | Skip writing output images (useful for benchmarking) |

#### Examples

```bash
# Basic reflow of a scanned image
pixi run python src/ocr_reflow/main.py images/page.png --layout --word-reflow

# Russian text with hyphenation
pixi run python src/ocr_reflow/main.py images/page.png --layout --word-reflow --lang ru

# Swedish text with binarization (improves diacritic detection)
pixi run python src/ocr_reflow/main.py images/page.png --layout --word-reflow --lang sv --bin

# Process page 17 of a DjVu file, wider output, smaller zoom
pixi run python src/ocr_reflow/main.py book.djvu --page 17 --layout --word-reflow --page-width 5000 --zoom-factor 2 --lang ru

# Process a specific PDF page
pixi run python src/ocr_reflow/main.py document.pdf --page 3 --layout --word-reflow

# Use rule-based TOC detection instead of LayoutLMv3
pixi run python src/ocr_reflow/main.py images/page.png --layout --toc-algorithm original

# Generate word segmentation visualization
pixi run python src/ocr_reflow/main.py images/page.png --layout --word-reflow --show-words
```

### Table of Contents Detection

When using `--layout`, the system automatically detects TOC pages and preserves vertical alignment of page numbers. Three algorithms are available via `--toc-algorithm`:

**LayoutLMv3 (default)** — Fine-tuned deep learning model:
- 100% accuracy on 54-page test set
- Downloads automatically from HuggingFace (~484 MB, one-time)
- Use: `--toc-algorithm layoutlm` (or omit — this is the default)

**Original Algorithm** — Rule-based approach:
- Analyzes right-edge alignment of text lines and page number patterns
- No model download required
- Use: `--toc-algorithm original`

**MTD Algorithm** — Multimodal Tree Decoder inspired approach:
- Research prototype; better for complex or non-traditional TOC layouts
- Use: `--toc-algorithm mtd`

### Word Reflow and Hyphenation

The `--word-reflow` flag switches from letter-level to word-level reflow. Words are treated as atomic image crops and placed on the output page as scaled images.

**Without `--lang`**: Words that overflow the line width move to the next line intact (no splitting).

**With `--lang`**: Overflowing words are split at grammatically correct hyphenation points:
1. The word crop is OCR'd with Tesseract (PSM 8)
2. pyphen finds valid hyphenation positions for the detected language
3. The rightmost position that fits the available width is chosen
4. A synthesized hyphen is appended to the first part
5. Fallback: if no letter fits, the word moves to the next line intact

Supported language codes: `ru` (Russian), `en` (English), `sv` (Swedish), and any language with a pyphen dictionary and Tesseract traineddata file.

**Tesseract language setup:**
```bash
# Install via apt
sudo apt install tesseract-ocr-rus tesseract-ocr-swe

# Or copy .traineddata files manually
sudo cp rus.traineddata /usr/share/tessdata/
```

### GPU Acceleration (CUDA)

Neural network models automatically use GPU when available:

```bash
# Verify CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU during processing
watch -n 1 nvidia-smi
```

**System Requirements for GPU:**
- NVIDIA GPU with CUDA support
- 2GB+ GPU memory recommended
- No configuration needed — GPU is used automatically

### Using in Jupyter Notebooks

```python
from ocr_reflow import process_document
import cv2
from matplotlib import pyplot as plt

result = process_document("document.png")

plt.figure(figsize=(12, 16))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cv2.imwrite("output.png", result)
```

See `notebooks/example_usage.ipynb` for a complete tutorial.

### Python Script Usage

```python
from ocr_reflow import process_document
import cv2

reflowed_page = process_document("your_document.png")
cv2.imwrite("reflowed_output.png", reflowed_page)
```

### Advanced Usage — Direct Module Access

```python
from ocr_reflow.main import process_document
import cv2

reflowed_page = process_document(
    "your_document.png",
    zoom_factor=1.5,
    new_page_width=800,
)

cv2.imwrite('output.png', reflowed_page)
```

## Features in Detail

### 1. Word-Level Reflow

The `--word-reflow` mode treats each word as an atomic image crop rather than assembling it letter by letter. This produces more natural results for:
- Scripts with complex ligatures
- Documents with variable letter spacing
- Cases where letter-level detection is imperfect

Word crops are scaled by `--zoom-factor` and placed left-to-right. When a word does not fit the remaining line width, it either moves to the next line (without `--lang`) or is split at a hyphenation point (with `--lang`).

### 2. Hyphenation and Word Splitting

With `--lang`, the split algorithm:
1. OCRs the word crop with Tesseract PSM 8 (single word mode) with an 8px white border for accuracy
2. Strips punctuation and queries pyphen for valid hyphenation positions
3. Maps character positions to pixel positions using letter-index approximation
4. Picks the rightmost cut that fits the available width
5. Synthesizes a hyphen glyph matching the word's font metrics
6. Falls back to moving the whole word to the next line if no cut fits

### 3. Word Split Prevention (Letter-Level Mode)

In letter-level mode, the system prevents awkward splits where only 1 character remains on a line:

- Before: `"fact"` splits as `"f"` on one line, `"act"` on next
- After: `"fact"` moves entirely to new line

This applies to both text and numbers (e.g., `"1950"` won't split as `"195"` + `"0"`).

### 4. Robust Line Spacing

Uses percentile-based calculations to handle outlier letters with incorrect baseline values:

- Calculates 95th percentile instead of maximum
- Applies safety cap at 2.5× typical letter height
- Prevents one bad character from ruining all line spacing

**Example output when outlier detected:**
```
[Line Spacing] Capping line height from 338 to 90 (detected outlier)
```

### 5. Paragraph Detection

Automatically detects paragraph breaks by analyzing:
- Horizontal indentation of first letters
- Short lines (lines significantly shorter than average)
- Preserves paragraph structure in reflowed output
- Applies book-style indentation (~3.5 character widths)

### 6. Baseline-Aware Placement

Each letter/word is placed with its baseline aligned correctly:
- Maintains proper vertical alignment
- Handles descenders (g, j, p, q, y) and ascenders (b, d, f, h, k, l, t)
- Ensures consistent text appearance

## Running Tests

```bash
# Test number splitting prevention
python tests/test_1950.py

# Test line spacing with outliers
python tests/test_outlier_spacing.py

# Test word split prevention
python tests/test_midword_fact.py

# Run all tests
pixi run python -m pytest tests/
```

## Development

### Code Formatting

```bash
pixi run black src/
```

### Adding Dependencies

```bash
# Add conda dependency
pixi add package-name

# Add PyPI dependency
pixi add --pypi package-name
```

## Pixi Environments

### CPU Environment (Default)
- **Use case:** Systems without NVIDIA GPU, Windows WSL, or when CUDA is not needed
- **Installation:** `pixi install` or `pixi install -e default`

### GPU Environment
- **Use case:** Systems with NVIDIA GPU and CUDA 12 drivers
- **Installation:** `pixi install -e gpu`

### Switching Environments

```bash
# Switch to CPU environment
pixi install -e default && pixi shell

# Switch to GPU environment (requires CUDA)
pixi install -e gpu && pixi shell
```

## Troubleshooting

### Line Spacing Too Large

Look for this output — it means the system detected and corrected an outlier:
```
[Line Spacing] Capping line height from X to Y (detected outlier)
```

### CUDA Errors

```bash
python -c "import torch; print(torch.cuda.is_available())"

# Fall back to CPU environment
pixi install -e default
```

### Import Errors

Make sure you're in the Pixi environment:
```bash
pixi shell
python src/ocr_reflow/main.py --help
```

### Tesseract Not Found (--lang)

```bash
# Install Tesseract
sudo apt install tesseract-ocr

# Install language packs
sudo apt install tesseract-ocr-rus tesseract-ocr-swe

# Verify
tesseract --list-langs
```

## Technical Details

### Algorithms Used

1. **Text Detection**: doctr's `detection_predictor` (detection-only, no text strings)
2. **Line Grouping**: Spatial clustering based on y-coordinates
3. **Word Boundary Detection**: Space analysis (threshold at 0.5× avg character width)
4. **Paragraph Detection**: Horizontal indentation analysis + short line detection
5. **Spacing Calculation**: 95th percentile with safety cap at 2.5× letter height
6. **Hyphenation**: pyphen dictionary lookup + Tesseract PSM 8 OCR + pixel-position mapping
7. **Layout Analysis**: DocLayout-YOLO for block classification (title, text, figure, table, formula)
8. **TOC Detection**: Fine-tuned LayoutLMv3 (default), rule-based, or MTD

### Key Data Structures

**`Word` dataclass** (`reflow_words.py`):
- `xmin, ymin, xmax, ymax`: Bounding box in original image coordinates
- `bl`: Baseline offset from bottom of bounding box
- `above`: pixels above baseline

**`Letter` dataclass** (`reflow.py`):
- `xmin, ymin, xmax, ymax`: Bounding box coordinates
- `bl`: Baseline offset from bottom of bounding box

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

Sergey Mikhno <sergey.mikhno@gmail.com>

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 0.2.0 (May 2026)
- Word-level reflow (`--word-reflow`) with whole-word image crops as atomic units
- Hyphenation support (`--lang`) using pyphen + Tesseract OCR for grammatical word splitting
- Binarization pre-processing (`--bin`) for improved diacritic detection
- DjVu and PDF input support via PyMuPDF at 300 DPI
- Multi-page document support (`--page N`)
- `zoom-factor` controls letter size independently of `page-width`
- Skew-aware per-word baseline detection
- Hyphen continuation across line breaks
- Word segmentation visualization (`--show-words`)
- Russian, English, Swedish language support for hyphenation

### Version 0.1.0 (January 2026)
- Initial release
- Text detection and character extraction
- Smart word wrapping with split prevention
- Robust line spacing calculation
- Paragraph detection and preservation
- Background color preservation

---

**Need help?** Open an issue or contact the author.
