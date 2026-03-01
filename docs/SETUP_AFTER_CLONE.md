# Quick Setup Guide After Git Clone

Follow these steps to set up the project after cloning from git.

## 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd segmentation
```

## 2️⃣ Install Pixi (if not already installed)

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

Restart your terminal after installation.

## 3️⃣ Install Dependencies

```bash
# Install all project dependencies
pixi install

# This installs:
# - Python 3.12
# - PyTorch + torchvision
# - doctr (OCR)
# - All other required packages
```

## 4️⃣ Download the LayoutLMv3 Model

The LayoutLMv3 TOC detection model (~500 MB) is hosted on HuggingFace and needs to be downloaded.

### Option A: Automatic Download (Easiest) ⭐

Just run the program - the model downloads automatically on first use:

```bash
pixi run python src/ocr_reflow/main.py images/mh_p005.png --layout
```

On first run, you'll see:
```
Downloading LayoutLMv3 TOC model from HuggingFace (~500 MB)...
This is a one-time download and may take 2-5 minutes...
✓ Model downloaded to: ~/.cache/ocr_reflow/models/layoutlmv3_toc/best_model
```

### Option B: Manual Download

If you want to download the model before running:

```bash
# Step 1: Install huggingface-hub (not available in conda)
pixi run pip install huggingface-hub

# Step 2: Download the model
pixi run python -c "
from huggingface_hub import snapshot_download
from pathlib import Path

model_path = Path('models/layoutlmv3_toc/best_model')
model_path.parent.mkdir(parents=True, exist_ok=True)

print('Downloading LayoutLMv3 model (~500 MB)...')
snapshot_download(
    repo_id='YOUR_USERNAME/layoutlmv3-toc-detector',
    local_dir=str(model_path)
)
print('✅ Model downloaded successfully!')
"
```

### Option C: Using HuggingFace CLI

```bash
# Step 1: Install huggingface-hub
pixi run pip install huggingface-hub

# Step 2: Download using hf CLI
pixi run hf hub download YOUR_USERNAME/layoutlmv3-toc-detector \
    --local-dir models/layoutlmv3_toc/best_model/ \
    --repo-type model
```

**Note**: `huggingface-hub` is not available in conda channels, so we use `pixi run pip install` to install it within the pixi environment.

## 5️⃣ Verify Installation

Check that all models are available:

```bash
pixi run python src/ocr_reflow/model_manager.py info
```

You should see:
```
✓ doclayout_yolo: 38.8 MB
✓ layoutlmv3_toc: 483.8 MB
✓ doctr: 220.4 MB (auto-downloaded)
```

## 6️⃣ Test the Installation

```bash
# Test basic OCR reflow
pixi run python src/ocr_reflow/main.py images/kf_p015.png

# Test with layout analysis
pixi run python src/ocr_reflow/main.py images/mh_p005.png --layout

# Test TOC detection
pixi run python src/ocr_reflow/main.py images/mh_p005.png --layout --toc-algorithm layoutlm
```

## ✅ You're Ready!

The project is now set up and ready to use.

### Quick Reference

**Run OCR reflow:**
```bash
pixi run python src/ocr_reflow/main.py input.png
```

**With layout analysis:**
```bash
pixi run python src/ocr_reflow/main.py input.png --layout
```

**Check models:**
```bash
pixi run python src/ocr_reflow/model_manager.py info
```

**Use in Jupyter:**
```bash
pixi run jupyter lab notebooks/example_usage.ipynb
```

---

## Troubleshooting

### Model not downloading?

**Check internet connection:**
```bash
ping huggingface.co
```

**Manually install huggingface-hub and try again:**
```bash
pixi run pip install huggingface-hub
pixi run python -c "from huggingface_hub import snapshot_download; snapshot_download('YOUR_USERNAME/layoutlmv3-toc-detector', local_dir='models/layoutlmv3_toc/best_model')"
```

### Import errors?

**Reinstall dependencies:**
```bash
pixi install
```

**Or clean and reinstall:**
```bash
rm -rf .pixi
pixi install
```

### Commands not working?

**Make sure you're using `pixi run` prefix:**
```bash
# ✅ Correct
pixi run python src/ocr_reflow/main.py image.png

# ❌ Wrong (unless in pixi shell)
python src/ocr_reflow/main.py image.png
```

**Or activate pixi shell:**
```bash
pixi shell
# Now you can run commands without 'pixi run' prefix
python src/ocr_reflow/main.py image.png
```

### Need help?

See the main README.md or:
- `docs/INSTALL.md` - Detailed installation guide
- `docs/JUPYTER_GUIDE.md` - Using in Jupyter
- `HUGGINGFACE_UPLOAD_GUIDE.md` - Model hosting details

---

**Total setup time: ~5-10 minutes** (including model download)
