# Models Directory

This directory contains all machine learning models used by the OCR Reflow package.

## Directory Structure

```
models/
├── README.md                                          # This file
├── doclayout_yolo_docstructbench_imgsz1024.pt        # Layout detection model (YOLO)
└── layoutlmv3_toc/                                    # TOC detection model (LayoutLMv3)
    └── best_model/
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        └── ...

# DocTR models are stored separately (managed by doctr library):
~/.cache/doctr/models/                                 # Text detection/OCR models
├── db_resnet50-*.zip                                  # Text detection
├── crnn_vgg16_bn-*.zip                               # Text recognition
└── ...
```

**Note**: DocTR models (~220 MB) are automatically downloaded and managed by the doctr library.
They are stored in `~/.cache/doctr/models/` and don't need to be included in this directory.
See `docs/DOCTR_MODELS.md` for details.

## Models

### 1. DocLayout-YOLO (`doclayout_yolo_docstructbench_imgsz1024.pt`)

**Purpose**: Document layout analysis - detects different regions in document images

**Type**: YOLOv10 object detection model

**Source**: https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench

**Size**: ~50 MB

**Detects**:
- Title blocks
- Plain text regions
- Figures and figure captions
- Tables and table captions
- Formulas and formula captions
- Abandoned/header/footer regions

**Download**:
```bash
# Option 1: Using huggingface-cli
pip install huggingface-hub
huggingface-cli download juliozhao/DocLayout-YOLO-DocStructBench \
    doclayout_yolo_docstructbench_imgsz1024.pt \
    --local-dir models/

# Option 2: Manual download
# Visit: https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench
# Download: doclayout_yolo_docstructbench_imgsz1024.pt
# Place in: models/
```

**Usage in code**:
```python
from model_manager import get_doclayout_yolo_path
model_path = get_doclayout_yolo_path()
```

---

### 2. Fine-tuned LayoutLMv3 (`layoutlmv3_toc/best_model/`)

**Purpose**: Table of Contents (TOC) detection - binary classification (TOC vs non-TOC)

**Type**: LayoutLMv3 sequence classification model (Microsoft/layoutlmv3-base fine-tuned)

**Training**: Fine-tuned on our custom dataset

**Size**: ~504 MB

**Performance**:
- Overall accuracy: 88.2% (30/34 correct)
- TOC detection: 82.4% (14/17 correct)
- Non-TOC detection: 94.1% (16/17 correct)
- 3.1x faster than rule-based approach

**Training data**: 34 pages (17 TOC + 17 non-TOC)

**Train the model**:
```bash
# This will create models/layoutlmv3_toc/best_model/
python train_layoutlmv3.py
```

**Training takes**: ~10-15 minutes on GPU (NVIDIA RTX 3050)

**Usage in code**:
```python
from model_manager import get_layoutlmv3_toc_path
model_path = get_layoutlmv3_toc_path()
```

---

### 3. DocTR Models (auto-downloaded)

**Purpose**: Text detection and recognition (OCR)

**Type**: Various deep learning models (DBNet, CRNN, etc.)

**Source**: https://github.com/mindee/doctr

**Cache location**: `~/.cache/doctr/models/` (managed by doctr library)

**Size**: ~100-200 MB (downloaded on first use)

**Note**: These models are automatically downloaded by the doctr library on first use. No manual download needed.

---

## Model Management

### Check installed models

```bash
python src/ocr_reflow/model_manager.py info
```

Output:
```
================================================================================
OCR REFLOW MODEL INFORMATION
================================================================================

Models directory: /path/to/project/models

Installed models:
  ✓ doclayout_yolo: 50.2 MB
    Path: /path/to/project/models/doclayout_yolo_docstructbench_imgsz1024.pt
  ✓ layoutlmv3_toc: 504.3 MB
    Path: /path/to/project/models/layoutlmv3_toc/best_model
================================================================================
```

### Download missing models

```bash
python src/ocr_reflow/model_manager.py download
```

### Verify all models

```bash
python src/ocr_reflow/model_manager.py check
```

---

## For Package Distribution

### DON'T include models in pip package

Models are too large for PyPI. Instead:

1. **User downloads on first run**: Models auto-download when needed
2. **Separate hosting**: Host models on HuggingFace Hub, GitHub Releases, or CDN
3. **Cache locally**: Store in `~/.cache/ocr_reflow/models/` or project `models/`

### Setup.py example

```python
setup(
    name="ocr-reflow",
    # ... other config ...
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "python-doctr>=0.6.0",
        "huggingface-hub>=0.16.0",  # For model downloads
    ],
)
```

### First-run experience

```python
# User installs
pip install ocr-reflow

# On first use, models auto-download
from ocr_reflow import process_document
result = process_document("image.png")  # Auto-downloads models (~600 MB)
```

---

## .gitignore

Add to `.gitignore` to avoid committing large model files:

```gitignore
# Large model files
models/*.pt
models/layoutlmv3_toc/best_model/
!models/README.md
```

But for development, you CAN commit them if:
- Using Git LFS (Large File Storage)
- Repository is private/internal
- Team needs quick setup

---

## Git LFS (Optional)

If you want to version control model files:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.pt"
git lfs track "models/layoutlmv3_toc/**"

# Commit
git add .gitattributes
git add models/
git commit -m "Add models with Git LFS"
```

---

## License

- **DocLayout-YOLO**: Check source repository license
- **LayoutLMv3 (base)**: Apache 2.0 (Microsoft)
- **Our fine-tuned model**: Same as your project license
- **DocTR**: Apache 2.0

---

## Updates

### Updating models

```bash
# Re-train TOC detector
python train_layoutlmv3.py

# Download latest DocLayout-YOLO
python src/ocr_reflow/model_manager.py download
```

### Model versioning

Consider versioning models:
```
models/
├── doclayout_yolo_v1.0.pt
├── doclayout_yolo_v1.1.pt  # Latest
└── layoutlmv3_toc/
    ├── v1.0_26pages/
    └── v1.1_34pages/  # Latest (best_model -> symlink here)
```

---

## Troubleshooting

### Model not found errors

```python
FileNotFoundError: DocLayout-YOLO model not found
```

**Solution**: Run `python src/ocr_reflow/model_manager.py download`

### Out of disk space

Models require ~600 MB total. Check available space:
```bash
df -h
```

### GPU memory errors

If models don't fit in GPU memory:
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""`
- Or reduce batch size in training

---

For more information, see the main README.md
