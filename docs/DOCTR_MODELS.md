# DocTR Models - No Action Needed! ✨

## Quick Answer: **Don't Store DocTR Models in Your Package**

DocTR (Document Text Recognition) manages its own models automatically. Here's why:

## How DocTR Works

### 1. **Automatic Download**
When you first use doctr:
```python
from doctr.models import ocr_predictor
model = ocr_predictor(pretrained=True)  # Downloads automatically on first use
```

### 2. **Automatic Caching**
Models are cached in `~/.cache/doctr/models/`:
```
~/.cache/doctr/models/
├── db_resnet50-ac60cadc.pt.zip      # Text detection model
├── crnn_vgg16_bn-9762b0b0.pt.zip    # Text recognition model
└── ...
```

### 3. **Automatic Version Management**
- DocTR library handles updates
- Checksums verify model integrity
- No manual intervention needed

## Where DocTR Models Live

| Model Type | Purpose | Location | Size | Managed By |
|-----------|---------|----------|------|------------|
| DocLayout-YOLO | Layout detection | `models/` | 39 MB | **Your package** |
| LayoutLMv3 TOC | TOC detection | `models/` | 484 MB | **Your package** |
| DocTR DB-ResNet | Text detection | `~/.cache/doctr/` | ~50 MB | **doctr library** ✨ |
| DocTR CRNN | Text recognition | `~/.cache/doctr/` | ~170 MB | **doctr library** ✨ |

**Total managed by you**: ~523 MB  
**Total managed by doctr**: ~220 MB (auto-downloaded)

## Why NOT Store DocTR Models

✅ **Already Optimized**: doctr handles caching perfectly  
✅ **Version Management**: Library manages updates automatically  
✅ **Cross-Platform**: Works on Linux, macOS, Windows  
✅ **Shared Cache**: All doctr projects share the same cache  
✅ **Industry Standard**: This is how PyTorch, TensorFlow, etc. work  

## Your Model Manager Integration

Your `model_manager.py` now **displays** DocTR info, but doesn't manage it:

```bash
$ python src/ocr_reflow/model_manager.py info

Models directory: /path/to/segmentation/models

Installed models:
  ✓ doclayout_yolo: 38.8 MB (managed by: ocr_reflow)
  ✓ layoutlmv3_toc: 483.8 MB (managed by: ocr_reflow)
  ✓ doctr: 220.4 MB (managed by: doctr library (auto-downloaded))
    Path: /home/user/.cache/doctr/models
    Models: 3 file(s)
    Note: These models are automatically managed by the doctr library
```

This is **informational only** - you don't need to do anything with DocTR models!

## First-Time User Experience

When a user runs your package for the first time:

```bash
# User installs your package
pip install ocr-reflow

# First run
python -m ocr_reflow.main image.png --layout
```

**What happens**:
1. ✅ DocLayout-YOLO loads from `models/` (you provide)
2. ✅ LayoutLMv3 loads from `models/` (you provide)
3. ✨ DocTR downloads its models to `~/.cache/doctr/` (automatic)
   - Shows progress: "Downloading db_resnet50..."
   - Takes ~30 seconds on first run
   - Never downloads again (cached)

## For Package Distribution

### Don't Include DocTR Models Because:

1. **Redundant**: Users who have doctr already have them cached
2. **Size**: Would add 220 MB to your package (unnecessary)
3. **Conflicts**: Could conflict with doctr's version management
4. **Platform-specific**: DocTR handles platform differences

### Your Package Only Needs:

```python
# setup.py
install_requires=[
    'python-doctr[torch]>=0.6.0',  # This is enough!
]
```

DocTR handles the rest automatically!

## Advanced: Checking DocTR Cache

```python
from pathlib import Path

# Check if DocTR models are cached
doctr_cache = Path.home() / ".cache" / "doctr" / "models"
if doctr_cache.exists():
    models = list(doctr_cache.glob("*.zip")) + list(doctr_cache.glob("*.pt"))
    print(f"DocTR has {len(models)} cached models")
    
    total_size = sum(f.stat().st_size for f in models)
    print(f"Total size: {total_size / (1024**2):.1f} MB")
```

## Troubleshooting

### Q: DocTR models fail to download

**A**: Check internet connection and firewall. DocTR downloads from HuggingFace/GitHub.

### Q: Where are DocTR models stored on Windows?

**A**: `C:\Users\<username>\.cache\doctr\models\`

### Q: Can I pre-download DocTR models?

**A**: Yes, but not recommended. Just let doctr handle it:
```python
# This triggers download
from doctr.models import ocr_predictor
model = ocr_predictor(pretrained=True)  # Will download if needed
```

### Q: Can I change DocTR cache location?

**A**: Yes, set environment variable:
```bash
export DOCTR_CACHE_DIR=/custom/path
```

### Q: Do I need to document DocTR in my README?

**A**: Optional, but helpful:
```markdown
## First Run

On first use, DocTR will automatically download text detection models (~220 MB).
This is a one-time download and takes about 30 seconds.
Models are cached in `~/.cache/doctr/` for future use.
```

## Comparison with Your Models

| Aspect | Your Models (YOLO, LayoutLMv3) | DocTR Models |
|--------|-------------------------------|--------------|
| **Storage** | `models/` in your repo | `~/.cache/doctr/` |
| **Management** | Your `model_manager.py` | doctr library |
| **Download** | Manual or on install | Auto on first use |
| **Updates** | You control | doctr controls |
| **Versioning** | Your choice | doctr chooses |
| **Git tracking** | Your choice | Never tracked |
| **Distribution** | Include or host separately | Never included |

## Best Practice: Document But Don't Manage

✅ **DO**:
- Mention DocTR in documentation
- Show it in `model_manager.py info` (informational)
- List it as a dependency in `setup.py`

❌ **DON'T**:
- Copy DocTR models to your `models/` folder
- Try to manage DocTR's cache
- Include DocTR models in your package
- Re-download or redistribute DocTR models

## Example: Other Libraries That Work This Way

- **PyTorch**: Models in `~/.cache/torch/hub/`
- **Transformers**: Models in `~/.cache/huggingface/`
- **torchvision**: Models in `~/.cache/torch/hub/checkpoints/`
- **doctr**: Models in `~/.cache/doctr/` ← You're here!

This is the **standard pattern** - don't fight it, embrace it! ✨

## Summary

### ✅ What We Did
- Added DocTR info to `model_manager.py info` command
- Updated documentation to explain DocTR is auto-managed
- Made it clear: **No action needed for DocTR!**

### ✨ What You Should Do
- **Nothing!** DocTR manages itself perfectly
- Just list `python-doctr` as a dependency
- Optionally mention in docs that models auto-download on first use

### 🎉 Result
- Your package manages: DocLayout-YOLO + LayoutLMv3 (~523 MB)
- DocTR manages: Its own models (~220 MB)
- Total system models: ~743 MB
- Everything works automatically! ✨

---

**Bottom Line**: DocTR models are like Python packages - they install themselves when needed. You don't package Python with your application, and you don't package DocTR models either! 🚀
