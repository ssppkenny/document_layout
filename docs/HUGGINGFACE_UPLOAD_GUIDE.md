# 🚀 Publishing Your LayoutLMv3 Model to HuggingFace

## ✅ Your Model is Ready!

Your LayoutLMv3 TOC detector has been trained successfully:
- **Validation Accuracy**: 85.71%
- **Model Location**: `models/layoutlmv3_toc/best_model/`
- **Model Size**: ~500 MB
- **Files Ready**: 
  - ✅ `model.safetensors` (model weights)
  - ✅ `config.json` (model configuration)
  - ✅ `processor_config.json` (processor config)
  - ✅ `tokenizer.json` + `tokenizer_config.json` (tokenizer)
  - ✅ `MODEL_CARD.md` (documentation for HuggingFace)

## 📝 Step-by-Step Upload Instructions

### Step 1: Create HuggingFace Account (if you don't have one)

1. Go to https://huggingface.co/join
2. Sign up with email/GitHub
3. Verify your email

### Step 2: Login to HuggingFace CLI

```bash
# In your terminal
pixi run hf auth login
```

It will ask for your **User Access Token**:
- Go to https://huggingface.co/settings/tokens
- Click "New token"
- Name it "model-upload" or similar
- Select "Write" permission
- Copy the token
- Paste it in the terminal (it won't show characters - that's normal!)

### Step 3: Upload Your Model

**Option A: Using the Upload Script (Recommended)**

```bash
# Run the upload script
python upload_to_huggingface.py

# Or specify custom repo name:
python upload_to_huggingface.py --repo-id YOUR_USERNAME/layoutlmv3-toc-detector
```

The script will:
1. ✅ Check all files are present
2. ✅ Create the repository on HuggingFace
3. ✅ Upload all model files (~500 MB)
4. ✅ Add the model card (README.md)

**Uploading will take 2-5 minutes** depending on your internet speed.

**Option B: Manual Upload (Alternative)**

```bash
# Install huggingface-hub (if not already installed)
pixi run pip install huggingface-hub

# Login (if not already)
pixi run hf auth login

# Upload using CLI
pixi run hf upload YOUR_USERNAME/layoutlmv3-toc-detector \
    models/layoutlmv3_toc/best_model/ \
    --repo-type model
```

### Step 4: Verify Upload

After upload completes:

1. **Visit your model page**: https://huggingface.co/YOUR_USERNAME/layoutlmv3-toc-detector

2. **Check files uploaded**:
   - ✅ model.safetensors (~500 MB)
   - ✅ config.json
   - ✅ processor_config.json
   - ✅ tokenizer files
   - ✅ README.md (model card)

3. **Edit README if needed**:
   - Click "Edit model card" on HuggingFace
   - Update your username, links, etc.
   - Save

### Step 5: Test Your Uploaded Model

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# Load from HuggingFace (downloads automatically)
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "YOUR_USERNAME/layoutlmv3-toc-detector"
)
processor = LayoutLMv3Processor.from_pretrained(
    "YOUR_USERNAME/layoutlmv3-toc-detector"
)

print("✅ Model loaded successfully from HuggingFace!")
```

## 🔧 Update Your Code to Use HuggingFace Model

### Option 1: Update `model_manager.py` (Recommended)

I've already prepared the code! Just update the repo ID:

```python
# In src/ocr_reflow/model_manager.py (around line 120)
def get_layoutlmv3_toc_path():
    """Get LayoutLMv3 model path, downloading from HuggingFace if needed."""
    from huggingface_hub import snapshot_download
    
    cache_dir = Path.home() / ".cache" / "ocr_reflow" / "models"
    model_path = cache_dir / "layoutlmv3_toc" / "best_model"
    
    if not model_path.exists():
        logger.info("Downloading LayoutLMv3 TOC model from HuggingFace...")
        snapshot_download(
            repo_id="YOUR_USERNAME/layoutlmv3-toc-detector",  # ← Update this!
            local_dir=model_path,
            cache_dir=cache_dir
        )
    
    return str(model_path)
```

### Option 2: Direct Download in `layoutlm_toc_detector.py`

```python
# Update layoutlm_toc_detector.py
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

# Load directly from HuggingFace
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "YOUR_USERNAME/layoutlmv3-toc-detector"
)
processor = LayoutLMv3Processor.from_pretrained(
    "YOUR_USERNAME/layoutlmv3-toc-detector"
)
```

## 📦 Package Distribution

After uploading to HuggingFace, your package users will:

```bash
# Install your package
pip install ocr-reflow

# On first run, model downloads automatically from HuggingFace
python -m ocr_reflow.main image.png --layout
# → Downloads model from HuggingFace (~500 MB, one-time)
# → Cached in ~/.cache/huggingface/ or ~/.cache/ocr_reflow/
```

## 🎯 Benefits of HuggingFace

✅ **Free hosting** - No costs for model storage or downloads
✅ **Unlimited bandwidth** - No download limits  
✅ **Version control** - Update model, users get latest version  
✅ **Professional** - Industry standard (PyTorch, OpenAI, etc. use it)  
✅ **Discoverable** - Others can find and cite your model  
✅ **Easy integration** - Works with `transformers` library  
✅ **Auto-caching** - Downloads once, cached forever  

## 📊 Model Performance Summary

To add to your HuggingFace model card:

```markdown
## Performance

- **Overall Accuracy**: 85.71%
- **Training Time**: ~5 epochs (early stopping)
- **Dataset**: 34 pages (27 train + 7 validation)
- **Hardware**: NVIDIA RTX 3050 4GB
- **Comparison**: 
  - Rule-based: 85.3% accuracy, 17.7s per page
  - This model: 85.71% accuracy, 3.1s per page (✅ 5.7x faster)
```

## 🐛 Troubleshooting

### Error: "Not logged in"
```bash
pixi run hf auth login
```
Get token from: https://huggingface.co/settings/tokens

### Error: ImportError with HF_HUB_ENABLE_HF_TRANSFER
This is a version compatibility issue. Fix with:
```bash
pixi run pip install huggingface-hub
```

Or if already installed:
```bash
pixi run pip install --upgrade huggingface-hub
```

### Error: "Repository not found"
- Check repo name matches your username
- Create repo first on HuggingFace website if needed

### Upload is slow
- Normal! 500 MB takes 2-5 minutes on typical connection
- Progress bar will show upload status

### Model card not showing correctly
- Edit on HuggingFace website: Click "Edit model card"
- Or update `MODEL_CARD.md` and re-upload

## 🎉 Next Steps

After successful upload:

1. ✅ Share your model: https://huggingface.co/YOUR_USERNAME/layoutlmv3-toc-detector
2. ✅ Update your README.md to reference the HuggingFace model
3. ✅ Test downloading and using the model
4. ✅ Update your package to download from HuggingFace

## 📝 Example README Update

Add to your main README.md:

```markdown
## Models

This package uses a fine-tuned LayoutLMv3 model for TOC detection:

- **Model**: [layoutlmv3-toc-detector](https://huggingface.co/YOUR_USERNAME/layoutlmv3-toc-detector)
- **Accuracy**: 85.71%
- **Auto-downloads** on first use (~500 MB)

```

---

**Ready to upload?** Run:
```bash
python upload_to_huggingface.py
```

Good luck! 🚀
