# 🚀 Quick Start: Upload to HuggingFace (3 Commands!)

## ✅ Everything is Ready!

Your model has been trained and prepared for HuggingFace. Just 3 commands:

---

## 1️⃣ **Login to HuggingFace**

```bash
pixi run hf auth login
```

- Create account: https://huggingface.co/join (if needed)
- Get token: https://huggingface.co/settings/tokens
- Paste token when prompted

**If you get an ImportError**, fix it with:
```bash
pixi run pip install huggingface-hub
```

Or if already installed:
```bash
pixi run pip install --upgrade huggingface-hub
```

---

## 2️⃣ **Upload Model**

```bash
pixi run python upload_to_huggingface.py
```

- Takes 2-5 minutes (~500 MB upload)
- Creates repository automatically
- Uploads all files + documentation

---

## 3️⃣ **Update Code**

Edit `src/ocr_reflow/model_manager.py` (line ~140):

```python
# Change this line:
repo_id = "ssppkenny/layoutlmv3-toc-detector"

# To your username:
repo_id = "YOUR_USERNAME/layoutlmv3-toc-detector"
```

---

## ✅ Done!

Your model is now published at:
**https://huggingface.co/YOUR_USERNAME/layoutlmv3-toc-detector**

Test it:
```python
from transformers import LayoutLMv3ForSequenceClassification
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "YOUR_USERNAME/layoutlmv3-toc-detector"
)
print("✅ Works!")
```

---

## 📚 Need Help?

See detailed guide: **`HUGGINGFACE_UPLOAD_GUIDE.md`**

---

**That's it!** 🎉
