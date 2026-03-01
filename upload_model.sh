#!/bin/bash
# Upload LayoutLMv3 TOC Model to HuggingFace Hub
#
# This script uploads the fine-tuned model with 100% accuracy (54 pages training)

set -e

echo "================================================================================"
echo "HUGGINGFACE MODEL UPLOAD - LayoutLMv3 TOC Detector"
echo "================================================================================"
echo ""
echo "Model Details:"
echo "  - Validation Accuracy: 100.00%"
echo "  - Training Dataset: 54 pages (27 TOC + 27 non-TOC)"
echo "  - Model Size: ~484 MB"
echo "  - Training Date: February 21, 2026"
echo ""
echo "================================================================================"
echo ""

# Step 1: Check if logged in
echo "Step 1: Checking HuggingFace login status..."
if pixi run hf whoami &>/dev/null; then
    USERNAME=$(pixi run hf whoami 2>/dev/null | head -1)
    echo "✓ Logged in as: $USERNAME"
else
    echo ""
    echo "⚠️  You need to login to HuggingFace first!"
    echo ""
    echo "To login:"
    echo "  1. Get your token from: https://huggingface.co/settings/tokens"
    echo "  2. Run: pixi run hf auth login"
    echo "  3. Paste your token when prompted"
    echo ""
    echo "Then run this script again."
    echo ""
    exit 1
fi

# Step 2: Upload model
echo ""
echo "Step 2: Uploading model to HuggingFace..."
echo ""

# Default repo name
REPO_ID="${USERNAME}/layoutlmv3-toc-detector"

# Allow custom repo name
if [ ! -z "$1" ]; then
    REPO_ID="$1"
fi

echo "Repository: $REPO_ID"
echo ""
echo "Uploading model files (~484 MB)..."
echo "This may take 2-5 minutes depending on your connection..."
echo ""

# Upload using hf hub
pixi run hf hub upload "$REPO_ID" \
    models/layoutlmv3_toc/best_model/ \
    --repo-type model \
    --commit-message "Upload fine-tuned LayoutLMv3 TOC detector (100% accuracy on 54 pages)" \
    2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ SUCCESS! Model uploaded to HuggingFace Hub"
    echo "================================================================================"
    echo ""
    echo "View your model at:"
    echo "  https://huggingface.co/$REPO_ID"
    echo ""
    echo "Load the model in Python:"
    echo "  from transformers import LayoutLMv3ForSequenceClassification"
    echo "  model = LayoutLMv3ForSequenceClassification.from_pretrained('$REPO_ID')"
    echo ""
    echo "Next steps:"
    echo "  1. Visit the model page and verify files"
    echo "  2. Edit the README if needed"
    echo "  3. Update model_manager.py with your repo ID"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "❌ Upload failed"
    echo "================================================================================"
    echo ""
    echo "Possible issues:"
    echo "  - Not logged in: Run 'pixi run hf auth login'"
    echo "  - Network issues: Check your internet connection"
    echo "  - Permissions: Ensure your token has write access"
    echo ""
    exit 1
fi
