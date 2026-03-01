#!/usr/bin/env python3
"""
Upload LayoutLMv3 TOC model to HuggingFace Hub

This script uploads the fine-tuned LayoutLMv3 model to HuggingFace Hub.

Prerequisites:
1. Install huggingface-hub: pip install huggingface-hub
2. Login: huggingface-cli login
3. Train the model: python train_layoutlmv3.py

Usage:
    python upload_to_huggingface.py [--repo-id USERNAME/REPO_NAME]
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        logger.error("huggingface_hub not installed!")
        logger.error("Install with: pip install huggingface-hub")
        return False

def check_model_exists(model_path):
    """Check if the model files exist."""
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model directory not found: {model_path}")
        logger.error("Train the model first: python train_layoutlmv3.py")
        return False

    # Check for essential files
    required_files = [
        "config.json",
        "preprocessor_config.json",
    ]

    # Check for model weights (either .bin or .safetensors)
    has_weights = (
        (model_path / "pytorch_model.bin").exists() or
        (model_path / "model.safetensors").exists()
    )

    if not has_weights:
        logger.error("Model weights not found (pytorch_model.bin or model.safetensors)")
        logger.error("Train the model first: python train_layoutlmv3.py")
        return False

    missing_files = [f for f in required_files if not (model_path / f).exists()]
    if missing_files:
        logger.warning(f"Some files missing: {missing_files}")
        logger.warning("Model may not work correctly on HuggingFace")

    return True

def prepare_model_card(model_path, model_card_path):
    """Copy model card to model directory."""
    model_path = Path(model_path)
    readme_path = model_path / "README.md"

    if model_card_path and Path(model_card_path).exists():
        logger.info(f"Copying model card from {model_card_path}")
        shutil.copy(model_card_path, readme_path)
    else:
        logger.warning("No model card found, creating basic README")
        with open(readme_path, 'w') as f:
            f.write("""# LayoutLMv3 TOC Detector

Fine-tuned LayoutLMv3 model for detecting Table of Contents pages.

## Usage

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

model = LayoutLMv3ForSequenceClassification.from_pretrained("YOUR_USERNAME/layoutlmv3-toc-detector")
processor = LayoutLMv3Processor.from_pretrained("YOUR_USERNAME/layoutlmv3-toc-detector")
```

For detailed usage, see: https://github.com/ssppkenny/segmentation
""")

def upload_model(model_path, repo_id, commit_message="Upload fine-tuned LayoutLMv3 TOC detector"):
    """Upload model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()

        # Check if user is logged in
        try:
            user = api.whoami()
            logger.info(f"Logged in as: {user['name']}")
        except Exception as e:
            logger.error("Not logged in to HuggingFace!")
            logger.error("Run: huggingface-cli login")
            return False

        # Create repo if it doesn't exist
        try:
            logger.info(f"Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            logger.info(f"✓ Repository created/exists: https://huggingface.co/{repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return False

        # Upload all files from model directory
        logger.info(f"Uploading model files from {model_path}...")
        logger.info("This may take a few minutes (model is ~500 MB)...")

        try:
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            logger.info(f"✓ Model uploaded successfully!")
            logger.info(f"View at: https://huggingface.co/{repo_id}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload LayoutLMv3 TOC model to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ssppkenny/layoutlmv3-toc-detector",
        help="HuggingFace repo ID (format: username/repo-name)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/layoutlmv3_toc/best_model",
        help="Path to model directory"
    )
    parser.add_argument(
        "--model-card",
        type=str,
        default="models/layoutlmv3_toc/MODEL_CARD.md",
        help="Path to model card (README.md)"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload fine-tuned LayoutLMv3 TOC detector (88.2% accuracy)",
        help="Commit message for the upload"
    )

    args = parser.parse_args()

    print("="*80)
    print("HUGGINGFACE MODEL UPLOAD")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Repository: {args.repo_id}")
    print("="*80)
    print()

    # Step 1: Check huggingface_hub
    logger.info("Step 1: Checking huggingface_hub installation...")
    if not check_huggingface_hub():
        return 1
    logger.info("✓ huggingface_hub is installed")
    print()

    # Step 2: Check model exists
    logger.info("Step 2: Checking model files...")
    if not check_model_exists(args.model_path):
        return 1
    logger.info("✓ Model files found")
    print()

    # Step 3: Prepare model card
    logger.info("Step 3: Preparing model card...")
    prepare_model_card(args.model_path, args.model_card)
    logger.info("✓ Model card ready")
    print()

    # Step 4: Upload
    logger.info("Step 4: Uploading to HuggingFace Hub...")
    if not upload_model(args.model_path, args.repo_id, args.commit_message):
        return 1

    print()
    print("="*80)
    print("✓ SUCCESS! Model uploaded to HuggingFace Hub")
    print("="*80)
    print(f"View your model at: https://huggingface.co/{args.repo_id}")
    print()
    print("Next steps:")
    print("1. Visit your model page and verify files uploaded correctly")
    print("2. Edit the README if needed")
    print("3. Test loading the model:")
    print(f'   from transformers import AutoModelForSequenceClassification')
    print(f'   model = AutoModelForSequenceClassification.from_pretrained("{args.repo_id}")')
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
