"""
LayoutLMv3-based Table of Contents Detection

Since the original MTD model weights are not publicly available, this module provides
an alternative using Microsoft's LayoutLMv3, which is a pre-trained model for document
understanding that can be adapted for TOC detection.

LayoutLMv3 is similar to MTD in that it combines:
- Visual features (from image patches)
- Text features (from BERT-like encoder)
- Layout features (bounding box positions)

Paper: "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"
Model: microsoft/layoutlmv3-base
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Detect CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    logger.info(f"✓ CUDA available for LayoutLMv3! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("⚠ CUDA not available for LayoutLMv3. Using CPU")


@dataclass
class TOCEntity:
    """Entity detected in the document"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    is_toc_entry: bool = False
    confidence: float = 0.0


def detect_toc_with_layoutlm(image_path: str, min_toc_entries: int = 4) -> Tuple[bool, float, Dict]:
    """
    Detect TOC using fine-tuned LayoutLMv3 model.

    This uses our fine-tuned LayoutLMv3 model trained specifically for TOC detection
    with 34 pages (17 TOC + 17 non-TOC, achieving 88.2% accuracy).

    Args:
        image_path: Path to document image
        min_toc_entries: Minimum number of TOC entries required (not used with fine-tuned model)

    Returns:
        Tuple of (is_toc, confidence, metadata)
    """
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
        from PIL import Image

        # Import model manager for centralized model path
        try:
            from model_manager import get_layoutlmv3_toc_path
        except ImportError:
            try:
                from .model_manager import get_layoutlmv3_toc_path
            except ImportError:
                # Fallback to hardcoded path
                logger.warning("model_manager not available, using fallback path")
                def get_layoutlmv3_toc_path():
                    from pathlib import Path
                    return str(Path(__file__).parent.parent.parent / "models" / "layoutlmv3_toc" / "best_model")

        # Get model path from model manager
        try:
            model_path = get_layoutlmv3_toc_path()
        except FileNotFoundError as e:
            logger.warning(str(e))
            return False, 0.0, {'reason': 'Fine-tuned model not found. Run: python train_layoutlmv3.py'}

        logger.info("Loading fine-tuned LayoutLMv3 model...")

        # Load fine-tuned model for sequence classification (TOC vs non-TOC)
        processor = LayoutLMv3Processor.from_pretrained(model_path)
        model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()

        logger.info(f"✓ Fine-tuned LayoutLMv3 model loaded on {DEVICE}")

        # Load image
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Extract text with OCR using doctr
        from doctr.models import ocr_predictor
        from doctr.io import DocumentFile

        ocr_model = ocr_predictor(pretrained=True)
        doc = DocumentFile.from_images(image_path)
        ocr_result = ocr_model(doc)

        # Extract words and boxes
        words = []
        boxes = []

        if hasattr(ocr_result, 'export'):
            doc_dict = ocr_result.export()

            for page in doc_dict.get('pages', []):
                for block in page.get('blocks', []):
                    for line in block.get('lines', []):
                        for word_data in line.get('words', []):
                            text = word_data.get('value', '').strip()
                            if text:
                                geometry = word_data.get('geometry', [[0, 0], [1, 1]])
                                if len(geometry) >= 2:
                                    x0 = int(geometry[0][0] * w)
                                    y0 = int(geometry[0][1] * h)
                                    x1 = int(geometry[1][0] * w)
                                    y1 = int(geometry[1][1] * h)
                                    words.append(text)
                                    # LayoutLMv3 expects normalized boxes [0-1000]
                                    boxes.append([
                                        int((x0 / w) * 1000),
                                        int((y0 / h) * 1000),
                                        int((x1 / w) * 1000),
                                        int((y1 / h) * 1000)
                                    ])

        if len(words) == 0:
            return False, 0.0, {'reason': 'No text detected in image'}

        # Prepare input for model
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # Move to device
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

        # Get prediction
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        is_toc = (prediction == 1)

        metadata = {
            'model': 'fine-tuned LayoutLMv3',
            'model_path': model_path,
            'accuracy': '88.2% on 34-page test set',
            'confidence': confidence,
            'prediction': 'TOC' if is_toc else 'NOT TOC'
        }

        return is_toc, confidence, metadata

    except Exception as e:
        logger.error(f"LayoutLMv3 detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0, {'reason': f'Error: {str(e)}'}



# Fallback: Simple detection without LayoutLMv3
def detect_toc_simple(image_path: str, min_toc_entries: int = 4) -> Tuple[bool, float, Dict]:
    """
    Simple TOC detection without deep learning models.
    Falls back to geometric analysis.
    """
    try:
        from doctr.models import detection_predictor
        from doctr.io import DocumentFile

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0, {'error': 'Failed to load image'}

        h, w = img.shape[:2]

        # Use doctr for word detection
        model = detection_predictor(pretrained=True)
        doc = DocumentFile.from_images(image_path)
        result = model(doc)
        words = result[0]["words"]

        # Convert to absolute coordinates
        words[:, 0] = (words[:, 0] * w).astype(np.int32)
        words[:, 1] = (words[:, 1] * h).astype(np.int32)
        words[:, 2] = (words[:, 2] * w).astype(np.int32)
        words[:, 3] = (words[:, 3] * h).astype(np.int32)

        # Check for TOC patterns
        right_aligned_count = 0
        for word_box in words:
            x0, y0, x1, y1 = word_box
            # Check if right-aligned (within 20% of right edge)
            if x1 > w * 0.8:
                right_aligned_count += 1

        # Simple heuristic
        if right_aligned_count >= min_toc_entries:
            confidence = min(0.7, right_aligned_count / len(words))
            is_toc = True
        else:
            confidence = right_aligned_count / min_toc_entries * 0.5
            is_toc = False

        metadata = {
            'model': 'simple_geometric',
            'num_words': len(words),
            'right_aligned_words': right_aligned_count,
            'confidence': confidence
        }

        return is_toc, confidence, metadata

    except Exception as e:
        logger.error(f"Simple detection failed: {e}")
        return False, 0.0, {'error': str(e)}


if __name__ == "__main__":
    print("LayoutLMv3 TOC Detector")
    print("=" * 60)
    print("\nNote: The original MTD model weights are not publicly available.")
    print("This module uses LayoutLMv3 as an alternative.")
    print("\nLayoutLMv3 Model:")
    print("  - Pre-trained on document understanding tasks")
    print("  - Combines vision + text + layout features")
    print("  - Available on Hugging Face: microsoft/layoutlmv3-base")
    print("  - Size: ~133M parameters")
    print("\nTo use:")
    print("  pip install transformers")
    print("  python -c 'from layoutlm_toc_detector import detect_toc_with_layoutlm'")
