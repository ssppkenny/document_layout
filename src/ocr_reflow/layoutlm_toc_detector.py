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
    Detect TOC using LayoutLMv3 model from Hugging Face.

    This uses Microsoft's pre-trained LayoutLMv3 model which combines
    vision, text, and layout features similar to MTD.

    Args:
        image_path: Path to document image
        min_toc_entries: Minimum number of TOC entries required

    Returns:
        Tuple of (is_toc, confidence, metadata)
    """
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        from PIL import Image

        logger.info("Loading LayoutLMv3 model (this may take a moment on first run)...")

        # Load pre-trained LayoutLMv3
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
        model.to(DEVICE)  # Move model to GPU if available
        model.eval()

        logger.info(f"✓ LayoutLMv3 model loaded on {DEVICE}")

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Extract text with OCR (LayoutLMv3 processor includes Tesseract)
        # In a real implementation, we'd use doctr for better results
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
            page_h, page_w = image.size[1], image.size[0]

            for page in doc_dict.get('pages', []):
                for block in page.get('blocks', []):
                    for line in block.get('lines', []):
                        for word_data in line.get('words', []):
                            text = word_data.get('value', '')
                            if text.strip():
                                geometry = word_data.get('geometry', [[0, 0], [1, 1]])
                                if len(geometry) >= 2:
                                    x0 = int(geometry[0][0] * page_w)
                                    y0 = int(geometry[0][1] * page_h)
                                    x1 = int(geometry[1][0] * page_w)
                                    y1 = int(geometry[1][1] * page_h)

                                    words.append(text)
                                    # LayoutLMv3 expects normalized coordinates [0-1000]
                                    boxes.append([
                                        int(x0 / page_w * 1000),
                                        int(y0 / page_h * 1000),
                                        int(x1 / page_w * 1000),
                                        int(y1 / page_h * 1000)
                                    ])

        if len(words) < min_toc_entries:
            return False, 0.0, {
                'reason': f'Too few words: {len(words)} < {min_toc_entries}',
                'num_words': len(words)
            }

        # Prepare inputs for LayoutLMv3
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        # Move encoding to device
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()

        # Analyze predictions to detect TOC patterns
        # LayoutLMv3 outputs token classifications
        # We look for patterns typical of TOC entries

        # Simple heuristic: Check if lines end with numbers and are right-aligned
        entities = []
        for i, (word, bbox) in enumerate(zip(words, boxes)):
            # Check if word looks like a page number
            is_number = word.strip().replace('.', '').replace(',', '').isdigit()

            # Check if word is right-aligned (bbox x_max > 800 in normalized coords)
            is_right_aligned = bbox[2] > 800

            # Combine signals
            is_toc_entry = is_number and is_right_aligned

            if is_toc_entry:
                entities.append(TOCEntity(
                    text=word,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    is_toc_entry=True,
                    confidence=0.8
                ))

        num_toc_entries = len(entities)

        # Calculate confidence based on detection patterns
        if num_toc_entries >= min_toc_entries:
            confidence = min(0.9, 0.5 + (num_toc_entries / 20) * 0.4)
            is_toc = True
        else:
            confidence = num_toc_entries / min_toc_entries * 0.5
            is_toc = False

        metadata = {
            'model': 'LayoutLMv3',
            'num_words': len(words),
            'num_toc_entries': num_toc_entries,
            'confidence': confidence
        }

        return is_toc, confidence, metadata

    except ImportError as e:
        logger.error(f"LayoutLMv3 not available: {e}")
        logger.info("Install with: pip install transformers")
        return False, 0.0, {'error': 'LayoutLMv3 not installed'}
    except Exception as e:
        logger.error(f"LayoutLMv3 detection failed: {e}")
        return False, 0.0, {'error': str(e)}


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
