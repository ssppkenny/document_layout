"""
LayoutLMv3 Fine-Tuning for Table of Contents Detection

This script creates a training dataset from your images and fine-tunes
LayoutLMv3 for TOC detection.

Dataset Creation:
- Uses existing images with known TOC/non-TOC labels
- Extracts text using doctr OCR
- Creates annotations for training

Training:
- Fine-tunes microsoft/layoutlmv3-base
- Uses binary classification (TOC vs non-TOC)
- Saves best model checkpoint
- Uses GPU if available
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # AdamW moved to torch.optim in newer versions
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Paths
    images_dir: str = "images"
    output_dir: str = "models/layoutlmv3_toc"
    dataset_json: str = "dataset/toc_dataset.json"

    # Model
    model_name: str = "microsoft/layoutlmv3-base"

    # Training
    num_epochs: int = 10
    batch_size: int = 1  # Reduced to 1 for 4GB GPU
    gradient_accumulation_steps: int = 4  # Accumulate gradients to simulate batch size 4
    learning_rate: float = 5e-5
    warmup_steps: int = 50
    max_length: int = 512

    # Data split
    train_ratio: float = 0.8

    # Early stopping
    patience: int = 3


# Known TOC and non-TOC pages (from our tests)
KNOWN_LABELS = {
    # TOC pages (label=1) - 27 total
    'mh_p005.png': 1,
    'hlw_p009.png': 1,
    'dlr_p006.png': 1,
    'its_p008.png': 1,
    'kf_p003.png': 1,
    # TOC pages batch 1
    'acd_p006.png': 1,
    'acd_p007.png': 1,
    'acd_p008.png': 1,
    'acd_p009.png': 1,
    'itp_p010.png': 1,
    'itp_p011.png': 1,
    'itp_p012.png': 1,
    'itp_p013.png': 1,
    # TOC pages batch 2
    'efl_p005.png': 1,
    'efl_p006.png': 1,
    'lhe_p006.png': 1,
    'lhe_p007.png': 1,
    'sedg_p007.png': 1,
    # TOC pages batch 3 (NEW - Feb 2026)
    'its_p009.png': 1,
    'its_p010.png': 1,
    'its_p011.png': 1,
    'its_p012.png': 1,
    'pia_p008.png': 1,
    'pia_p009.png': 1,
    'pia_p010.png': 1,
    'pia_p011.png': 1,
    'pia_p012.png': 1,

    # Non-TOC pages (label=0) - 27 total
    'dvurog_p017.png': 0,
    'dvurog_p019.png': 0,
    'dvurog_p076.png': 0,
    'sedg_p598.png': 0,
    'jtg_p033.png': 0,
    'hlw_p040.png': 0,
    'mh_p013.png': 0,
    'kf_p015.png': 0,
    'kf_p016.png': 0,
    # Non-TOC pages batch 1
    'dvurog_p018.png': 0,
    'dvurog_p020.png': 0,
    'lw_p039.png': 0,
    'mh_p010.png': 0,
    # Non-TOC pages batch 2
    'efl_p050.png': 0,
    'efl_p051.png': 0,
    'lhe_p017.png': 0,
    'lhe_p018.png': 0,
    # Non-TOC pages batch 3 (NEW - Feb 2026)
    'its_p015.png': 0,
    'its_p016.png': 0,
    'its_p017.png': 0,
    'its_p018.png': 0,
    'pia_p013.png': 0,
    'pia_p014.png': 0,
    'pia_p015.png': 0,
    'pia_p016.png': 0,
    'pia_p017.png': 0,
    'pia_p018.png': 0,
}


# ============================================================================
# Dataset Creation
# ============================================================================

def extract_text_with_ocr(image_path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Extract text and bounding boxes using doctr OCR.

    Returns:
        Tuple of (words, boxes) where boxes are normalized [0-1000]
    """
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile

    # Load OCR model
    model = ocr_predictor(pretrained=True)

    # Load image
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    # Load image for dimensions
    img = Image.open(image_path).convert("RGB")
    page_w, page_h = img.size

    words = []
    boxes = []

    # Extract from doctr result
    if hasattr(result, 'export'):
        doc_dict = result.export()
        for page in doc_dict.get('pages', []):
            for block in page.get('blocks', []):
                for line in block.get('lines', []):
                    for word_data in line.get('words', []):
                        text = word_data.get('value', '').strip()
                        if text:
                            geometry = word_data.get('geometry', [[0, 0], [1, 1]])
                            if len(geometry) >= 2:
                                # Convert to absolute coordinates
                                x0 = geometry[0][0] * page_w
                                y0 = geometry[0][1] * page_h
                                x1 = geometry[1][0] * page_w
                                y1 = geometry[1][1] * page_h

                                # Normalize to [0, 1000] for LayoutLMv3
                                box = [
                                    int((x0 / page_w) * 1000),
                                    int((y0 / page_h) * 1000),
                                    int((x1 / page_w) * 1000),
                                    int((y1 / page_h) * 1000)
                                ]

                                words.append(text)
                                boxes.append(box)

    return words, boxes


def create_dataset(config: TrainingConfig):
    """
    Create dataset JSON from images with known labels.
    """
    logger.info("Creating dataset from images...")

    # Create dataset directory
    os.makedirs(os.path.dirname(config.dataset_json), exist_ok=True)

    dataset = []

    for filename, label in tqdm(KNOWN_LABELS.items(), desc="Processing images"):
        image_path = os.path.join(config.images_dir, filename)

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        try:
            # Extract text and boxes
            words, boxes = extract_text_with_ocr(image_path)

            if len(words) == 0:
                logger.warning(f"No text extracted from {filename}")
                continue

            # Add to dataset
            dataset.append({
                'image_path': image_path,
                'words': words,
                'boxes': boxes,
                'label': label,  # 0=non-TOC, 1=TOC
                'filename': filename
            })

            logger.info(f"✓ {filename}: {len(words)} words, label={label}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    # Save dataset
    with open(config.dataset_json, 'w') as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"✓ Dataset saved: {config.dataset_json}")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  TOC pages: {sum(1 for d in dataset if d['label'] == 1)}")
    logger.info(f"  Non-TOC pages: {sum(1 for d in dataset if d['label'] == 0)}")

    return dataset


# ============================================================================
# PyTorch Dataset
# ============================================================================

class TOCDataset(Dataset):
    """
    PyTorch Dataset for TOC detection with LayoutLMv3.
    """

    def __init__(self, data: List[Dict], processor: LayoutLMv3Processor, max_length: int = 512):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image = Image.open(item['image_path']).convert("RGB")

        # Prepare inputs for LayoutLMv3
        encoding = self.processor(
            image,
            item['words'],
            boxes=item['boxes'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add label
        encoding['labels'] = torch.tensor(item['label'], dtype=torch.long)

        return encoding


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps  # Scale loss

        # Backward pass
        loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Calculate accuracy
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

        # Update stats
        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
            'acc': f"{100*correct/total:.2f}%"
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Calculate accuracy
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(config: TrainingConfig):
    """
    Main training function.
    """
    logger.info("=" * 80)
    logger.info("LAYOUTLMV3 FINE-TUNING FOR TOC DETECTION")
    logger.info("=" * 80)

    # Create dataset if needed
    if not os.path.exists(config.dataset_json):
        dataset = create_dataset(config)
    else:
        logger.info(f"Loading existing dataset: {config.dataset_json}")
        with open(config.dataset_json, 'r') as f:
            dataset = json.load(f)

    if len(dataset) < 2:
        logger.error("Not enough samples in dataset. Need at least 2 samples.")
        return

    # Split dataset
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    split_idx = int(len(dataset) * config.train_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]

    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val: {len(val_data)} samples")

    # Load processor and model
    logger.info(f"Loading model: {config.model_name}")
    processor = LayoutLMv3Processor.from_pretrained(config.model_name, apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2  # Binary classification: TOC vs non-TOC
    )
    model.to(DEVICE)
    logger.info(f"✓ Model loaded on {DEVICE}")

    # Create datasets and dataloaders
    train_dataset = TOCDataset(train_data, processor, config.max_length)
    val_dataset = TOCDataset(val_data, processor, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info("=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    best_val_acc = 0
    patience_counter = 0

    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        logger.info("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, config.gradient_accumulation_steps)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save model
            model_path = os.path.join(config.output_dir, "best_model")
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)

            logger.info(f"✓ New best model saved! Val Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config.patience}")

        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Model saved to: {config.output_dir}/best_model")

    return model, best_val_acc


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = TrainingConfig()

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠ No GPU available. Training will be slow on CPU.")

    # Train
    try:
        model, best_acc = train_model(config)
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"Final accuracy: {best_acc*100:.2f}%")
        logger.info(f"\nTo use the fine-tuned model:")
        logger.info(f"  1. Update layoutlm_toc_detector.py to load from: {config.output_dir}/best_model")
        logger.info(f"  2. Run: python src/ocr_reflow/main.py IMAGE --layout --toc-algorithm layoutlm")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
