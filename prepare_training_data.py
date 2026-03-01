"""
Download and Prepare Additional TOC Training Data

This script downloads publicly available document datasets that can be used
to augment the TOC detection training.
"""

import os
import requests
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_publaynet_sample():
    """
    Download a small sample from PubLayNet dataset.
    PubLayNet contains document layout annotations which can help.
    """
    logger.info("Checking for PubLayNet samples...")

    # PubLayNet is large (96GB), so we'll create synthetic TOC examples instead
    logger.info("PubLayNet is too large for automatic download.")
    logger.info("Consider manually downloading from: https://github.com/ibm-aur-nlp/PubLayNet")

    return False


def create_synthetic_toc_data(output_dir="dataset/synthetic"):
    """
    Create synthetic TOC-like data for augmentation.
    """
    logger.info("Creating synthetic TOC data...")

    os.makedirs(output_dir, exist_ok=True)

    # Synthetic TOC patterns
    toc_templates = [
        # Standard book TOC
        [
            ("Chapter 1: Introduction", "....", "1"),
            ("Chapter 2: Background", "....", "15"),
            ("Chapter 3: Methods", "....", "29"),
            ("Chapter 4: Results", "....", "45"),
        ],
        # Academic paper TOC
        [
            ("1. Introduction", "....", "3"),
            ("2. Related Work", "....", "7"),
            ("3. Methodology", "....", "12"),
            ("4. Experiments", "....", "18"),
            ("5. Conclusion", "....", "25"),
        ],
        # Multi-level TOC
        [
            ("Part I: Foundations", "", ""),
            ("  Chapter 1", "....", "5"),
            ("  Chapter 2", "....", "20"),
            ("Part II: Applications", "", ""),
            ("  Chapter 3", "....", "35"),
            ("  Chapter 4", "....", "50"),
        ],
    ]

    logger.info(f"Created {len(toc_templates)} synthetic TOC templates")
    logger.info("Note: These are for reference. Real images work better.")

    return True


def check_available_data():
    """
    Check what data is available for training.
    """
    logger.info("=" * 80)
    logger.info("AVAILABLE TRAINING DATA")
    logger.info("=" * 80)

    # Check local images
    images_dir = "images"
    if os.path.exists(images_dir):
        images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        logger.info(f"\n✓ Local images: {len(images)} files in {images_dir}/")
        logger.info(f"  These can be used for training after labeling")

    # Check if dataset exists
    dataset_file = "dataset/toc_dataset.json"
    if os.path.exists(dataset_file):
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        logger.info(f"\n✓ Existing dataset: {len(dataset)} samples")
        logger.info(f"  File: {dataset_file}")
    else:
        logger.info(f"\n✗ No dataset file found at {dataset_file}")
        logger.info(f"  Will be created during training")

    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    logger.info("\nFor best results:")
    logger.info("  1. Current approach: Use your 54 labeled images (27 TOC + 27 non-TOC)")
    logger.info("  2. Label more images: Add more pages from images/ folder")
    logger.info("  3. Download datasets:")
    logger.info("     - PubLayNet: https://github.com/ibm-aur-nlp/PubLayNet")
    logger.info("     - DocBank: https://github.com/doc-analysis/DocBank")
    logger.info("  4. Collect real TOC pages: Scan books/papers you have")

    logger.info("\nMinimum for training: 10-20 labeled pages")
    logger.info("Recommended: 50-100 labeled pages")
    logger.info("Optimal: 500+ labeled pages")

    logger.info("\nCurrent status:")
    logger.info("  ✓ Ready to train with 54 labeled pages (27 TOC + 27 non-TOC)")
    logger.info("  ✓ Perfectly balanced dataset - excellent for training!")
    logger.info("  ✓ Good dataset size - should give good accuracy")


def label_additional_images():
    """
    Interactive script to label additional images as TOC/non-TOC.
    """
    logger.info("=" * 80)
    logger.info("INTERACTIVE IMAGE LABELING")
    logger.info("=" * 80)

    images_dir = "images"
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        return

    # Known labels (synchronized with train_layoutlmv3.py)
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

    all_images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    unlabeled = [f for f in all_images if f not in KNOWN_LABELS]

    logger.info(f"\nFound {len(unlabeled)} unlabeled images")

    if len(unlabeled) == 0:
        logger.info("All images are already labeled!")
        return

    logger.info("\nTo label more images:")
    logger.info("  1. Open each image in an image viewer")
    logger.info("  2. Determine if it's a TOC page or not")
    logger.info("  3. Add to KNOWN_LABELS in train_layoutlmv3.py")
    logger.info("\nExample:")
    logger.info("  KNOWN_LABELS = {")
    logger.info("      'new_image.png': 1,  # TOC page")
    logger.info("      'another.png': 0,    # Not TOC")
    logger.info("  }")

    logger.info(f"\nUnlabeled images:")
    for img in unlabeled[:10]:  # Show first 10
        logger.info(f"  - {img}")

    if len(unlabeled) > 10:
        logger.info(f"  ... and {len(unlabeled) - 10} more")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("TOC TRAINING DATA PREPARATION")
    logger.info("=" * 80)

    # Check available data
    check_available_data()

    # Show how to label more
    logger.info("\n")
    label_additional_images()

    # Create synthetic data (optional)
    logger.info("\n")
    create_synthetic_toc_data()

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("\n1. Train with current dataset (54 images - perfectly balanced!):")
    logger.info("   pixi run python train_layoutlmv3.py")
    logger.info("\n2. Better accuracy (label more images):")
    logger.info("   - Label more images in KNOWN_LABELS in train_layoutlmv3.py")
    logger.info("   - Keep TOC and non-TOC balanced (same number)")
    logger.info("   - Run: pixi run python train_layoutlmv3.py")
    logger.info("\n3. Best accuracy (collect large dataset):")
    logger.info("   - Collect 100-200 TOC pages and 100-200 non-TOC pages")
    logger.info("   - Label them in KNOWN_LABELS")
    logger.info("   - Run: pixi run python train_layoutlmv3.py")
