"""
Model Manager for OCR Reflow

Handles model paths, downloads, and caching for all ML models used in the project.
Models are stored in the models/ directory at the project root.

Directory structure:
    models/
    ├── doclayout_yolo_docstructbench_imgsz1024.pt    # Layout detection (YOLO)
    ├── yolo26n_doc_layout.pt                         # YOLOv26 (ensemble) layout detection
    ├── layoutlmv3_toc/                                 # TOC detection (fine-tuned)
    │   └── best_model/
    │       ├── config.json
    │       ├── model.safetensors
    │       └── ...
    └── doctr/                                          # Text detection (auto-cached)
        └── (cached by doctr library)
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# YOLOv26 model constants (used for ensemble layout detection)
_YOLO26_REPO = "Armaggheddon/yolo26-document-layout"
_YOLO26_FILENAME = "yolo26n_doc_layout.pt"


def get_project_root() -> Path:
    """Get the project root directory (where models/ folder is located)."""
    # This file is in src/ocr_reflow/, go up 2 levels to reach project root
    return Path(__file__).parent.parent.parent


def get_models_dir() -> Path:
    """Get the models directory path."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def get_doclayout_yolo_path() -> str:
    """
    Get path to DocLayout-YOLO model.

    Returns:
        str: Path to the .pt model file

    Raises:
        FileNotFoundError: If model file is not found
    """
    model_path = get_models_dir() / "doclayout_yolo_docstructbench_imgsz1024.pt"

    if not model_path.exists():
        error_msg = f"""
DocLayout-YOLO model not found at: {model_path}

To download the model:
1. Manual download:
   - Go to: https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench
   - Download: doclayout_yolo_docstructbench_imgsz1024.pt
   - Place in: {get_models_dir()}/

2. Or use huggingface-cli:
   pip install huggingface-hub
   huggingface-cli download juliozhao/DocLayout-YOLO-DocStructBench \\
       doclayout_yolo_docstructbench_imgsz1024.pt \\
       --local-dir {get_models_dir()}
"""
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Using DocLayout-YOLO model: {model_path}")
    return str(model_path)


def get_layoutlmv3_toc_path() -> str:
    """
    Get path to fine-tuned LayoutLMv3 TOC detection model.

    Downloads from HuggingFace into models/layoutlmv3_toc/ if not already present.

    Returns:
        str: Path to the model directory

    Raises:
        FileNotFoundError: If model is not found and cannot be downloaded
    """
    repo_id = "ssppkenny/layoutlmv3-toc-detector"
    model_path = get_models_dir() / "layoutlmv3_toc"

    if model_path.exists() and (model_path / "config.json").exists():
        logger.info(f"Using LayoutLMv3 TOC model: {model_path}")
        return str(model_path)

    print(f"LayoutLMv3 TOC model not found at: {model_path}")
    print(f"Downloading from HuggingFace ({repo_id})...")
    print("This is a one-time download (~500 MB) and may take a few minutes...")

    try:
        from huggingface_hub import snapshot_download

        model_path.mkdir(parents=True, exist_ok=True)

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
            )
            print(f"Model downloaded to: {model_path}")
            logger.info(f"LayoutLMv3 TOC model downloaded to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            raise FileNotFoundError(
                f"Could not download model from HuggingFace ({repo_id}): {e}"
            ) from e

        return str(model_path)

    except ImportError:
        error_msg = (
            f"huggingface_hub is not installed. Cannot download the LayoutLMv3 TOC model.\n"
            f"Install it with: pip install huggingface-hub\n"
            f"Then re-run to auto-download from: {repo_id}"
        )
        print(f"ERROR: {error_msg}")
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)



def download_doclayout_yolo():
    """Download DocLayout-YOLO model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading DocLayout-YOLO model from HuggingFace...")

        filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            cache_dir=get_models_dir(),
            local_dir=get_models_dir(),
            local_dir_use_symlinks=False
        )

        logger.info(f"✓ DocLayout-YOLO model downloaded to: {filepath}")
        return filepath

    except ImportError:
        error_msg = """
huggingface_hub not installed.

Install with:
    pip install huggingface-hub

Or download manually from:
    https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench
"""
        logger.error(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def get_yolo26_path() -> str:
    """
    Get path to YOLOv26 nano layout model (used in ensemble with doclayout-yolo).

    Raises:
        FileNotFoundError: If model file is not found
    """
    model_path = get_models_dir() / _YOLO26_FILENAME
    if not model_path.exists():
        error_msg = f"""
YOLOv26 model not found at: {model_path}

To download the model automatically: ensure_all_models() will download it.
Or manually: huggingface-cli download {_YOLO26_REPO} {_YOLO26_FILENAME} --local-dir {get_models_dir()}
"""
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    logger.info(f"Using YOLOv26 model: {model_path}")
    return str(model_path)


def download_yolo26():
    """Download YOLOv26 model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading YOLOv26 model from HuggingFace...")
        filepath = hf_hub_download(
            repo_id=_YOLO26_REPO,
            filename=_YOLO26_FILENAME,
            cache_dir=get_models_dir(),
            local_dir=get_models_dir(),
            local_dir_use_symlinks=False,
        )
        logger.info(f"✓ YOLOv26 model downloaded to: {filepath}")
        return filepath
    except ImportError:
        error_msg = "huggingface_hub not installed. Install with: pip install huggingface-hub"
        logger.error(error_msg)
        raise ImportError(error_msg)
    except Exception as e:
        logger.error(f"Failed to download YOLOv26 model: {e}")
        raise


def ensure_all_models():
    """
    Ensure all required models are available, downloading if necessary.

    Returns:
        dict: Paths to all models
    """
    models = {}

    # Check DocLayout-YOLO
    try:
        models['doclayout_yolo'] = get_doclayout_yolo_path()
    except FileNotFoundError:
        logger.info("DocLayout-YOLO not found, attempting download...")
        models['doclayout_yolo'] = download_doclayout_yolo()

    # Check YOLOv26 (required for ensemble layout detection)
    try:
        models['yolo26'] = get_yolo26_path()
    except FileNotFoundError:
        logger.info("YOLOv26 not found, attempting download...")
        models['yolo26'] = download_yolo26()

    # Check LayoutLMv3 TOC
    try:
        models['layoutlmv3_toc'] = get_layoutlmv3_toc_path()
    except FileNotFoundError:
        logger.warning("Fine-tuned LayoutLMv3 model not found. Run 'python train_layoutlmv3.py' to train.")
        models['layoutlmv3_toc'] = None

    # DocTR models are auto-downloaded by the library
    logger.info("DocTR models will be auto-downloaded by the library on first use")

    return models


def get_cache_info():
    """Get information about cached models."""
    info = {
        'models_dir': str(get_models_dir()),
        'models': {}
    }

    # DocLayout-YOLO
    yolo_path = get_models_dir() / "doclayout_yolo_docstructbench_imgsz1024.pt"
    if yolo_path.exists():
        size_mb = yolo_path.stat().st_size / (1024 * 1024)
        info['models']['doclayout_yolo'] = {
            'path': str(yolo_path),
            'size_mb': f"{size_mb:.1f}",
            'exists': True,
            'managed_by': 'ocr_reflow'
        }
    else:
        info['models']['doclayout_yolo'] = {'exists': False}

    # YOLOv26
    yolo26_path = get_models_dir() / _YOLO26_FILENAME
    if yolo26_path.exists():
        size_mb = yolo26_path.stat().st_size / (1024 * 1024)
        info['models']['yolo26'] = {
            'path': str(yolo26_path),
            'size_mb': f"{size_mb:.1f}",
            'exists': True,
            'managed_by': 'ocr_reflow'
        }
    else:
        info['models']['yolo26'] = {'exists': False}

    # LayoutLMv3 TOC
    layoutlm_path = get_models_dir() / "layoutlmv3_toc"
    if layoutlm_path.exists():
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in layoutlm_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        info['models']['layoutlmv3_toc'] = {
            'path': str(layoutlm_path),
            'size_mb': f"{size_mb:.1f}",
            'exists': True,
            'managed_by': 'ocr_reflow'
        }
    else:
        info['models']['layoutlmv3_toc'] = {'exists': False}

    # DocTR models (managed by doctr library)
    from pathlib import Path
    doctr_cache = Path.home() / ".cache" / "doctr" / "models"
    if doctr_cache.exists():
        try:
            total_size = sum(f.stat().st_size for f in doctr_cache.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            num_files = len(list(doctr_cache.rglob('*.pt'))) + len(list(doctr_cache.rglob('*.zip')))
            info['models']['doctr'] = {
                'path': str(doctr_cache),
                'size_mb': f"{size_mb:.1f}",
                'num_models': num_files,
                'exists': True,
                'managed_by': 'doctr library (auto-downloaded)',
                'note': 'These models are automatically managed by the doctr library'
            }
        except Exception as e:
            info['models']['doctr'] = {
                'path': str(doctr_cache),
                'exists': True,
                'note': f'Directory exists but could not calculate size: {e}'
            }
    else:
        info['models']['doctr'] = {
            'exists': False,
            'note': 'Will be auto-downloaded by doctr on first use'
        }

    return info


if __name__ == "__main__":
    # CLI for model management
    import argparse

    parser = argparse.ArgumentParser(description="Manage OCR Reflow models")
    parser.add_argument('command', choices=['info', 'download', 'check'],
                       help='Command to execute')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == 'info':
        info = get_cache_info()
        print("\n" + "="*80)
        print("OCR REFLOW MODEL INFORMATION")
        print("="*80)
        print(f"\nModels directory: {info['models_dir']}")
        print("\nInstalled models:")
        for name, data in info['models'].items():
            if data['exists']:
                if 'managed_by' in data:
                    manager_info = f" (managed by: {data['managed_by']})"
                else:
                    manager_info = ""

                if name == 'doctr':
                    print(f"  ✓ {name}: {data['size_mb']} MB{manager_info}")
                    print(f"    Path: {data['path']}")
                    print(f"    Models: {data.get('num_models', 'N/A')} file(s)")
                    if 'note' in data:
                        print(f"    Note: {data['note']}")
                else:
                    print(f"  ✓ {name}: {data['size_mb']} MB{manager_info}")
                    print(f"    Path: {data['path']}")
            else:
                print(f"  ✗ {name}: NOT FOUND")
                if 'note' in data:
                    print(f"    Note: {data['note']}")
        print("="*80 + "\n")

    elif args.command == 'download':
        print("Downloading missing models...")
        models = ensure_all_models()
        print("\n✓ All available models ready!")

    elif args.command == 'check':
        models = ensure_all_models()
        print("\n✓ Model check complete")
