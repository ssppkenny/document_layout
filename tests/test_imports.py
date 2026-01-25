#!/usr/bin/env python3
"""
Test script to verify ocr_reflow package installation and imports.
Run this after installing the package to ensure everything works.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 50)

    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False

    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ torchvision import failed: {e}")
        return False

    try:
        import cv2
        print(f"✓ opencv-python {cv2.__version__}")
    except ImportError as e:
        print(f"✗ opencv-python import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False

    try:
        import scipy
        print(f"✓ scipy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ scipy import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False

    try:
        from doctr.models import detection_predictor
        print(f"✓ python-doctr (detection_predictor imported)")
    except ImportError as e:
        print(f"✗ python-doctr import failed: {e}")
        return False

    try:
        import shapely
        print(f"✓ shapely {shapely.__version__}")
    except ImportError as e:
        print(f"✗ shapely import failed: {e}")
        return False

    print("-" * 50)

    try:
        import ocr_reflow
        print(f"✓ ocr_reflow package imported")
    except ImportError as e:
        print(f"✗ ocr_reflow import failed: {e}")
        print("\nMake sure you installed the package with:")
        print("  pip install -e .")
        return False

    try:
        from ocr_reflow import process_document, Letter
        print(f"✓ ocr_reflow.process_document imported")
        print(f"✓ ocr_reflow.Letter imported")
    except ImportError as e:
        print(f"✗ ocr_reflow components import failed: {e}")
        return False

    print("-" * 50)
    print("✓ All imports successful!")
    return True

def main():
    """Main test function."""
    print("\n" + "=" * 50)
    print("OCR Reflow Package Import Test")
    print("=" * 50 + "\n")

    success = test_imports()

    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Package is properly installed!")
        print("\nYou can now use the package:")
        print("  from ocr_reflow import process_document")
        print("  result = process_document('your_image.png')")
        sys.exit(0)
    else:
        print("FAILURE: Some imports failed.")
        print("\nPlease check the errors above and:")
        print("  1. Ensure you're in the correct environment")
        print("  2. Run: pip install -e .")
        print("  3. Check pixi.toml for missing dependencies")
        sys.exit(1)
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
