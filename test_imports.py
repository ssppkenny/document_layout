#!/usr/bin/env python3
"""Test script to verify imports work correctly"""

import sys
import os

# Add the ocr_reflow directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ocr_reflow'))

print("Testing imports...")

try:
    from device_utils import get_device_for_doctr
    print("✓ device_utils imported successfully")
except ImportError as e:
    print(f"✗ device_utils import failed: {e}")

try:
    from reflow import create_page_with_word_wrapping
    print("✓ reflow imported successfully")
except ImportError as e:
    print(f"✗ reflow import failed: {e}")

try:
    from divide_conquer_4d import divide_conquer_4d, Point4D
    print("✓ divide_conquer_4d imported successfully")
except ImportError as e:
    print(f"✗ divide_conquer_4d import failed: {e}")

try:
    from layout import layout as analyze_layout
    print("✓ layout imported successfully")
    print(f"  analyze_layout function: {analyze_layout}")
except ImportError as e:
    print(f"✗ layout import failed: {e}")

print("\nTesting device detection...")
try:
    from device_utils import get_device_for_doctr
    device = get_device_for_doctr()
    print(f"✓ Detected device: {device}")
except Exception as e:
    print(f"✗ Device detection failed: {e}")

print("\nAll import tests completed!")
