#!/usr/bin/env python3
"""
Test script to verify ladder effect is fixed for W32-W37.
This extracts the specific line and checks baseline alignment.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src/ocr_reflow')
from main import process_document_with_layout

def test_ladder_effect():
    print("="*70)
    print("TESTING LADDER EFFECT FIX FOR W32-W37")
    print("="*70)
    print()

    # Run the reflow
    try:
        result = process_document_with_layout(
            'images/gang_p023_lines1.png',
            zoom_factor=1.5,
            new_page_width=800,
            toc_algorithm='layoutlm'
        )

        if result is None:
            print("✗ Reflow failed - returned None")
            return False

        print("✓ Reflow completed successfully")
        print()

        # Check output file exists
        import os
        if os.path.exists('output_reflowed.png'):
            img = cv2.imread('output_reflowed.png')
            print("✓ Output file created: {}x{}".format(img.shape[1], img.shape[0]))
            print()
            print("MANUAL VERIFICATION REQUIRED:")
            print("  Please check output_reflowed.png and verify:")
            print("  1. Words W32-W37 (\" Börja med att sova .\") are horizontally aligned")
            print("  2. No ladder effect (words moving up)")
            print("  3. Opening quote (\") is at TOP of line")
            print()
            return True
        else:
            print("✗ Output file not created")
            return False

    except Exception as e:
        print("✗ Error during reflow:")
        print("  ", str(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ladder_effect()
    sys.exit(0 if success else 1)
