#!/usr/bin/env python3
"""
Test the notebook example to ensure it works
"""

import sys
sys.path.insert(0, '../src')

from ocr_reflow.main import process_document_with_layout
import cv2

print("=" * 70)
print("TESTING NOTEBOOK EXAMPLE")
print("=" * 70)

# Test the exact code from the notebook
input_image = 'images/jtg_p033.png'

print(f"\nProcessing: {input_image}")
print("This will:")
print("  - Detect layout (titles, text, figures, formulas)")
print("  - Skip skew detection for titles")
print("  - Reflow text with word wrapping")
print("  - Add extra spacing around titles")
print("  - Preserve figures and formulas as images")
print()

try:
    reflowed_page = process_document_with_layout(
        input_image,
        zoom_factor=2.5,
        new_page_width=2000
    )

    print(f"\n✓ Processing complete")
    print(f"  Output size: {reflowed_page.shape[1]}x{reflowed_page.shape[0]}")

    # Save output
    output_path = '../output_reflowed_example.png'
    cv2.imwrite(output_path, reflowed_page)
    print(f"✓ Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("SUCCESS: Notebook example works correctly")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("FAILED: Notebook example has issues")
    print("=" * 70)
    sys.exit(1)
