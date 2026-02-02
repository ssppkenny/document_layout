#!/usr/bin/env python3
"""
Test script for layout-based document processing.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from docs.main import process_document_with_layout
import cv2

def test_layout_processing(image_path):
    """Test the layout-based processing on an image."""
    print(f"Testing layout-based processing on: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    try:
        # Process the document with layout analysis
        result = process_document_with_layout(image_path, zoom_factor=2.5, new_page_width=2000)

        # Save the result
        output_path = "test_layout_output.png"
        cv2.imwrite(output_path, result)
        print(f"\nSuccess! Output saved to: {output_path}")
        print(f"Output dimensions: {result.shape[1]}x{result.shape[0]}")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_layout_integration.py <image_path>")
        sys.exit(1)

    test_layout_processing(sys.argv[1])
