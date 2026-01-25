"""
Test script to verify the package installation and basic functionality.
"""

import sys


def test_import():
    """Test that the package can be imported."""
    print("Testing package import...")
    try:
        from ocr_reflow import process_document, Letter
        print("✓ Package imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    dependencies = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("doctr", "python-doctr"),
        ("shapely", "Shapely"),
    ]

    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} not found")
            all_ok = False

    return all_ok


def test_letter_class():
    """Test the Letter dataclass."""
    print("\nTesting Letter class...")
    try:
        from ocr_reflow import Letter
        letter = Letter(xmin=10, ymin=20, xmax=30, ymax=40, bl=5)
        assert letter.xmin == 10
        assert letter.ymin == 20
        assert letter.xmax == 30
        assert letter.ymax == 40
        assert letter.bl == 5
        print("✓ Letter class works correctly")
        return True
    except Exception as e:
        print(f"✗ Letter class test failed: {e}")
        return False


def test_module_structure():
    """Test that all expected modules are accessible."""
    print("\nTesting module structure...")
    try:
        from ocr_reflow import (
            process_document,
            Letter,
            find_rects,
            margins,
            create_page_with_word_wrapping,
        )
        print("✓ All expected functions are accessible")
        return True
    except ImportError as e:
        print(f"✗ Module structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("OCR Reflow Package Test Suite")
    print("=" * 60)

    tests = [
        test_import,
        test_dependencies,
        test_letter_class,
        test_module_structure,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All tests passed ({passed}/{total})")
        print("=" * 60)
        print("\n🎉 Package is ready to use!")
        print("\nNext steps:")
        print("  1. Try the example notebook: notebooks/example_usage.ipynb")
        print("  2. Run the CLI: ocr-reflow <image.png>")
        print("  3. Check out docs/JUPYTER_GUIDE.md for Jupyter usage")
        return 0
    else:
        print(f"✗ Some tests failed ({passed}/{total} passed)")
        print("=" * 60)
        print("\nPlease check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
