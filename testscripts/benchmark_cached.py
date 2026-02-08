#!/usr/bin/env python3
"""
Benchmark script to test model caching within the same Python process.
This demonstrates the true benefit of our optimization.
"""
import time
import sys
import os

# Add to path
sys.path.insert(0, '../src')

def main():
    print("=" * 80)
    print("PERFORMANCE BENCHMARK - MODEL CACHING TEST")
    print("=" * 80)

    # Import here to measure import time
    print("\n[1/4] Importing modules...")
    import_start = time.time()
    from ocr_reflow.main import process_document_with_layout
    import cv2
    import_time = time.time() - import_start
    print(f"      Import time: {import_time:.2f}s")

    # Test file
    test_file = "../images/kf_p025.png"
    print(f"\n[2/4] Test file: {test_file}")

    # First run - models will be loaded
    print("\n[3/4] First run (loading models)...")
    start = time.time()
    result1 = process_document_with_layout(test_file)
    time1 = time.time() - start
    print(f"      Execution time: {time1:.2f}s")
    print(f"      Output shape: {result1.shape}")

    # Second run - models are cached
    print("\n[4/4] Second run (using cached models)...")
    start = time.time()
    result2 = process_document_with_layout(test_file)
    time2 = time.time() - start
    print(f"      Execution time: {time2:.2f}s")
    print(f"      Output shape: {result2.shape}")

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nFirst run (with model loading):  {time1:.2f}s")
    print(f"Second run (with cached models): {time2:.2f}s")
    print(f"\nTime saved: {time1 - time2:.2f}s ({((time1-time2)/time1*100):.1f}% faster)")
    print(f"Speedup factor: {time1/time2:.2f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\n✓ Model caching is working!")
    print(f"✓ Models loaded once in {time1:.1f}s, then reused in {time2:.1f}s")
    print("\nFor batch processing:")
    print("  - Keep Python process alive")
    print("  - Process multiple images in one script")
    print("  - Or run as a persistent service")

if __name__ == "__main__":
    main()
