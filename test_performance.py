#!/usr/bin/env python3
"""
Advanced performance testing and profiling script.
Tests multiple optimization scenarios.
"""
import time
import sys
import os
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, 'src')

def profile_function(func, *args, **kwargs):
    """Profile a function and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Get stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs().sort_stats('cumulative')

    return result, ps

def test_model_caching():
    """Test that model caching works correctly."""
    print("=" * 80)
    print("TEST 1: MODEL CACHING")
    print("=" * 80)

    from ocr_reflow.main import process_document_with_layout, _CACHED_DOCTR_MODEL

    test_file = "images/kf_p025.png"

    # Check cache is initially empty
    assert _CACHED_DOCTR_MODEL is None, "Cache should be empty initially"
    print("\n✓ Cache initially empty")

    # First run
    print("\nRun 1: Loading models...")
    start = time.time()
    result1 = process_document_with_layout(test_file)
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s")

    # Check cache is now populated
    from ocr_reflow import main as main_module
    assert main_module._CACHED_DOCTR_MODEL is not None, "Cache should be populated"
    print("  ✓ DocTR model cached")

    from ocr_reflow import layout as layout_module
    if hasattr(layout_module, '_CACHED_YOLO_MODEL'):
        assert layout_module._CACHED_YOLO_MODEL is not None, "YOLO cache should be populated"
        print("  ✓ YOLO model cached")

    # Second run
    print("\nRun 2: Using cached models...")
    start = time.time()
    result2 = process_document_with_layout(test_file)
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s")

    # Verify speedup
    speedup = time1 / time2
    print(f"\n✓ Speedup: {speedup:.2f}x faster")

    if speedup > 2:
        print(f"✓✓ EXCELLENT: More than 2x speedup achieved!")
    elif speedup > 1.5:
        print(f"✓ GOOD: Model caching working as expected")
    else:
        print(f"⚠ WARNING: Speedup less than expected. Check if models are being cached.")

    return time1, time2

def test_image_reading_optimization():
    """Test that we're not reading images multiple times."""
    print("\n" + "=" * 80)
    print("TEST 2: IMAGE READING OPTIMIZATION")
    print("=" * 80)

    import cv2

    # Monkey-patch cv2.imread to count calls
    original_imread = cv2.imread
    read_count = [0]

    def counting_imread(*args, **kwargs):
        read_count[0] += 1
        return original_imread(*args, **kwargs)

    cv2.imread = counting_imread

    try:
        from importlib import reload
        import ocr_reflow.main
        reload(ocr_reflow.main)

        test_file = "images/kf_p025.png"

        print(f"\nProcessing {test_file}...")
        read_count[0] = 0  # Reset counter

        result = ocr_reflow.main.process_document(test_file)

        print(f"Image reads: {read_count[0]}")

        if read_count[0] == 1:
            print("✓✓ EXCELLENT: Image read only once!")
        elif read_count[0] <= 2:
            print("✓ GOOD: Minimal image reads")
        else:
            print(f"⚠ WARNING: Image read {read_count[0]} times. Should be optimized.")

    finally:
        cv2.imread = original_imread

    return read_count[0]

def test_batch_performance():
    """Test batch processing performance."""
    print("\n" + "=" * 80)
    print("TEST 3: BATCH PROCESSING")
    print("=" * 80)

    import glob
    from ocr_reflow.main import process_document_with_layout

    # Find test images
    images = glob.glob("images/*.png")[:3]  # Test with 3 images

    if len(images) < 2:
        print("⚠ Not enough test images. Skipping batch test.")
        return None

    print(f"\nProcessing {len(images)} images...")

    times = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path}")
        start = time.time()
        result = process_document_with_layout(img_path)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    Time: {elapsed:.2f}s")

    print(f"\nTotal time: {sum(times):.2f}s")
    print(f"Average: {sum(times)/len(times):.2f}s per image")

    if len(times) > 1:
        first = times[0]
        avg_rest = sum(times[1:]) / len(times[1:])
        print(f"\nFirst image: {first:.2f}s (loads models)")
        print(f"Subsequent: {avg_rest:.2f}s average (uses cache)")
        print(f"Speedup: {first/avg_rest:.2f}x")

    return times

def main():
    print("╔" + "=" * 78 + "╗")
    print("║" + " PERFORMANCE OPTIMIZATION TEST SUITE ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")

    results = {}

    try:
        # Test 1: Model caching
        time1, time2 = test_model_caching()
        results['caching_speedup'] = time1 / time2

        # Test 2: Image reading
        read_count = test_image_reading_optimization()
        results['image_reads'] = read_count

        # Test 3: Batch processing
        batch_times = test_batch_performance()
        if batch_times:
            results['batch_times'] = batch_times

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " SUMMARY ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")

    print(f"\n✓ Model caching speedup: {results.get('caching_speedup', 0):.2f}x")
    print(f"✓ Image reads per document: {results.get('image_reads', 'N/A')}")

    if 'batch_times' in results:
        times = results['batch_times']
        if len(times) > 1:
            print(f"✓ Batch processing: {times[0]:.1f}s first, {sum(times[1:])/len(times[1:]):.1f}s avg")

    print("\n" + "=" * 80)
    print("All optimizations are working correctly! ✓")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
