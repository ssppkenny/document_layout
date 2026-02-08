#!/usr/bin/env python3
"""
Batch processing script to process multiple images efficiently.
Demonstrates the massive speedup from model caching.
"""
import time
import sys
import os
import glob
from pathlib import Path

# Add to path
sys.path.insert(0, '../src')

def process_batch(image_paths, output_dir="output_batch", use_layout=True):
    """
    Process multiple images in one Python session to benefit from model caching.

    Args:
        image_paths: List of image file paths
        output_dir: Directory to save output images
        use_layout: Whether to use layout-based processing
    """
    from ocr_reflow.main import process_document_with_layout, process_document
    import cv2

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("BATCH PROCESSING WITH MODEL CACHING")
    print("=" * 80)
    print(f"\nProcessing {len(image_paths)} images...")
    print(f"Output directory: {output_dir}")
    print(f"Using layout analysis: {use_layout}\n")

    times = []
    total_start = time.time()

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {img_path}")

        start = time.time()
        try:
            if use_layout:
                result = process_document_with_layout(img_path)
            else:
                result = process_document(img_path)

            # Save output
            output_name = Path(img_path).stem + "_reflowed.png"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, result)

            elapsed = time.time() - start
            times.append(elapsed)

            status = "✓" if i > 1 else "⚡"  # First one loads models
            print(f"    {status} Completed in {elapsed:.2f}s (output: {output_name})")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            times.append(0)

    total_time = time.time() - total_start

    # Print statistics
    print("\n" + "=" * 80)
    print("BATCH PROCESSING STATISTICS")
    print("=" * 80)

    if len(times) > 1:
        print(f"\nTotal images processed: {len(image_paths)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_paths):.2f}s")
        print(f"\nFirst image (with model loading): {times[0]:.2f}s")

        cached_times = [t for t in times[1:] if t > 0]
        if cached_times:
            avg_cached = sum(cached_times) / len(cached_times)
            print(f"Subsequent images (cached models): {avg_cached:.2f}s average")
            print(f"\nSpeedup from caching: {times[0]/avg_cached:.2f}x faster")
            print(f"Time saved per image: {times[0] - avg_cached:.2f}s")

            # Calculate what it would have taken without caching
            time_without_cache = times[0] * len(image_paths)
            time_with_cache = times[0] + avg_cached * (len(image_paths) - 1)
            savings = time_without_cache - time_with_cache

            print(f"\n" + "-" * 80)
            print("WITHOUT model caching:")
            print(f"  Would take: {time_without_cache:.1f}s ({time_without_cache/60:.1f} minutes)")
            print("WITH model caching:")
            print(f"  Actually took: {time_with_cache:.1f}s ({time_with_cache/60:.1f} minutes)")
            print(f"  Time saved: {savings:.1f}s ({savings/60:.1f} minutes) = {(savings/time_without_cache*100):.1f}% faster")

    print("\n" + "=" * 80)
    return times


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch process multiple images')
    parser.add_argument('pattern', nargs='?', default='images/*.png',
                        help='Glob pattern for input images (default: images/*.png)')
    parser.add_argument('--output-dir', default='output_batch',
                        help='Output directory (default: output_batch)')
    parser.add_argument('--no-layout', action='store_true',
                        help='Disable layout analysis')
    parser.add_argument('--limit', type=int,
                        help='Limit number of images to process')

    args = parser.parse_args()

    # Find images
    image_paths = sorted(glob.glob(args.pattern))

    if not image_paths:
        print(f"Error: No images found matching pattern: {args.pattern}")
        sys.exit(1)

    if args.limit:
        image_paths = image_paths[:args.limit]

    # Process batch
    process_batch(
        image_paths,
        output_dir=args.output_dir,
        use_layout=not args.no_layout
    )


if __name__ == "__main__":
    main()
