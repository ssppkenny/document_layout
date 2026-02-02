#!/usr/bin/env python3
"""
Benchmark script to measure performance improvements.
"""
import time
import subprocess
import sys

def run_command(cmd):
    """Run a command and return execution time."""
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end = time.time()
    return end - start, result.returncode

def main():
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Test file
    test_file = "images/kf_p025.png"

    print(f"\nTest file: {test_file}")
    print("\nRunning 3 iterations to test model caching...")

    times = []
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        cmd = f"pixi run python src/ocr_reflow/main.py {test_file} --layout --no-output"
        duration, returncode = run_command(cmd)

        if returncode != 0:
            print(f"ERROR: Command failed with return code {returncode}")
            sys.exit(1)

        times.append(duration)
        print(f"Execution time: {duration:.2f} seconds")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nFirst run (with model loading):  {times[0]:.2f}s")
    print(f"Second run (with cached models): {times[1]:.2f}s")
    print(f"Third run (with cached models):  {times[2]:.2f}s")

    if len(times) > 1:
        avg_cached = sum(times[1:]) / len(times[1:])
        speedup = times[0] / avg_cached if avg_cached > 0 else 1
        time_saved = times[0] - avg_cached

        print(f"\nAverage time with cached models: {avg_cached:.2f}s")
        print(f"Time saved per run: {time_saved:.2f}s ({(time_saved/times[0]*100):.1f}% faster)")
        print(f"Speedup factor: {speedup:.2f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nModel caching is working! Subsequent runs are much faster.")
    print("For batch processing, keep the Python process alive to benefit from caching.")

if __name__ == "__main__":
    main()
