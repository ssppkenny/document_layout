#!/usr/bin/env python3
"""
Test script to verify line detection works correctly for all test cases
"""

import subprocess
import sys

test_cases = [
    ("notebooks/out0.png", 12, "12-line paragraph"),
    ("images/kf_16_par.png", 7, "7-line paragraph"),
    ("images/out2.png", 7, "7-line paragraph"),
    ("notebooks/out3.png", 5, "5-line paragraph"),
    ("images/out5.png", 6, "6-line paragraph"),
]

print("=" * 80)
print("LINE DETECTION TEST SUITE")
print("=" * 80)

all_passed = True

for image_path, expected_lines, description in test_cases:
    print(f"\nTesting: {image_path} ({description})")
    print(f"Expected: {expected_lines} lines")

    # Run diagnostic script
    result = subprocess.run(
        ["pixi", "run", "python", "diagnose_segmentation.py", image_path],
        capture_output=True,
        text=True
    )

    # Parse output to find detected lines
    for line in result.stdout.split('\n'):
        if "Detected lines (by margin detection, AFTER MERGING):" in line:
            detected_lines = int(line.split(':')[1].strip().split()[0])

            if detected_lines == expected_lines:
                print(f"✓ PASS: Detected {detected_lines} lines")
            else:
                print(f"✗ FAIL: Detected {detected_lines} lines (expected {expected_lines})")
                all_passed = False
            break
    else:
        print(f"✗ FAIL: Could not parse output")
        all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    sys.exit(1)
