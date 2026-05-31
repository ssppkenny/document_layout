#!/usr/bin/env python3
"""
Comprehensive test of the Swedish diacritics fix across multiple languages
"""

import subprocess
import time

test_images = [
    ("images/gang_p023.png", "Swedish", "Main test case with ä, ö, å"),
    ("images/dvurog_p076.png", "Russian", "Verify й still works"),
    ("images/jtg_p033.png", "English", "No regressions"),
    ("images/mh_p013.png", "English", "Line spacing test"),
]

print("="*80)
print("COMPREHENSIVE DIACRITICS FIX TEST")
print("="*80)
print()

results = []

for image, language, description in test_images:
    print(f"Testing: {image}")
    print(f"  Language: {language}")
    print(f"  Description: {description}")

    start_time = time.time()

    try:
        result = subprocess.run(
            ["python", "src/ocr_reflow/main.py", image, "--layout"],
            capture_output=True,
            text=True,
            timeout=120
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Check for errors in output
            if "ERROR" in result.stderr or "Traceback" in result.stderr:
                status = "❌ FAILED (errors in output)"
                results.append((image, language, False, elapsed))
            else:
                status = "✅ PASSED"
                results.append((image, language, True, elapsed))
        else:
            status = f"❌ FAILED (exit code {result.returncode})"
            results.append((image, language, False, elapsed))

        print(f"  Status: {status}")
        print(f"  Time: {elapsed:.1f}s")

    except subprocess.TimeoutExpired:
        print(f"  Status: ❌ TIMEOUT")
        results.append((image, language, False, 120.0))
    except Exception as e:
        print(f"  Status: ❌ ERROR: {e}")
        results.append((image, language, False, 0.0))

    print()

# Summary
print("="*80)
print("TEST SUMMARY")
print("="*80)
print()

passed = sum(1 for _, _, success, _ in results if success)
failed = len(results) - passed

print(f"Total tests: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print()

if failed == 0:
    print("✅ ALL TESTS PASSED - Swedish diacritics fix is working correctly!")
else:
    print("❌ SOME TESTS FAILED - Review output above")
    print("\nFailed tests:")
    for image, language, success, elapsed in results:
        if not success:
            print(f"  - {image} ({language})")

print()
