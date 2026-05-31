#!/usr/bin/env python3
"""
Test to verify Swedish character fixes
Checks for the specific issues mentioned: Börja → Böörja, inför → inföör
"""

import sys

print("=======================================================================")
print("SWEDISH CHARACTER FIX VERIFICATION")
print("=======================================================================")
print()
print("Test Image: images/gang_p023.png")
print()
print("Known Issues (BEFORE fix):")
print("  1. ö becomes öö (letter doubled)")
print("  2. ä becomes äi (split in middle)")
print("  3. å is split")
print("  4. 'i' with dot splits as 'i' then '.'")
print()
print("Expected words with problems:")
print("  - Börja → Böörja")
print("  - inför → inföör")
print()
print("=======================================================================")
print("ANALYSIS RESULTS:")
print("=======================================================================")
print()

# Read analysis results
import subprocess
result = subprocess.run(
    ["pixi", "run", "python", "analyze_swedish_words.py"],
    capture_output=True,
    text=True,
    timeout=120
)

# Count diacritics detected
lines = result.stdout.split('\n')
tall_diacritics = sum(1 for line in lines if 'DIACRITIC(TALL)' in line)
standard_diacritics = sum(1 for line in lines if line.strip().endswith('[DIACRITIC]'))

print(f"✓ Detected {tall_diacritics} tall narrow diacritics (Swedish ö, ä dots)")
print(f"✓ Detected {standard_diacritics} standard diacritics (dots on i, j, etc.)")
print()

# Check word 7 specifically (had the double-dot issue)
word7_section = []
in_word7 = False
for line in lines:
    if 'EXAMINING WORD 7' in line:
        in_word7 = True
    elif 'EXAMINING WORD' in line and 'WORD 7' not in line:
        in_word7 = False
    if in_word7:
        word7_section.append(line)

print("Word 7 Analysis (likely contains ö):")
print("-" * 60)
diacritic_lines = [l for l in word7_section if 'DIACRITIC' in l]
for line in diacritic_lines:
    print(line)

if any('DIACRITIC(TALL)' in l for l in diacritic_lines):
    count = sum(1 for l in diacritic_lines if 'DIACRITIC(TALL)' in l)
    print()
    print(f"✅ SUCCESS: {count} tall narrow diacritic(s) detected in word 7")
    print("   These will be merged with the base letter (ö), preventing doubling")
else:
    print()
    print("❌ PROBLEM: No tall diacritics detected - ö may still be doubled")

print()
print("=======================================================================")
print("REFLOW TEST:")
print("=======================================================================")
print()
print("Running full reflow...")

result = subprocess.run(
    ["pixi", "run", "python", "src/ocr_reflow/main.py", "images/gang_p023.png", "--layout"],
    capture_output=True,
    text=True,
    timeout=180
)

if result.returncode == 0:
    print("✅ Reflow completed successfully")
    print(f"   Output saved to: output_reflowed.png")
    print()
    print("To visually verify the fix:")
    print("  1. Open output_reflowed.png")
    print("  2. Look for words with ö, ä, å")
    print("  3. Check that letters are not doubled or split")
    print("  4. Check that 'i' dots stay above the letter")
else:
    print(f"❌ Reflow failed with return code {result.returncode}")
    print("Error output:")
    print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

print()
print("=======================================================================")
print("SUMMARY:")
print("=======================================================================")
print()
print("The fix adds detection for 'tall narrow diacritics' which are common")
print("in Swedish and other Scandinavian languages. These diacritics are:")
print("  - Narrow relative to their height (w < h * 0.5)")
print("  - Small area relative to word size (< 10% of word area)")
print("  - Located in top 60% of word")
print()
print("This prevents the two dots in ö/ä from being treated as separate")
print("letters and ensures they're merged with the base character.")
print()
