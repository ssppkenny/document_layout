#! /usr/bin/env python3
"""Spell-check Russian text in an EPUB using Hunspell.

Usage:
    python spellcheck_epub.py creativity_ru.epub -o creativity_ru_checked.epub --fix
"""

from __future__ import annotations

import argparse
import html as html_mod
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF\u0500-\u052F\-]+")

CUSTOM_DICT = {
    "креативность", "креативности", "креативностью",
    "креативному", "креативном", "креативная", "креативную",
    "креативной", "креативное", "креативные", "креативных",
    "креативным", "креативными", "креативного",
    "франклин", "бестселлер",
}


_HUNSPELL_CHUNK = 2000

def hunspell_check_words(words: list[str]) -> dict[str, list[str]]:
    """Check words via hunspell. Process one at a time for reliability."""
    errors: dict[str, list[str]] = {}

    for idx, word in enumerate(words):
        if idx % 500 == 0 and idx > 0:
            print(f"  ... checked {idx}/{len(words)} words", file=sys.stderr)

        result = subprocess.run(
            ["hunspell", "-d", "ru_RU", "-a"],
            input=word + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Last non-empty, non-@ line is the result for this word
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and not l.startswith("@")]
        line = lines[-1] if lines else ""

        if line.startswith(("*", "-", "+")):
            # correct
            continue
        if line.startswith("&"):
            # & word count offset: sug1, sug2, ...
            if ":" in line:
                sug_part = line.split(":", 1)[1].strip()
                errors[word] = [s.strip() for s in sug_part.split(", ")]
        elif line.startswith("#"):
            errors[word] = []

    return errors


def strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", "", html)
    return html_mod.unescape(text)


def case_preserving(original: str, correction: str) -> str:
    if not original or not correction:
        return correction
    if original[0].isupper() and correction[0].islower():
        return correction[0].upper() + correction[1:]
    if original.isupper() and len(original) > 1:
        return correction.upper()
    return correction


def main():
    parser = argparse.ArgumentParser(description="Spell-check Russian EPUB")
    parser.add_argument("epub", type=str, help="Input EPUB file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output EPUB path")
    parser.add_argument("--fix", action="store_true", help="Auto-fix misspellings")
    args = parser.parse_args()

    epub_path = Path(args.epub)
    if not args.output:
        output_path = epub_path.parent / f"{epub_path.stem}_checked.epub"
    else:
        output_path = Path(args.output)

    print(f"Reading {epub_path} ...", file=sys.stderr)
    with zipfile.ZipFile(epub_path, "r") as z:
        names = z.namelist()
        file_data: dict[str, bytes] = {}
        text_files: list[str] = []
        for n in names:
            data = z.read(n)
            if n.endswith((".xhtml", ".ncx", ".xml", ".opf")):
                text_files.append(n)
            else:
                file_data[n] = data

        word_counter: Counter[str] = Counter()
        html_by_file: dict[str, str] = {}

        for n in text_files:
            html = z.read(n).decode("utf-8", errors="replace")
            html_by_file[n] = html
            plain = strip_html(html)
            words = [w.lower() for w in CYRILLIC_RE.findall(plain)]
            word_counter.update(words)

        unique_words = sorted(
            w for w in word_counter if len(w) > 1 and w not in CUSTOM_DICT
        )
        print(f"Found {len(unique_words):,} unique Russian words to check", file=sys.stderr)

        # Batch-check all words via hunspell
        errors = hunspell_check_words(unique_words)
        print(f"Found {len(errors):,} misspelled words", file=sys.stderr)

        # Categorize for report
        proper_names: list[tuple[str, int, list[str]]] = []
        real_errors: list[tuple[str, int, list[str]]] = []
        garbage: list[tuple[str, int, list[str]]] = []
        ocr_artifacts: list[tuple[str, int, list[str]]] = []

        # Check original title-case for proper names
        title_case_set: set[str] = set()
        for n in text_files:
            html = html_by_file[n]
            plain = strip_html(html)
            for m in CYRILLIC_RE.finditer(plain):
                w = m.group()
                if w[0].isupper() and len(w) > 1:
                    title_case_set.add(w.lower())

        for word, suggestions in sorted(errors.items()):
            count = word_counter.get(word, 0)
            if word in title_case_set:
                proper_names.append((word, count, suggestions))
            elif len(word) > 25:
                garbage.append((word, count, suggestions))
            elif word.endswith(("оо", "ув", "уо", "увс", "аах", "уах", "ею", "ою", "ойю")):
                ocr_artifacts.append((word, count, suggestions))
            else:
                real_errors.append((word, count, suggestions))

        # Report
        if errors:
            def print_group(header: str, items: list, max_show: int = 25):
                if not items:
                    return
                print(f"\n{header} ({len(items)} words)", file=sys.stderr)
                print("-" * 60, file=sys.stderr)
                for word, count, suggestions in items[:max_show]:
                    sug_str = ", ".join(suggestions[:2]) if suggestions else "(none)"
                    print(f"  {word:<28} {count:>3}  → {sug_str}", file=sys.stderr)
                if len(items) > max_show:
                    print(f"  ... and {len(items) - max_show} more", file=sys.stderr)

            print_group("PROPER NAMES", proper_names, 15)
            print_group("REAL MISSPELLINGS", real_errors, 30)
            print_group("OCR/TRANSLATION ARTIFACTS", ocr_artifacts, 15)
            print_group("GARBAGE (long strings)", garbage, 10)
            print(file=sys.stderr)

        # Apply fixes
        if args.fix and errors:
            corrections = {k: v[0] for k, v in errors.items() if v}
            print(f"Applying {len(corrections):,} corrections ...", file=sys.stderr)

            def replace_word(m: re.Match) -> str:
                original = m.group()
                lower = original.lower()
                if lower in corrections:
                    return case_preserving(original, corrections[lower])
                return original

            for n in text_files:
                html = html_by_file[n]
                html = CYRILLIC_RE.sub(replace_word, html)
                html_by_file[n] = html

            tmp_path = output_path.with_suffix(".spellcheck.tmp.epub")
            print(f"Writing {output_path} ...", file=sys.stderr)
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
                mi = zipfile.ZipInfo("mimetype")
                mi.compress_type = zipfile.ZIP_STORED
                zout.writestr(mi, file_data.get("mimetype", b"application/epub+zip"))
                for n in names:
                    if n in text_files:
                        data = html_by_file.get(n, "").encode("utf-8")
                    else:
                        data = file_data.get(n)
                    if data is not None:
                        zout.writestr(n, data)
            shutil.move(tmp_path, output_path)
            print(f"Done: {output_path}", file=sys.stderr)

    total_occ = sum(word_counter[w] for w in errors)
    print(f"Total: {len(errors)} unique misspelled words, {total_occ} total occurrences")
    if args.fix:
        print(f"Fixed → {output_path}")


if __name__ == "__main__":
    main()
