#! /usr/bin/env python3
"""Translate a single chapter from creativity.epub with a given model.

Usage:
    python translate_chapter.py t5
    python translate_chapter.py nllb
    python translate_chapter.py m2m100
"""

from __future__ import annotations

import sys
import zipfile

from translate_epub import T5Translator, NLLBTranslator, M2M100Translator, _translate_html

CHAPTER_FILE = "OEBPS/chi-franklin-0007.xhtml"

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "t5"
    cls = {"t5": T5Translator, "nllb": NLLBTranslator, "m2m100": M2M100Translator}[model_name]

    with zipfile.ZipFile("creativity.epub") as z:
        html = z.read(CHAPTER_FILE).decode("utf-8")

    translator = cls("ru", device="cuda")
    translator._ensure_model()

    print(f"Translating {CHAPTER_FILE} with {model_name}...", file=sys.stderr)
    translated = _translate_html(html, translator)

    out_name = f"chapter_{model_name}.xhtml"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(translated)
    print(f"Written to {out_name}", file=sys.stderr)

    # Print text-only for quick inspection
    import re
    text_only = re.sub(r"<[^>]*>", "", translated)
    text_only = re.sub(r"\s+", " ", text_only).strip()
    print(f"\n=== {model_name.upper()} OUTPUT ({len(text_only)} chars) ===")
    print(text_only[:3000])
    print("...")

if __name__ == "__main__":
    main()
