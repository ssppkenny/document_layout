"""Post-hoc hunspell correction on an already-built EPUB, using book
vocabulary and trigram language model to disambiguate multi-suggestion words.

Operates on plain text only (extracted from HTML, leaving markup intact).

Strategy:
- Split HTML into tag/non-tag segments.
- Build vocabulary of correctly-spelled words from plain text of the entire book.
- For each misspelled word with multiple hunspell suggestions, find the
  suggestion(s) with minimum edit distance.
- If exactly one of those appears elsewhere in the book's vocabulary → apply it.
- If multiple match, use a trigram LM (trained on book text) to score
  candidates by context: P(candidate | left_context).
- Still ambiguous → skip (conservative).

Usage:
    python fix_epub_spelling.py <epub> <lang> [-o output.epub]
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path


def _edit_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for ca in a:
        cur = [prev[0] + 1]
        for cb, pb in zip(b, prev[1:]):
            cur.append(min(prev[0] + (ca != cb), prev[1] + 1, cur[-1] + 1))
            prev = prev[1:]
        prev = cur
    return prev[-1]


def _split_html(html: str) -> list[dict]:
    """Split HTML into tag/plain segments.

    Returns list of {"type": "tag"|"text", "content": str}.
    """
    parts = re.split(r"(<[^>]*>)", html)
    result = []
    for p in parts:
        if not p:
            continue
        if p.startswith("<") and p.endswith(">"):
            result.append({"type": "tag", "content": p})
        else:
            result.append({"type": "text", "content": p})
    return result


def _plain_text(html: str) -> str:
    """Extract all plain text from HTML (strip tags)."""
    return re.sub(r"<[^>]*>", "", html)


def _words(text: str) -> set[str]:
    return {m.group().lower() for m in re.finditer(r"[а-яёА-ЯЁa-zA-Z]+", text)}


def _build_vocab(zf: zipfile.ZipFile) -> Counter:
    vocab: Counter = Counter()
    for name in zf.namelist():
        if not (name.endswith(".xhtml") or name.endswith(".ncx")):
            continue
        data = zf.read(name).decode("utf-8")
        plain = _plain_text(data)
        for w in _words(plain):
            vocab[w] += 1
    return vocab


def _hunspell_misspelled(text: str, lang: str) -> set[str]:
    proc = subprocess.run(
        ["hunspell", "-d", lang, "-l"],
        input=text, capture_output=True, text=True, timeout=60,
    )
    if proc.returncode not in (0, 1):
        return set()
    return set(proc.stdout.split())


def _hunspell_suggestions(word: str, lang: str) -> list[str]:
    proc = subprocess.run(
        ["hunspell", "-d", lang],
        input=word, capture_output=True, text=True, timeout=10,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("& "):
            after_colon = line.split(":", 1)
            if len(after_colon) == 2:
                return [s.strip() for s in after_colon[1].split(",")]
    return []


def _preserve_case(original: str, suggestion: str) -> str:
    if original.isupper():
        return suggestion.upper()
    if original[0].isupper() and len(original) > 1 and original[1:].islower():
        return suggestion.capitalize()
    return suggestion


def _corrections_for_text(
    text: str, lang: str, vocab: Counter,
    lm: _TrigramLM | None = None,
) -> dict[str, str]:
    """Find corrections for all misspelled words in *text* using vocab context."""
    misspelled = _hunspell_misspelled(text, lang)
    if not misspelled:
        return {}

    corrections: dict[str, str] = {}

    for word in sorted(misspelled, key=len, reverse=True):
        if not word.isalpha():
            continue
        # Protect actual dictionary words that appear 2+ times in the book —
        # likely names or valid terms.  Hunspell-flagged misspellings are NOT
        # protected even if frequent (handles пушина appearing on every page).
        _hunspell_check = subprocess.run(
            ["hunspell", "-d", lang, "-l"],
            input=word, capture_output=True, text=True, timeout=10,
        )
        if _hunspell_check.returncode in (0, 1) and not _hunspell_check.stdout.strip():
            if vocab.get(word.lower(), 0) > 1:
                continue

        suggestions = _hunspell_suggestions(word, lang)
        if not suggestions:
            continue

        min_dist = min(_edit_distance(word.lower(), s.lower()) for s in suggestions)
        max_dist = 1 if len(word) < 8 else 2
        if min_dist > max_dist:
            continue

        best = [
            s for s in suggestions
            if _edit_distance(word.lower(), s.lower()) == min_dist
        ]

        correction = None
        if len(best) == 1:
            correction = best[0]
        elif len(best) > 1:
            in_vocab = [s for s in best if s.lower() in vocab]
            if len(in_vocab) == 1:
                correction = in_vocab[0]
            elif len(in_vocab) > 1:
                # 1) Prefer candidates sharing first character with the
                #    misspelling — OCR rarely changes the first letter.
                first_match = [
                    s for s in in_vocab
                    if s[0].lower() == word[0].lower()
                ]
                if len(first_match) == 1:
                    correction = first_match[0]
                elif len(first_match) > 1:
                    in_vocab = first_match
                    # fall through to LM below
                # 2) Trigram / bigram context scoring
                if correction is None and lm is not None:
                    ctx_words = _left_context(text, word)
                    if ctx_words:
                        scored = [(s, lm.score(s, ctx_words)) for s in in_vocab]
                        valid = [(s, sc) for s, sc in scored if sc > -float("inf")]
                        if len(valid) == 1:
                            correction = valid[0][0]
                        elif len(valid) > 1:
                            best_ctx = max(valid, key=lambda x: x[1])
                            correction = best_ctx[0]

        if correction is not None:
            # Never insert spaces (prevents splitting compound words)
            if " " in correction:
                continue
            # For capitalized words (likely proper names): block if correction
            # changes the first two letters (Мінковского → Московского)
            if word[0].isupper() and len(word) >= 2 and len(correction) >= 2:
                if word[:2].lower() != correction[:2].lower():
                    continue
            # Don't correct very short words (high risk of false positives)
            if len(word) < 4 and len(correction) < 4:
                continue
            # Reject corrections that are not real words
            if not _is_known_word(correction, lang, vocab):
                continue
            corrections[word] = _preserve_case(word, correction)

    return corrections


def _is_known_word(word: str, lang: str, vocab: Counter) -> bool:
    """True if *word* is either in the book vocabulary or in hunspell's dict."""
    if word.lower() in vocab:
        return True
    proc = subprocess.run(
        ["hunspell", "-d", lang, "-l"],
        input=word, capture_output=True, text=True, timeout=10,
    )
    # hunspell -l lists misspelled words. Empty output = word is correct.
    return proc.returncode in (0, 1) and not proc.stdout.strip()


class _TrigramLM:
    """Trigram language model built from book text.

    Scores a word by its probability in context using stupid backoff:
    P(w | w₁,w₂) → P(w | w₂) → P(w).
    """

    def __init__(self, text: str):
        self._unigrams: Counter = Counter()
        self._bigrams: Counter = Counter()
        self._trigrams: Counter = Counter()
        self._total = 0

        for sent in re.split(r"[.!?]\s+", text):
            tokens = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", sent.lower())
            for t in tokens:
                self._unigrams[t] += 1
                self._total += 1
            for a, b in zip(tokens, tokens[1:]):
                self._bigrams[(a, b)] += 1
            for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
                self._trigrams[(a, b, c)] += 1

    def score(self, word: str, left: list[str]) -> float:
        """Log probability P(word | left[-2:]), no unigram fallback.

        Returns -inf when the context pattern is unseen — forces conservative
        skip rather than guessing from unigram frequency alone.
        """
        w = word.lower()
        if len(left) >= 2:
            ctx = (left[-2].lower(), left[-1].lower(), w)
            num = self._trigrams.get(ctx, 0)
            den = self._bigrams.get((left[-2].lower(), left[-1].lower()), 0)
            if num > 0 and den > 0:
                return math.log(num / den)
        if len(left) >= 1:
            ctx = (left[-1].lower(), w)
            num = self._bigrams.get(ctx, 0)
            den = self._unigrams.get(left[-1].lower(), 0)
            if num > 0 and den > 0:
                return math.log(num / den)
        return -float("inf")


def _left_context(text: str, word: str, n: int = 3) -> list[str]:
    """Return up to *n* words before the first occurrence of *word* in *text*."""
    idx = text.lower().find(word.lower())
    if idx == -1:
        return []
    before = text[:idx]
    tokens = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", before)
    return tokens[-n:] if len(tokens) >= n else tokens


def fix_spelling_in_html(html: str, corrections: dict[str, str]) -> str:
    """Apply hunspell corrections to an HTML string, preserving markup."""
    if not corrections:
        return html

    # Apply corrections to each text segment (leave tags untouched)
    segments = _split_html(html)
    result = []
    for seg in segments:
        if seg["type"] == "tag":
            result.append(seg["content"])
        else:
            text = seg["content"]
            for wrong, right in corrections.items():
                text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text)
            result.append(text)

    return "".join(result)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fix spelling in EPUB using hunspell + book-vocabulary context",
    )
    parser.add_argument("epub", type=str, help="Path to EPUB file")
    parser.add_argument("lang", type=str, help="Hunspell language (e.g. ru_RU)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show correction list")
    args = parser.parse_args()

    epub_path = Path(args.epub)
    output_path = Path(args.output) if args.output else epub_path

    # Validate hunspell
    try:
        proc = subprocess.run(
            ["hunspell", "-d", args.lang, "-l"],
            input="test", capture_output=True, text=True, timeout=10,
        )
        if proc.returncode not in (0, 1):
            print(f"Error: hunspell '{args.lang}' failed: {proc.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print("Error: hunspell not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {epub_path} ...", file=sys.stderr)

    with zipfile.ZipFile(epub_path, "r") as z:
        vocab = _build_vocab(z)
        print(f"Vocabulary: {len(vocab)} unique words", file=sys.stderr)

        # Gather all plain text for LM and corrections
        all_plain = ""
        for item in z.infolist():
            if item.filename.endswith(".xhtml"):
                data = z.read(item.filename).decode("utf-8")
                all_plain += _plain_text(data) + "\n"

        lm = _TrigramLM(all_plain)

        corrections = _corrections_for_text(all_plain, args.lang, vocab, lm=lm)
        print(f"Corrections: {len(corrections)}", file=sys.stderr)

        if args.verbose:
            for wrong, right in sorted(
                corrections.items(), key=lambda x: -len(x[0]),
            ):
                print(f"  {wrong:30s} → {right}", file=sys.stderr)

        if not corrections:
            print("No corrections needed.", file=sys.stderr)
            return

        # Second pass: apply corrections to each file (count only)
        corrected_files = 0
        for item in z.infolist():
            if not (item.filename.endswith(".xhtml") or item.filename.endswith(".ncx")):
                continue
            data = z.read(item.filename).decode("utf-8")
            fixed = fix_spelling_in_html(data, corrections)
            if fixed != data:
                corrected_files += 1

        if corrected_files == 0:
            print("No corrections matched in file output.", file=sys.stderr)
            return

        # Re-pack EPUB
        tmp = output_path.with_suffix(".fix.tmp")
        try:
            with zipfile.ZipFile(epub_path, "r") as zin:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                    for item in zin.infolist():
                        data = zin.read(item.filename)
                        if item.filename.endswith((".xhtml", ".ncx", ".opf")):
                            fixed = fix_spelling_in_html(
                                data.decode("utf-8"), corrections,
                            )
                            data = fixed.encode("utf-8")
                        zout.writestr(item, data)

            shutil.move(tmp, output_path)
            print(f"Written {output_path}", file=sys.stderr)
        except:
            if tmp.exists():
                tmp.unlink()
            raise


if __name__ == "__main__":
    main()
