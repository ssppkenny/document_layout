#! /usr/bin/env python3
"""Translate an EPUB to a target language using a T5 or NLLB model.

Usage:
    python translate_epub.py creativity.epub -l ru -o creativity_ru.epub         # T5 (default)
    python translate_epub.py creativity.epub -l ru -o creativity_nllb.epub --model nllb

The model is loaded on CPU by default to leave GPU free for OCR.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer, NllbTokenizer, T5ForConditionalGeneration, T5Tokenizer


# Flores language codes for NLLB
_FLORES_CODES = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "zh": "zho_Hans",
}

# ---------------------------------------------------------------------------
# HTML tag/text splitter
# ---------------------------------------------------------------------------

def _split_html(html: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> list[str]:
    return _SENTENCE_RE.split(text)


# ---------------------------------------------------------------------------
# Translation engine – abstract base
# ---------------------------------------------------------------------------

class Translator:
    def __init__(self, target_lang: str, device: str = "cpu"):
        self.target_lang = target_lang
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        raise NotImplementedError

    def _model_generate(self, texts: list[str], **gen_kw) -> list[str]:
        raise NotImplementedError

    @torch.no_grad()
    def translate(self, text: str) -> str:
        self._ensure_model()
        if not text.strip():
            return text
        results = self._model_generate([text])
        return results[0]

    @torch.no_grad()
    def translate_batch(self, texts: list[str], batch_size: int = 4) -> list[str]:
        """Translate multiple texts in batched model calls."""
        self._ensure_model()
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return texts

        results: dict[int, str] = {}
        for chunk_start in range(0, len(non_empty), batch_size):
            chunk = non_empty[chunk_start : chunk_start + batch_size]
            decoded = self._model_generate([t for _, t in chunk])
            for (orig_idx, _), d in zip(chunk, decoded):
                results[orig_idx] = d

        return [results.get(i, t) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# T5 translator
# ---------------------------------------------------------------------------

class T5Translator(Translator):
    MODEL_NAME = "utrobinmv/t5_translate_en_ru_zh_base_200"

    def _ensure_model(self):
        if self._model is not None:
            return
        print(f"Loading T5 model on {self.device}...", file=sys.stderr)
        t0 = time.perf_counter()
        self._model = T5ForConditionalGeneration.from_pretrained(self.MODEL_NAME)
        self._model.to(self.device)
        self._model.eval()
        self._tokenizer = T5Tokenizer.from_pretrained(self.MODEL_NAME)
        print(f"  done ({time.perf_counter()-t0:.1f}s)", file=sys.stderr)

    def _model_generate(self, texts: list[str], **gen_kw) -> list[str]:
        prefix = f"translate to {self.target_lang}: "
        inputs = self._tokenizer(
            [prefix + t for t in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            **gen_kw,
        )
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# NLLB translator
# ---------------------------------------------------------------------------

class NLLBTranslator(Translator):
    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def _ensure_model(self):
        if self._model is not None:
            return
        print(f"Loading NLLB model on {self.device}...", file=sys.stderr)
        t0 = time.perf_counter()
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else None,
        )
        self._model.to(self.device)
        self._model.eval()
        self._tokenizer = NllbTokenizer.from_pretrained(self.MODEL_NAME, src_lang="eng_Latn")
        print(f"  done ({time.perf_counter()-t0:.1f}s)", file=sys.stderr)

    def _model_generate(self, texts: list[str], **gen_kw) -> list[str]:
        tgt_code = _FLORES_CODES.get(self.target_lang, "rus_Cyrl")
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self._model.generate(
            **inputs,
            forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.1,
            **gen_kw,
        )
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# M2M-100 translator
# ---------------------------------------------------------------------------

class M2M100Translator(Translator):
    MODEL_NAME = "facebook/m2m100_418M"

    def _ensure_model(self):
        if self._model is not None:
            return
        print(f"Loading M2M-100 model on {self.device}...", file=sys.stderr)
        t0 = time.perf_counter()
        self._model = M2M100ForConditionalGeneration.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else None,
        )
        self._model.to(self.device)
        self._model.eval()
        self._tokenizer = M2M100Tokenizer.from_pretrained(self.MODEL_NAME, src_lang="en")
        print(f"  done ({time.perf_counter()-t0:.1f}s)", file=sys.stderr)

    def _model_generate(self, texts: list[str], **gen_kw) -> list[str]:
        tgt_code = self._tokenizer.get_lang_id(self.target_lang)
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self._model.generate(
            **inputs,
            forced_bos_token_id=tgt_code,
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            **gen_kw,
        )
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Translate one XHTML file: tag/text segmentation approach
# ---------------------------------------------------------------------------

_MATH_RE = re.compile(r"\$[^$]*?\$|\$\$[^$]*?\$\$")

_MAX_CHARS = 800  # split texts longer than this at sentence boundaries
# Tags that separate text into independent translation groups
_BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "div", "blockquote", "td", "th", "dt", "dd", "figcaption",
    "html", "head", "body", "section", "nav", "article", "aside",
    "table", "tr", "thead", "tbody", "tfoot", "colgroup", "caption",
    "ul", "ol", "dl", "menu",
    "header", "footer", "main", "figure", "figcaption",
    "style", "script", "meta", "link", "title",
}


def _is_block_tag(tag: str) -> bool:
    m = re.match(r"</?(\w+)", tag)
    return m is not None and m.group(1).lower() in _BLOCK_TAGS


def _should_skip(text: str) -> bool:
    if not text.strip():
        return True
    return bool(re.match(
        r"^[\d\s\-–—.,;:!?()\[\]{}\"'«»№#*/\\&%@$€£+=\^_~|<>`\n\r]*$", text,
    ))


def _translate_html(html: str, translator: Translator) -> str:
    segments = _split_html(html)

    # Group consecutive text segments separated only by non-block tags.
    groups: list[list[int]] = []
    current: list[int] = []

    for i, seg in enumerate(segments):
        if seg["type"] == "text":
            current.append(i)
        elif _is_block_tag(seg["content"]):
            if current:
                groups.append(current)
                current = []

    if current:
        groups.append(current)

    # Collect all text groups, split long ones at sentence boundaries
    group_texts: list[str] = []
    group_indices: list[int] = []  # index into groups list
    for gi, group in enumerate(groups):
        full_text = "".join(segments[i]["content"] for i in group).strip()
        if full_text and not _should_skip(full_text):
            if len(full_text) > _MAX_CHARS:
                chunks = _split_sentences(full_text)
                # Merge small chunks back to ~_MAX_CHARS each
                merged: list[str] = []
                buf: list[str] = []
                buf_len = 0
                for c in chunks:
                    if buf_len + len(c) > _MAX_CHARS and buf:
                        merged.append(" ".join(buf))
                        buf = [c]
                        buf_len = len(c)
                    else:
                        buf.append(c)
                        buf_len += len(c)
                if buf:
                    merged.append(" ".join(buf))
                group_texts.extend(merged)
                group_indices.extend([gi] * len(merged))
            else:
                group_texts.append(full_text)
                group_indices.append(gi)

    translations = translator.translate_batch(group_texts) if group_texts else []

    # Collect translations per group (may be multiple chunks per group)
    from collections import defaultdict
    per_group: dict[int, list[str]] = defaultdict(list)
    for ri, gi in enumerate(group_indices):
        if ri < len(translations):
            per_group[gi].append(translations[ri])

    for gi, group in enumerate(groups):
        replacement = " ".join(per_group.get(gi, [""]))
        for i in group:
            segments[i]["content"] = ""
        segments[group[0]]["content"] = replacement

    result = "".join(s["content"] for s in segments)
    # Strip empty inline tags left behind
    result = re.sub(r"<(b|i|em|strong|span)[^>]*>\s*</\1>", "", result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Translate an EPUB using T5 or NLLB")
    parser.add_argument("epub", type=str, help="Input EPUB file")
    parser.add_argument("-l", "--lang", default="ru", help="Target language code (ru, zh, en)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output EPUB path")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--model", default="t5", choices=["t5", "nllb", "m2m100"],
                        help="Translation model: t5 (default), nllb, or m2m100")
    args = parser.parse_args()

    epub_path = Path(args.epub)
    stem = f"{epub_path.stem}_{args.lang}{'_' + args.model if args.model != 't5' else ''}"
    output_path = Path(args.output) if args.output else epub_path.parent / f"{stem}.epub"

    cls = {"t5": T5Translator, "nllb": NLLBTranslator, "m2m100": M2M100Translator}[args.model]
    translator = cls(args.lang, device=args.device)
    translator._ensure_model()

    print(f"Reading {epub_path} ...", file=sys.stderr)
    with zipfile.ZipFile(epub_path, "r") as z:
        names = z.namelist()

        # Separate files that need translation from those that don't
        file_data: dict[str, bytes] = {}
        xhtml_files: list[str] = []
        for n in names:
            data = z.read(n)
            if n.endswith((".xhtml", ".ncx")):
                xhtml_files.append(n)
            else:
                file_data[n] = data

        # Translate XHTML/NCX files
        total_chars = 0
        for n in xhtml_files:
            html = z.read(n).decode("utf-8")
            total_chars += len(html)

        t_start = time.perf_counter()
        print(f"Translating {len(xhtml_files)} files (~{total_chars:,} chars)...", file=sys.stderr)

        for idx, n in enumerate(xhtml_files, 1):
            print(f"  [{idx}/{len(xhtml_files)}] {n} ...", file=sys.stderr)
            html = z.read(n).decode("utf-8")
            translated = _translate_html(html, translator)
            file_data[n] = translated.encode("utf-8")

        elapsed = time.perf_counter() - t_start
        print(f"Translation done in {elapsed:.0f}s ({total_chars/elapsed:.0f} chars/s)", file=sys.stderr)

    # Update OPF language metadata
    opf = file_data.get("OEBPS/content.opf", b"").decode("utf-8")
    if opf:
        # Add or replace dc:language
        if '<dc:language>' in opf:
            opf = re.sub(r'<dc:language>[^<]*</dc:language>', f'<dc:language>{args.lang}</dc:language>', opf)
        else:
            opf = opf.replace('<dc:title>', f'<dc:language>{args.lang}</dc:language>\n  <dc:title>')
        file_data["OEBPS/content.opf"] = opf.encode("utf-8")

    # Re-pack EPUB
    tmp_path = output_path.with_suffix(".translate.tmp.epub")
    print(f"Writing {output_path} ...", file=sys.stderr)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        mi = zipfile.ZipInfo("mimetype")
        mi.compress_type = zipfile.ZIP_STORED
        zout.writestr(mi, file_data.get("mimetype", b"application/epub+zip"))
        for n in names:
            data = file_data.get(n)
            if data is not None:
                zout.writestr(n, data)

    shutil.move(tmp_path, output_path)
    print(f"Done: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
