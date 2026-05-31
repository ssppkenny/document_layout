#!/usr/bin/env python3
"""
patch_epub_toc.py — Rebuild the TOC of an existing EPUB from its printed TOC pages.

Usage:
    python patch_epub_toc.py <epub> --toc-pages N[,N...] [-o output.epub]

The script reads the TOC page sections already present in content.xhtml,
parses dot-leader entries (title . . . . N), and rewrites nav.xhtml + toc.ncx.
No DjVu, no OCR, no re-processing of body pages.
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces — for fuzzy matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def _edit1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long_)):
        if long_[:i] + long_[i + 1:] == short:
            return True
    return False


def _title_matches(name: str, candidate: str) -> bool:
    """True if all words (>=3 chars) of name fuzzy-match words in candidate."""
    nw = [w for w in _norm(name).split() if len(w) >= 3]
    cw = _norm(candidate).split()
    if not nw:
        return False
    return all(any(_edit1(n, c) for c in cw) for n in nw)


# ---------------------------------------------------------------------------
# Parse TOC pages from content.xhtml
# ---------------------------------------------------------------------------

def parse_toc_pages(content: str, toc_page_nums: list[int]) -> list[tuple[str, int]]:
    """
    Extract (title, page_number) pairs from the printed TOC sections.

    Strategy:
    1. Split content.xhtml into per-page sections.
    2. For each TOC page:
       a. block title divs → section headers (no page number in dot-leaders;
          we'll resolve their page via body-title scan).
       b. block text divs → parse dot-leader lines: "Title . . . . N"
    3. Return list of (title, printed_page_number) sorted by page number.
       Entries where the page number could not be parsed get printed_page=-1
       and will be resolved later via body-title scan.
    """
    # Split into sections keyed by page number
    section_re = re.compile(r'<section[^>]*id="page-(\d+)"[^>]*>(.*?)(?=<section|$)', re.DOTALL)
    sections: dict[int, str] = {}
    for m in section_re.finditer(content):
        sections[int(m.group(1))] = m.group(2)

    # Dot-leader line pattern: anything, then 3+ ". " sequences, then a number.
    # Unanchored so finditer can extract multiple entries from a single block
    # (the VLM often merges consecutive TOC lines into one text block).
    dot_leader_re = re.compile(r'(.+?)\s*(?:\.\s*){3,}(\d+)')

    entries: list[tuple[str, int]] = []   # (title, printed_page)
    section_headers: list[str] = []       # block title texts on TOC pages
    all_toc_titles: list[str] = []        # every title string seen on TOC pages (for sweep whitelist)

    for pn in toc_page_nums:
        sec_html = sections.get(pn, "")
        if not sec_html:
            print(f"  Warning: page {pn} not found in content.xhtml", file=sys.stderr)
            continue

        # Extract block title and block text divs
        block_re = re.compile(
            r'<div class="block (title|text)"[^>]*>(.*?)</div>', re.DOTALL
        )
        for bm in block_re.finditer(sec_html):
            kind = bm.group(1)
            raw = _strip_tags(bm.group(2)).strip()
            # Collapse multiple spaces/newlines for display but keep newlines for line splitting
            if kind == "title":
                title_text = re.sub(r"\s+", " ", raw).strip()
                # Skip "СОДЕРЖАНИЕ" / "ОГЛАВЛЕНИЕ" — these are the TOC heading itself
                if re.match(r"^(СОДЕРЖАНИЕ|ОГЛАВЛЕНИЕ|TABLE\s+OF\s+CONTENTS)$",
                            title_text, re.IGNORECASE):
                    continue
                section_headers.append(title_text)
                all_toc_titles.append(title_text)
            else:
                # The VLM often merges consecutive TOC lines into a single
                # text block without newlines.  Use finditer on the collapsed
                # text to extract ALL dot-leader entries.
                raw_clean = re.sub(r"\s+", " ", raw).strip()
                if not raw_clean:
                    continue
                matched_some = False
                for m2 in dot_leader_re.finditer(raw_clean):
                    matched_some = True
                    title = m2.group(1).strip().rstrip(",.;:")
                    pg = int(m2.group(2))
                    entries.append((title, pg))
                    all_toc_titles.append(title)
                # Fallback for text blocks with no parseable dot-leaders:
                # could be a title with dots swallowing the page number.
                if not matched_some and len(raw_clean) > 3:
                    entries.append((raw_clean, -1))
                    all_toc_titles.append(raw_clean)

    return section_headers, entries, all_toc_titles


# ---------------------------------------------------------------------------
# Body-title scan fallback
# ---------------------------------------------------------------------------

def find_page_by_title(name: str, sections: dict[int, str],
                       toc_page_nums: set[int]) -> int | None:
    """Scan body page sections for a block title that fuzzy-matches name."""
    for pn, sec_html in sorted(sections.items()):
        if pn in toc_page_nums:
            continue
        for m in re.finditer(
            r'<div class="block title"[^>]*>(.*?)</div>', sec_html, re.DOTALL
        ):
            candidate = re.sub(r"\s+", " ", _strip_tags(m.group(1))).strip()
            if _title_matches(name, candidate):
                return pn
    return None


# ---------------------------------------------------------------------------
# NAV / NCX builders
# ---------------------------------------------------------------------------

def build_nav(title: str, entries: list[tuple[str, int]]) -> str:
    items = "\n".join(
        f'      <li><a href="content.xhtml#page-{pg}">{_escape(t)}</a></li>'
        for t, pg in entries
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en">
<head><meta charset="utf-8"/><title>{_escape(title)}</title></head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>
{items}
    </ol>
  </nav>
</body>
</html>
"""


def build_ncx(title: str, uid: str, entries: list[tuple[str, int]]) -> str:
    nav_points = ""
    for i, (t, pg) in enumerate(entries, 1):
        nav_points += f"""  <navPoint id="np-{i}" playOrder="{i}">
    <navLabel><text>{_escape(t)}</text></navLabel>
    <content src="content.xhtml#page-{pg}"/>
  </navPoint>
"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="{_escape(uid)}"/>
    <meta name="dtb:depth" content="1"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle><text>{_escape(title)}</text></docTitle>
  <navMap>
{nav_points}  </navMap>
</ncx>
"""


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epub", help="Input EPUB file")
    parser.add_argument("--toc-pages", required=True,
                        help="Comma-separated page numbers of the printed TOC, e.g. '429,430'")
    parser.add_argument("-o", "--output", help="Output EPUB (default: overwrite input)")
    args = parser.parse_args()

    epub_path = Path(args.epub)
    output_path = Path(args.output) if args.output else epub_path
    toc_page_nums = [int(p.strip()) for p in args.toc_pages.split(",")]

    print(f"Reading {epub_path} ...", file=sys.stderr)
    with zipfile.ZipFile(epub_path) as zf:
        names = zf.namelist()
        content = zf.read("OEBPS/content.xhtml").decode()
        opf = zf.read("OEBPS/content.opf").decode()
        file_data = {n: zf.read(n) for n in names
                     if n not in ("OEBPS/nav.xhtml", "OEBPS/toc.ncx")}

    # Extract book title and UID from OPF
    title_m = re.search(r"<dc:title[^>]*>([^<]+)</dc:title>", opf)
    uid_m = re.search(r"<dc:identifier[^>]*>([^<]+)</dc:identifier>", opf)
    book_title = title_m.group(1).strip() if title_m else epub_path.stem
    book_uid = uid_m.group(1).strip() if uid_m else "unknown"

    # Build sections dict for body-title fallback
    section_re = re.compile(r'<section[^>]*id="page-(\d+)"[^>]*>(.*?)(?=<section|$)', re.DOTALL)
    sections: dict[int, str] = {int(m.group(1)): m.group(2) for m in section_re.finditer(content)}
    available_pages = set(sections.keys())
    toc_page_set = set(toc_page_nums)

    print(f"Parsing TOC pages {toc_page_nums} ...", file=sys.stderr)
    section_headers, raw_entries, all_toc_titles = parse_toc_pages(content, toc_page_nums)

    print(f"  Section headers: {section_headers}", file=sys.stderr)
    print(f"  Raw entries parsed: {len(raw_entries)}", file=sys.stderr)

    # Resolve section headers via body-title scan
    final_entries: list[tuple[str, int]] = []

    for hdr in section_headers:
        pg = find_page_by_title(hdr, sections, toc_page_set)
        if pg is not None:
            final_entries.append((hdr, pg))
            print(f"  [header] p{pg:3d}: {hdr}", file=sys.stderr)
        else:
            print(f"  [header] NOT FOUND: {hdr!r}", file=sys.stderr)

    # Resolve dot-leader entries
    for title, printed_pg in raw_entries:
        if printed_pg != -1:
            # Use printed page number directly (offset assumed 0 — verify below)
            if printed_pg in available_pages:
                final_entries.append((title, printed_pg))
            else:
                print(f"  [skip] p{printed_pg} not in EPUB: {title!r}", file=sys.stderr)
        else:
            # Fallback: body-title scan
            pg = find_page_by_title(title, sections, toc_page_set)
            if pg is not None:
                final_entries.append((title, pg))
                print(f"  [fallback] p{pg:3d}: {title!r}", file=sys.stderr)
            else:
                print(f"  [skip] no match: {title!r}", file=sys.stderr)

    # Pass 4: for any known TOC name not yet placed, try body-title scan.
    # This catches names that were parsed from dot-leaders but whose
    # find_page_by_title failed earlier (e.g. because the page was claimed),
    # AND names that appear in section_headers but not yet placed.
    placed_titles = {_norm(t) for t, _ in final_entries}
    for name in all_toc_titles:
        if _norm(name) in placed_titles:
            continue
        pg = find_page_by_title(name, sections, toc_page_set)
        if pg is not None:
            final_entries.append((name, pg))
            placed_titles.add(_norm(name))
            print(f"  [pass4]    p{pg:3d}: {name!r}", file=sys.stderr)

    # Pass 5: sweep unclaimed body block-title headings that fuzzy-match
    # a title from the printed TOC text. This catches chapter titles whose
    # dot-leader lines were completely lost to OCR (no text survived at all).
    claimed_pages = {pg for _, pg in final_entries}
    placed_titles_norm = {_norm(t) for t, _ in final_entries}
    placed_titles_orig = [t for t, _ in final_entries]
    for pn, sec_html in sorted(sections.items()):
        if pn in toc_page_set or pn in claimed_pages:
            continue
        for m in re.finditer(
            r'<div class="block title"[^>]*>(.*?)</div>', sec_html, re.DOTALL
        ):
            candidate = re.sub(r"\s+", " ", _strip_tags(m.group(1))).strip()
            candidate = candidate.rstrip(",.;:")
            if not candidate or _norm(candidate) in placed_titles_norm:
                continue
            # Fuzzy dedup: skip if candidate fuzzy-matches an already placed
            # title — prevents running headers from being added as duplicates.
            if any(
                _title_matches(candidate, t) or _title_matches(t, candidate)
                for t in placed_titles_orig
            ):
                continue
            # Only add if it fuzzy-matches a title from the printed TOC
            if any(
                _title_matches(candidate, k) or _title_matches(k, candidate)
                for k in all_toc_titles
            ):
                final_entries.append((candidate, pn))
                claimed_pages.add(pn)
                placed_titles_norm.add(_norm(candidate))
                placed_titles_orig.append(candidate)
                print(f"  [sweep]    p{pn:3d}: {candidate!r}", file=sys.stderr)
                break

    # Sort by page, deduplicate (keep first occurrence per page)
    seen_pages: set[int] = set()
    seen_titles: set[str] = set()
    toc: list[tuple[str, int]] = []
    for title, pg in sorted(final_entries, key=lambda x: x[1]):
        key = _norm(title)
        if pg not in seen_pages and key not in seen_titles:
            seen_pages.add(pg)
            seen_titles.add(key)
            toc.append((title, pg))

    print(f"\nTOC: {len(toc)} entries", file=sys.stderr)
    for t, pg in toc:
        print(f"  p{pg:4d}: {t}", file=sys.stderr)

    # Build new nav and ncx
    new_nav = build_nav(book_title, toc).encode()
    new_ncx = build_ncx(book_title, book_uid, toc).encode()

    # Write output EPUB
    tmp_path = output_path.with_suffix(".tmp.epub")
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        # mimetype must be first and uncompressed
        mi = zipfile.ZipInfo("mimetype")
        mi.compress_type = zipfile.ZIP_STORED
        zout.writestr(mi, file_data.pop("mimetype", b"application/epub+zip"))
        for name, data in file_data.items():
            zout.writestr(name, data)
        zout.writestr("OEBPS/nav.xhtml", new_nav)
        zout.writestr("OEBPS/toc.ncx", new_ncx)

    tmp_path.replace(output_path)
    print(f"\nEPUB written: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
