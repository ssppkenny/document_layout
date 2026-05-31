#!/usr/bin/env python3
"""Re-render SVG formulas in an EPUB from corrected data-latex attributes.

Usage:
    pixi run python rerender_svg.py magistr.epub -o magistr_fixed.epub
"""

import argparse
import html as html_mod
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from ocr_reflow.epub_export import MathRenderer, _uniquify_svg_ids

MATHJAX_JS = Path('src/ocr_reflow/static/mathjax/tex-svg-full.js').resolve()
SVG_RE = re.compile(
    r'(<svg\s[^>]*?data-latex="([^"]*)"[^>]*>)(.*?)</svg>',
    re.DOTALL,
)


def find_svgs(html_str: str) -> list[dict]:
    results = []
    for m in SVG_RE.finditer(html_str):
        results.append({
            'start': m.start(),
            'end': m.end(),
            'latex': html_mod.unescape(m.group(2)),
        })
    return results


def is_display(html_str: str, start: int, end: int) -> bool:
    before = html_str[:start]
    after = html_str[end:]
    p_open = before.rfind('<p')
    p_close = after.find('</p>')
    if p_open == -1 or p_close == -1:
        return False
    context = before[p_open:] + after[:p_close + 4]
    cleaned = re.sub(r'<svg[^>]*>.*?</svg>', '', context)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    return cleaned.strip() == ''


def process_epub(input_path: str, output_path: str) -> dict:
    renderer = MathRenderer(MATHJAX_JS)
    renderer.__enter__()
    try:
        xhtml_files = []
        total = 0
        unique = set()
        cache_hits = 0

        with zipfile.ZipFile(input_path, 'r') as zin:
            for name in zin.namelist():
                if name.endswith('.xhtml') or name.endswith('.ncx'):
                    xhtml_files.append(name)

            processed = {}
            for fname in xhtml_files:
                html_str = zin.read(fname).decode('utf-8')
                svgs = find_svgs(html_str)
                if not svgs:
                    processed[fname] = html_str
                    continue

                total += len(svgs)
                replacements = []
                for svg in svgs:
                    display = is_display(html_str, svg['start'], svg['end'])
                    key = (svg['latex'], display)
                    unique.add(key)
                    new_svg = renderer.render(svg['latex'], display)
                    replacements.append((svg['start'], svg['end'], new_svg))

                replacements.reverse()
                for start, end, new_svg in replacements:
                    html_str = html_str[:start] + new_svg + html_str[end:]

                processed[fname] = html_str

            cache_hits = total - len(unique)

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zout:
                for name in zin.namelist():
                    data = zin.read(name)
                    if name in processed:
                        data = processed[name].encode('utf-8')
                    compress = zipfile.ZIP_STORED if name == 'mimetype' else zipfile.ZIP_DEFLATED
                    zout.writestr(name, data, compress_type=compress)

        with open(output_path, 'wb') as f:
            f.write(buf.getvalue())

        return {'files': len(xhtml_files), 'svgs': total,
                'unique_formulas': len(unique), 'cache_hits': cache_hits}

    finally:
        renderer.__exit__(None, None, None)


def main():
    p = argparse.ArgumentParser(description='Re-render SVG formulas in EPUB')
    p.add_argument('input', help='Input EPUB file')
    p.add_argument('-o', '--output', default=None,
                   help='Output EPUB file (default: input_rerendered.epub)')
    args = p.parse_args()

    inp = Path(args.input).resolve()
    out = Path(args.output or inp.stem + '_rerendered.epub').resolve()

    t0 = time.time()
    stats = process_epub(str(inp), str(out))
    elapsed = time.time() - t0

    print(f'  Files processed: {stats["files"]}')
    print(f'  SVG formulas found: {stats["svgs"]}')
    print(f'  Unique formulas: {stats["unique_formulas"]}')
    print(f'  Cache hits (reused SVGs): {stats["cache_hits"]}')
    print(f'  Time: {elapsed:.1f}s')
    print(f'  Output: {out}')


if __name__ == '__main__':
    main()
