#!/usr/bin/env python3
"""Wrap formulas in a group div + insert <br/> between SVGs for multi-line display.

CSS handles centering: the group is display:inline-block centered by parent,
with left-aligned content so all lines share the same x coordinate.
"""

import io
import re
import zipfile
from pathlib import Path


def process_xhtml(html_str: str) -> str:
    def process_block(m):
        inner = m.group(1)
        inner = re.sub(r'(</svg>)\s*(<svg)', r'\1<br/>\2', inner)
        return f'<div class="block formula"><div class="formula-group">{inner}</div></div>'

    return re.sub(
        r'<div class="block formula">(.*?)</div>',
        process_block,
        html_str,
        flags=re.DOTALL,
    )


def patch_css(css: str) -> str:
    # Remove old p.math-display rule if present
    css = re.sub(r'\.block\.formula\s+p\.math-display\s*\{[^}]*\}', '', css)
    # Add new group rule
    if '.formula-group' not in css:
        css += '\n.block.formula .formula-group {\n  display: inline-block;\n  text-align: left;\n}\n'
    return css


def process_epub(path: str):
    path = Path(path)
    out = path.parent / f"{path.stem}_grouped.epub"

    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                data = zin.read(name)
                if name.endswith('.xhtml'):
                    data = process_xhtml(data.decode("utf-8")).encode("utf-8")
                elif name == 'OEBPS/style.css':
                    data = patch_css(data.decode("utf-8")).encode("utf-8")
                compress = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
                zout.writestr(name, data, compress_type=compress)

    with open(out, "wb") as f:
        f.write(buf.getvalue())

    print(f"  Written: {out}")


if __name__ == "__main__":
    import sys
    process_epub(sys.argv[1] if len(sys.argv) > 1 else "magistr_rerendered.epub")
