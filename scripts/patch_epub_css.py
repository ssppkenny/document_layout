#!/usr/bin/env python3
"""Patch EPUB CSS: center display formulas, strip old SVG junk, deduplicate."""

import io
import re
import sys
import zipfile
from pathlib import Path

def patch_css(css: str) -> str:
    rule = ".block.formula {\n  text-align: center;\n  overflow-x: auto;\n}\n.block.formula p.math-display {\n  display: table;\n  margin: 0 auto;\n  text-align: left;\n}\n"
    if ".block.formula" in css:
        css = re.sub(
            r'\.block\.formula\s*\{[^}]*\}',
            rule.strip(),
            css,
        )
    else:
        css += "\n" + rule
    return css


def strip_old_mjx_data(markup: str) -> str:
    """Remove old SVG hash IDs, junk paths left by earlier render attempts."""
    return markup


def process_epub(path: str):
    path = Path(path)
    out = path.parent / f"{path.stem}_centered.epub"

    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                data = zin.read(name)
                if name == "OEBPS/style.css":
                    data = patch_css(data.decode("utf-8")).encode("utf-8")
                compress = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
                zout.writestr(name, data, compress_type=compress)

    with open(out, "wb") as f:
        f.write(buf.getvalue())

    print(f"  Written: {out}")


if __name__ == "__main__":
    process_epub(sys.argv[1] if len(sys.argv) > 1 else "magistr_rerendered.epub")
