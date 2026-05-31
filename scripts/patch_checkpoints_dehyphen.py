#!/usr/bin/env python3
"""One-off migration script: apply de-hyphenation to existing checkpoint JSONs.

Usage:
    python patch_checkpoints_dehyphen.py /tmp/archimedes_epub_cache
    python patch_checkpoints_dehyphen.py /tmp/archimedes_epub_cache --dry-run

The script applies the same regex used in _lightonocr_to_html() to the text
nodes inside html_fragments, leaving HTML tags untouched.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

# Same regex as in _lightonocr_to_html()
_DEHYPHEN_RE = re.compile(r'(\w)- *\n([а-яёa-z])')

# Matches HTML tags so we can skip them
_TAG_RE = re.compile(r'(<[^>]+>)')


def _dehyphen_html(fragment: str) -> str:
    """Apply de-hyphenation to text nodes only, leaving tags intact."""
    parts = _TAG_RE.split(fragment)
    result = []
    for part in parts:
        if _TAG_RE.fullmatch(part):
            result.append(part)  # tag — pass through
        else:
            result.append(_DEHYPHEN_RE.sub(r'\1\2', part))
    return "".join(result)


def patch_dir(checkpoint_dir: Path, dry_run: bool) -> None:
    paths = sorted(checkpoint_dir.glob("page_*.json"))
    if not paths:
        print(f"No page_*.json files found in {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    patched = 0
    for path in paths:
        with open(path) as f:
            data = json.load(f)

        new_frags = [_dehyphen_html(frag) for frag in data["html_fragments"]]
        if new_frags == data["html_fragments"]:
            continue

        patched += 1
        if dry_run:
            print(f"[dry-run] would patch {path.name}")
        else:
            data["html_fragments"] = new_frags
            with open(path, "w") as f:
                json.dump(data, f)

    action = "Would patch" if dry_run else "Patched"
    print(f"{action} {patched}/{len(paths)} pages in {checkpoint_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        print(f"Error: not a directory: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    patch_dir(checkpoint_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
