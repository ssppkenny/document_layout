"""
reflow_words.py — word-level reflow engine.

Instead of splitting every word into individual letter boxes (via find_rects),
this module treats whole word image crops as the atomic reflow unit.

Only the word that straddles a line boundary is split — and only into two
image crops (left part / right part), not into individual letters.
find_rects is called at most once per output line (for the straddling word).

Usage:
    from reflow_words import create_page_word_reflow, words_to_wordlines

Comparison with the original reflow.py:
    - reflow.py:  every word → letters via find_rects → reflow letters
    - reflow_words.py: words are the primitive; find_rects only for split words
"""

import cv2
import numpy as np
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language helpers: Tesseract OCR + pyphen hyphenation
# ---------------------------------------------------------------------------

# Map ISO 639-1 two-letter codes → Tesseract three-letter lang codes
_TESS_LANG_MAP = {
    'ru': 'rus',
    'en': 'eng',
    'sv': 'swe',
    'de': 'deu',
    'fr': 'fra',
    'es': 'spa',
    'nl': 'nld',
    'pl': 'pol',
    'cs': 'ces',
    'sk': 'slk',
    'hu': 'hun',
    'fi': 'fin',
    'da': 'dan',
    'nb': 'nor',
    'no': 'nor',
    'pt': 'por',
    'it': 'ita',
    'ro': 'ron',
    'uk': 'ukr',
    'bg': 'bul',
    'hr': 'hrv',
    'lt': 'lit',
    'lv': 'lav',
    'et': 'est',
    'sl': 'slv',
    'af': 'afr',
}


def _ocr_word_char_boxes(word_img: np.ndarray, lang: str) -> Tuple[str, List[int]]:
    """
    OCR a single word image crop with Tesseract and return per-character boxes.

    Returns (text, xs) where `text` is the cleaned, lowercased letter string
    and `xs` is a list of x-right positions (in crop coordinates, i.e. relative
    to the crop's left edge) aligned 1:1 with the characters of `text`.

    Runs `tesseract ... makebox` directly (PSM 7, single text line) which emits
    one character box per line; pyphen break positions can then be mapped to
    exact pixel cuts.  Returns ('', []) on failure.
    """
    import os
    import subprocess
    import tempfile

    tess_lang = _TESS_LANG_MAP.get(lang, lang)

    # Add a small white border so Tesseract doesn't clip edge glyphs.
    pad = 8
    padded = cv2.copyMakeBorder(word_img, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # The pixi env's LD_LIBRARY_PATH points at a libcurl.so.4 without version
    # info; tesseract then emits linker warnings that break pytesseract's
    # version parsing (SystemExit).  Run tesseract directly with a clean env.
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH', None)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, padded)
        proc = subprocess.run(
            ['tesseract', tmp_path, 'stdout', '-l', tess_lang,
             '--psm', '8', '--oem', '1', 'makebox'],
            capture_output=True, text=True, env=env, timeout=30,
        )
    except Exception as e:
        logger.debug(f"Tesseract OCR failed: {e}")
        return '', []
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if proc.returncode != 0:
        logger.debug(f"Tesseract OCR failed (rc={proc.returncode}): "
                     f"{proc.stderr.strip()}")
        return '', []

    chars = []
    xs = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        ch = parts[0]
        if not ch or not ch.strip():
            continue
        ch = ch.strip().lower()
        if not ch.isalpha():
            continue
        try:
            left = int(parts[1])
            right = int(parts[3])
        except ValueError:
            continue
        # Boxes are in padded-image coordinates; subtract pad → crop coords.
        chars.append(ch)
        xs.append(right - pad)
    return ''.join(chars), xs


def _pyphen_break_positions(word_text: str, lang: str) -> List[int]:
    """
    Return a sorted list of valid hyphenation character positions (0-indexed,
    counting from the start of `word_text`) using pyphen.

    E.g. for 'hyphenation' might return [2, 5, 7] meaning splits after
    characters 2, 5, 7 → 'hy-', 'phen-', 'at-', 'ion'.

    Returns [] if pyphen is unavailable or no splits found.
    """
    try:
        import pyphen
    except ImportError:
        logger.debug("pyphen not available — skipping grammatical hyphenation")
        return []

    if not word_text:
        return []

    # Try exact lang code first, then base language
    dic = None
    for code in (lang, lang.split('_')[0]):
        if code in pyphen.LANGUAGES:
            dic = pyphen.Pyphen(lang=code)
            break
    if dic is None:
        logger.debug(f"pyphen: no dictionary for lang={lang!r}")
        return []

    positions = []
    for left, _right in dic.iterate(word_text):
        pos = len(left)
        if 0 < pos < len(word_text):
            positions.append(pos)
    return sorted(set(positions))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Word:
    """A word bounding box in the original image coordinate space."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    bl: int = 0    # descender offset: pixels from ymax UP to the text baseline
                   # 0 = baseline at ymax (no descender); positive = word has descender
    above: int = 0 # above-baseline height in original pixels (baseline_ymax - ref_ymin)
                   # shared across all words on the same line; set by words_to_wordlines

    @property
    def width(self) -> int:
        """Pixel width of the word bounding box (xmax - xmin)."""
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        """Pixel height of the word bounding box (ymax - ymin)."""
        return self.ymax - self.ymin


@dataclass
class _PlacedWord:
    """Internal: a word (or word-half) ready to be rendered on an output line."""
    word: Word          # bounding box in original image (may be a sub-crop)
    space_before: int   # scaled pixels of space to insert before this item
    is_split_half: bool = False  # True if this is a left/right half of a split word
    synth_image: object = None  # Optional np.ndarray: synthesized glyph (e.g. hyphen)
                                # When set, rendered directly instead of cropping original_image.


# ---------------------------------------------------------------------------
# Paragraph detection (word-level, mirrors reflow.py logic)
# ---------------------------------------------------------------------------

def _detect_paragraphs(lines: List[List[Word]]) -> List[int]:
    """
    Return a list of line indices (0-based) that start a new paragraph.
    Detection is based on horizontal indentation of the first word in each line,
    same strategy as detect_paragraphs_and_spacing_from_lines in reflow.py but
    operating on Word objects instead of Letter objects.
    """
    if not lines:
        return [0]

    first_xmins = []
    for line in lines:
        if line:
            first_xmins.append(min(w.xmin for w in line))

    if not first_xmins:
        return [0]

    avg = sum(first_xmins) / len(first_xmins)

    # Method 1: std-dev threshold
    if len(first_xmins) > 1:
        variance = sum((x - avg) ** 2 for x in first_xmins) / len(first_xmins)
        std = math.sqrt(variance)
        threshold = avg + 1.5 * std
    else:
        threshold = float('inf')

    para_starts_1 = {0}
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        xmin = min(w.xmin for w in line)
        if xmin > threshold:
            para_starts_1.add(i)

    # Method 2: jump from previous line
    para_starts_2 = {0}
    prev_xmin = first_xmins[0] if first_xmins else 0
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        xmin = min(w.xmin for w in line)
        if xmin > prev_xmin + 20:
            para_starts_2.add(i)
        prev_xmin = xmin

    # Use whichever method finds more paragraphs
    result = para_starts_1 if len(para_starts_1) >= len(para_starts_2) else para_starts_2
    return sorted(result)


# ---------------------------------------------------------------------------
# Word splitting
# ---------------------------------------------------------------------------

def _synthesize_hyphen(
    ref_word: 'Word',
    zoom_factor: float,
    background_color: tuple,
) -> np.ndarray:
    """
    Return a small BGR image of a hyphen glyph sized to match the rendered word.

    Geometry (all in scaled/output pixels):
      - Image height  = word.height × zoom_factor
      - Stroke height = max(2, round(7% of image height))
      - Stroke width  = max(4, round(35% of image height))  — typical hyphen proportion
      - Horizontal padding on each side = 2px
      - Image width   = stroke_width + 4px padding
      - Stroke placed at vertical centre of the image (x-height region)

    Foreground colour: near-black (30, 30, 30) — works on any light background.
    """
    img_h = max(4, int(ref_word.height * zoom_factor))
    stroke_h = max(2, round(img_h * 0.07))
    stroke_w = max(4, round(img_h * 0.35))
    pad = 2
    img_w = stroke_w + 2 * pad

    img = np.ones((img_h, img_w, 3), dtype=np.uint8)
    img[:] = background_color

    y_mid = img_h // 2
    y0 = max(0, y_mid - stroke_h // 2)
    y1 = min(img_h, y0 + stroke_h)
    x0 = pad
    x1 = pad + stroke_w
    img[y0:y1, x0:x1] = (30, 30, 30)

    return img

def _find_split_x(
    rects_sorted: list,
    word_xmin: int,
    target_x_in_word: int,
    PADDING: int = 2,
) -> Optional[int]:
    """
    Given letter boxes (sorted by xmin, in full-image coords) and a target
    x position (relative to word_xmin), return the absolute x coordinate of
    the best inter-letter cut point that is guaranteed to lie outside any
    letter's ink, or None if no such cut exists.

    find_rects adds PADDING pixels to every component box.  The actual ink
    boundaries are therefore:
        ink_right  = rx2 - PADDING
        ink_left   = rx1 + PADDING

    We scan left-to-right and track the rightmost letter whose ink_right fits
    within target_x_in_word.  The cut is placed at ink_right of that letter
    (i.e. just past the last ink pixel), which is always in whitespace or at
    worst at the very edge of the ink — never inside it.

    If the next letter's ink_left > cut_x (a true gap exists), the cut is
    safely in whitespace.  If the letters touch (ink_left <= cut_x), the cut
    is still at the ink_right of the fitting letter — not inside it.
    """
    cut_x = None
    for rx1, _ry1, rx2, _ry2 in rects_sorted:
        ink_right = rx2 - PADDING          # right edge of actual ink
        rel_ink_right = ink_right - word_xmin
        if rel_ink_right <= target_x_in_word:
            cut_x = ink_right              # safe: just past this letter's ink
    return cut_x


def _split_word(
    word: Word,
    remaining_px: int,          # scaled pixels available on current line
                                # (already excludes space for the hyphen glyph)
    zoom_factor: float,
    original_image: np.ndarray,
    find_rects_fn: Callable,
    use_binarization: bool = False,
    lang: Optional[str] = None,
) -> Tuple[Optional[Word], Word]:
    """
    Split `word` into (left_half, right_half) such that left_half fits within
    `remaining_px` scaled pixels.  The cut is guaranteed to fall outside any
    letter's ink (never through a letter body).

    Strategy:
    1. Call find_rects_fn with the requested binarization mode.
    2. If fewer than 2 components are returned (over-merged letters, e.g.
       touching Cyrillic stems under --bin), retry with use_binarization=False
       which uses tighter merge thresholds and may reveal inter-letter gaps.
    3. If `lang` is provided, OCR the word crop with Tesseract (char-level
       boxes) and use pyphen to find grammatically valid hyphenation points.
       Among all letter-gap cut positions that fit, keep the rightmost one
       whose cut lands on a pyphen break point.  If no grammatical break fits,
       return (None, word) — the whole word moves to the next line.
    4. Without `lang`, use _find_split_x to locate the cut at the ink_right of
       the rightmost fitting letter — guaranteed not to bisect any letter.
    5. If no valid cut is found (word too wide to fit even one letter), return
       (None, word) so the caller moves the whole word to the next line.
    """
    if remaining_px <= 0:
        return None, word

    # Convert remaining scaled pixels back to original-image pixels
    target_x_in_word = int(remaining_px / zoom_factor)  # relative to word.xmin

    word_box = [(word.xmin, word.ymin, word.xmax, word.ymax)]

    def _get_rects(use_bin: bool) -> list:
        """Run find_rects on the word bounding box to extract letter components.
        
        Args:
            use_bin: Whether to enable binarization thresholds.
        Returns:
            List of letter bounding box tuples, or empty list on failure.
        """
        try:
            return find_rects_fn(original_image, word_box, use_binarization=use_bin)
        except Exception as e:
            logger.warning(f"find_rects failed during word split: {e}")
            return []

    rects = _get_rects(use_binarization)

    # Fallback: if binarization mode produced < 2 components (over-merged),
    # retry without binarization — tighter merge thresholds may expose gaps.
    if len(rects) < 2 and use_binarization:
        rects_alt = _get_rects(False)
        if len(rects_alt) >= 2:
            logger.debug(
                f"_split_word: binarization gave {len(rects)} component(s), "
                f"retried without → {len(rects_alt)} components"
            )
            rects = rects_alt

    if len(rects) < 2:
        # Only one (or zero) components — cannot find an inter-letter gap.
        # Do not split: move whole word to next line.
        return None, word

    rects_sorted = sorted(rects, key=lambda r: r[0])

    # --- Pyphen-guided cut selection (when --lang is provided) ---
    # Grammar-strict: only split at a letter gap that aligns with a pyphen
    # break point.  If no grammatical break fits, the whole word moves to the
    # next line (no non-dictionary hyphenation).
    cut_x = None
    if lang:
        word_img = original_image[word.ymin:word.ymax, word.xmin:word.xmax]
        word_text, char_xs = _ocr_word_char_boxes(word_img, lang)
        break_positions = _pyphen_break_positions(word_text, lang) if word_text else []

        if break_positions and char_xs and len(rects_sorted) >= 2:
            # Tolerance for snapping a letter gap to a char box edge.
            widths = [rx2 - rx1 for rx1, _ry1, rx2, _ry2 in rects_sorted]
            median_w = float(np.median(widths)) if widths else 8.0
            tol = max(3.0, median_w * 0.3)

            # Among letter gaps that fit the budget, keep the rightmost one
            # whose nearest char box edge lands on a pyphen break point.
            for rx1, _ry1, rx2, _ry2 in rects_sorted:
                ink_right = rx2 - 2  # 2 = find_rects PADDING
                rel_ink_right = ink_right - word.xmin
                if rel_ink_right > target_x_in_word:
                    continue
                # Nearest OCR char box edge to this gap (crop coords)
                k = min(range(len(char_xs)),
                        key=lambda i: abs(char_xs[i] - rel_ink_right))
                if abs(char_xs[k] - rel_ink_right) <= tol and (k + 1) in break_positions:
                    cut_x = ink_right  # rightmost fitting grammatical gap
            if cut_x is not None:
                logger.debug(
                    f"_split_word: pyphen-guided cut at x={cut_x} "
                    f"(word={word_text!r}, breaks={break_positions})"
                )
    else:
        # No language requested — rightmost-fitting-gap strategy
        cut_x = _find_split_x(rects_sorted, word.xmin, target_x_in_word)

    if cut_x is None or cut_x <= word.xmin:
        # No valid cut — move whole word to next line
        return None, word

    # Clamp to word bounds
    cut_x = min(cut_x, word.xmax - 1)

    left  = Word(word.xmin, word.ymin, cut_x,      word.ymax, bl=word.bl, above=word.above)
    # Reject splits where the left half is too narrow (noise pixel at boundary)
    if (left.xmax - left.xmin) < max(10, int(word.width * 0.15)):
        return None, word
    right = Word(cut_x,     word.ymin, word.xmax,  word.ymax, bl=word.bl, above=word.above)
    return left, right


# ---------------------------------------------------------------------------
# Hyphen detection
# ---------------------------------------------------------------------------

def _find_trailing_hyphen_component(word: Word, image: np.ndarray):
    """
    Locate a trailing hyphen glyph in the word crop.

    Returns (cx, cw, ch) — the bounding box of the rightmost qualifying
    horizontal stroke in the rightmost 25% / middle-third zone — or None.

    Uses connected components of the FULL word crop (not just the zone) so
    that letter bodies stay whole: the horizontal strokes of letters like
    'e', 'c', 'a' are part of the tall letter component and are rejected by
    the height filter, while a true hyphen is its own short isolated stroke.
    """
    h = word.ymax - word.ymin
    w = word.xmax - word.xmin
    if h <= 0 or w <= 0:
        return None

    crop = image[word.ymin:word.ymax, word.xmin:word.xmax]
    if crop.size == 0:
        return None

    # Grayscale + binarize
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Zone: rightmost 25% of width, middle third of height (33%–67%)
    # Using middle third (not middle half) to exclude the dot of '!' which sits
    # in the bottom ~15% of the word box, while a true hyphen sits at x-height centre.
    x0 = max(0, w - w // 4)
    y0 = h // 3
    y1 = h - h // 3
    if y1 <= y0 or x0 >= w:
        return None

    # Connected components of the FULL crop (letters stay whole)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    if num_labels < 2:
        return None

    best = None
    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        if cw < 1 or ch < 1:
            continue
        # Must overlap the rightmost 25% zone
        if cx + cw <= x0:
            continue
        # Vertical centre must sit in the middle third (excludes ascender
        # diacritics and descender tails)
        if cy + ch / 2 < y0 or cy + ch / 2 > y1:
            continue
        # Must be at least 3px tall (eliminates 1–2px specks and serif artifacts)
        if ch < 3:
            continue
        # Must be wide enough to be a real stroke (not a speck)
        if cw < max(2, int(h * 0.12)):
            continue
        # Must be short (not a letter body)
        if ch > h * 0.30:
            continue
        # Must be horizontal (wider than tall)
        if cw / ch < 2.0:
            continue
        # Pick the rightmost qualifying component — that is the hyphen glyph
        if best is None or cx > best[0]:
            best = (cx, cw, ch)

    return best


def _ends_with_hyphen(word: Word, image: np.ndarray) -> bool:
    """
    Return True if the word image crop ends with a visible hyphen/dash.

    Algorithm (purely visual, no OCR text required):
    1. Crop the word from the image and binarize (Otsu).
    2. Find connected components of the FULL crop so letter bodies stay whole.
    3. A hyphen satisfies:
         - overlaps the rightmost 25% of the crop width
         - vertical centre in the middle third (33%–67% of word height)
         - height ≤ 0.30 × word_height  (not a tall letter body)
         - width  ≥ 0.12 × word_height  (not a speck)
         - aspect ratio w/h ≥ 2.0       (wider than tall — horizontal stroke)
    4. Return True if any such component is found.

    Robust against letters (e, c, a, з, г, с, …) whose horizontal strokes are
    part of the tall letter component and are rejected by the height filter.
    """
    return _find_trailing_hyphen_component(word, image) is not None


def _strip_trailing_hyphen(word: Word, image: np.ndarray) -> Word:
    """
    Return a new Word whose xmax is trimmed to exclude the trailing hyphen glyph.

    Uses the same component analysis as _ends_with_hyphen.  The new xmax is set
    to the left edge of the hyphen component (in original-image coordinates)
    minus a 2-pixel gap, so the hyphen ink is fully excluded from the crop.

    If no qualifying hyphen component is found (should not happen when called
    only after _ends_with_hyphen returned True), the original word is returned
    unchanged.
    """
    found = _find_trailing_hyphen_component(word, image)
    if found is None:
        return word

    # cx is relative to the crop, which starts at word.xmin in the image.
    hyphen_left_in_crop = found[0]

    # Trim to the rightmost ink column before the hyphen so the continuation
    # word joins tightly (no trailing blank from the word's own padding).
    crop = image[word.ymin:word.ymax, word.xmin:word.xmin + hyphen_left_in_crop]
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_cols = np.where(bw.any(axis=0))[0]
        if len(ink_cols):
            new_xmax = word.xmin + int(ink_cols[-1]) + 1
            new_xmax = max(word.xmin + 1, new_xmax)
            return Word(word.xmin, word.ymin, new_xmax, word.ymax, bl=word.bl, above=word.above)

    new_xmax = word.xmin + hyphen_left_in_crop - 2  # 2px gap before hyphen ink
    new_xmax = max(word.xmin + 1, new_xmax)         # always keep at least 1px
    return Word(word.xmin, word.ymin, new_xmax, word.ymax, bl=word.bl, above=word.above)


def _trim_leading_blank(word: Word, image: np.ndarray) -> Word:
    """
    Return a new Word whose xmin is advanced past the leading blank columns.

    Used for the continuation word of a hyphenated pair so it joins the
    stripped word tightly (no leading blank from the word's own padding).
    If no ink is found, the original word is returned unchanged.
    """
    crop = image[word.ymin:word.ymax, word.xmin:word.xmax]
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_cols = np.where(bw.any(axis=0))[0]
        if len(ink_cols):
            new_xmin = word.xmin + int(ink_cols[0])
            new_xmin = min(new_xmin, word.xmax - 1)
            return Word(new_xmin, word.ymin, word.xmax, word.ymax, bl=word.bl, above=word.above)
    return word


def _is_dash_word(word: Word) -> bool:
    """True if the word is a thin wide horizontal stroke (em-dash).

    Standalone em-dashes (direct-speech markers) are detected as words by
    _detect_standalone_dashes; they are much wider than tall.
    """
    return word.height > 0 and word.width >= 3 * word.height and word.height <= 12


# ---------------------------------------------------------------------------
# Main reflow function
# ---------------------------------------------------------------------------

def _remove_padding_fragments(crop: np.ndarray, padding: int) -> np.ndarray:
    """Erase foreign ink from the outer `padding`-px frame of a word crop.

    Word boxes are padded in main.py to prevent glyph clipping.  In tightly
    set text that padding captures fragments of the neighbouring lines or
    letters (descender tails from the line above, ascender bars from the
    line below, edges of adjacent letters).  Such fragments are detached
    from the word's own ink; if left in place they are pasted into the gaps
    of the reflowed output.

    Connected components that touch the core region (the crop minus the
    frame) are genuine glyph pixels and are preserved — including glyph
    parts that legitimately protrude into the frame.
    """
    if padding <= 0:
        return crop
    h, w = crop.shape[:2]
    if h <= 2 * padding or w <= 2 * padding:
        return crop

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if not (bw > 0).any():
        return crop

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8, cv2.CV_32S)
    if num <= 1:
        return crop

    # Foreign fragments can poke 1-2 antialiased rows/cols into the core, so
    # require contact with an inner core inset from the frame by `margin`.
    margin = max(2, padding // 2)

    # Local background colour estimated from the crop border.
    if crop.ndim == 3:
        border = np.concatenate([
            crop[0, :].reshape(-1, 3),
            crop[-1, :].reshape(-1, 3),
            crop[:, 0].reshape(-1, 3),
            crop[:, -1].reshape(-1, 3),
        ])
    else:
        border = np.concatenate([
            crop[0, :],
            crop[-1, :],
            crop[:, 0],
            crop[:, -1],
        ])
    fill = np.median(border, axis=0).astype(crop.dtype)

    out = crop.copy()
    for i in range(1, num):
        x, y, cw, ch, _ = stats[i]
        if x + cw > padding + margin and x < w - padding - margin and \
           y + ch > padding + margin and y < h - padding - margin:
            continue
        out[labels == i] = fill
    return out


def create_page_word_reflow(
    lines: List[List[Word]],
    original_image: np.ndarray,
    zoom_factor: float,
    new_page_width: int,
    find_rects_fn: Callable,
    left_margin: int = 50,
    right_margin: int = 50,
    top_margin: int = 50,
    bottom_margin: int = 50,
    preserve_line_breaks: bool = False,
    background_color: tuple = (220, 220, 220),
    use_binarization: bool = False,
    is_title: bool = False,
    lang: Optional[str] = None,
    word_padding: int = 5,
) -> np.ndarray:
    """
    Reflow a list of word lines onto a new page of width `new_page_width`.

    Words are placed as whole image crops.  Only the word that straddles a
    line boundary is split (into two crops) using find_rects_fn to locate the
    nearest inter-letter gap.

    Args:
        lines:            List of lines; each line is a list of Word objects
                          (coordinates in `original_image` space).
        original_image:   Source image — word crops are taken from here.
        zoom_factor:      Scale factor applied to all crops.
        new_page_width:   Width of the output image in pixels.
        find_rects_fn:    The find_rects function from main.py, injected to
                          avoid circular imports.  Signature:
                            find_rects(img, line_words, use_binarization=False)
                          where line_words is a list of (xmin,ymin,xmax,ymax).
        left_margin, right_margin, top_margin, bottom_margin: margins in px.
        preserve_line_breaks: If True, honour original line boundaries exactly
                          (no word-wrap, no splitting).
        background_color: BGR tuple for the page background.
        use_binarization: Passed through to find_rects_fn when splitting.
        is_title:         If True, suppress paragraph detection / indentation.
        word_padding:     Padding (in original pixels) applied to word boxes in
                          main.py.  Used to strip foreign ink (fragments of
                          neighbouring lines/letters) captured by the padding.

    Returns:
        Output page as a numpy BGR image.
    """
    # ------------------------------------------------------------------
    # Guard: empty input
    # ------------------------------------------------------------------
    empty_h = top_margin + bottom_margin + 100
    if not lines or all(not line for line in lines):
        page = np.ones((empty_h, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    available_width = new_page_width - left_margin - right_margin

    # ------------------------------------------------------------------
    # Paragraph detection
    # ------------------------------------------------------------------
    if is_title:
        para_line_starts = {0}
    else:
        para_line_starts = set(_detect_paragraphs(lines))

    # ------------------------------------------------------------------
    # Compute average word space (used for cross-line word gaps)
    # ------------------------------------------------------------------
    all_words_flat = [w for line in lines for w in line]
    if all_words_flat:
        avg_word_w = sum(w.width for w in all_words_flat) / len(all_words_flat)
        # Inter-word space derived from median word height × 30%.
        # Using word width (as before) produces values ~3× too large because
        # avg_word_width >> actual inter-word gap in the original scan.
        # Word height is a stable proxy for cap-height and gives a spacing
        # that matches typical typographic inter-word space (~25–35% of cap-height).
        all_heights = sorted(w.height for w in all_words_flat)
        median_word_h = all_heights[len(all_heights) // 2]
        avg_word_space = int(median_word_h * zoom_factor * 0.30)
    else:
        avg_word_space = 20

    # ------------------------------------------------------------------
    # Word-wrap loop — produce output_lines
    # Each output line: list of _PlacedWord
    # ------------------------------------------------------------------
    output_lines: List[dict] = []   # {'words': [_PlacedWord], 'para_start': bool}

    current_words: List[_PlacedWord] = []
    current_width = 0
    current_para_start = True

    def _flush(para_start: bool):
        """Flush the accumulated words as a completed output line.
        
        Appends the current line to output_lines and resets the accumulator.
        Args:
            para_start: Whether this line begins a new paragraph.
        """
        nonlocal current_words, current_width, current_para_start
        if current_words:
            output_lines.append({
                'words': current_words,
                'para_start': para_start,
            })
        current_words = []
        current_width = 0
        current_para_start = False

    # Track whether the previous original line ended with a hyphen.
    # When True, the first word of the next original line is a hyphenated
    # continuation and should be joined with zero inter-word space.
    prev_line_ends_with_hyphen = False
    # The stripped word that ended the previous line (reference geometry for
    # aligning the continuation word's baseline/cap-height).
    prev_hyphen_word = None

    for line_idx, line in enumerate(lines):
        if not line:
            continue

        is_para_start_line = line_idx in para_line_starts
        sorted_words = sorted(line, key=lambda w: w.xmin)

        # Direct-speech lines (beginning with an em-dash) must always start a
        # new output line — flush any accumulated words so the dash begins a line.
        if sorted_words and _is_dash_word(sorted_words[0]):
            _flush(current_para_start)

        if preserve_line_breaks:
            # Hard line break at every original line boundary
            _flush(is_para_start_line)
            placed = [
                _PlacedWord(
                    word=w,
                    space_before=0 if j == 0 else _inter_word_gap(sorted_words[j-1], w, zoom_factor, avg_word_space),
                )
                for j, w in enumerate(sorted_words)
            ]
            output_lines.append({'words': placed, 'para_start': is_para_start_line})
            prev_line_ends_with_hyphen = False
            continue

        # Detect hyphen at end of this original line (for the *next* iteration).
        # If found, also strip the hyphen glyph from the word crop so it is not
        # rendered in the output (the continuation word follows with zero space).
        this_line_ends_with_hyphen = _ends_with_hyphen(sorted_words[-1], original_image)
        if this_line_ends_with_hyphen:
            sorted_words[-1] = _strip_trailing_hyphen(sorted_words[-1], original_image)
            prev_hyphen_word = sorted_words[-1]

        for word_idx, word in enumerate(sorted_words):
            scaled_w = int(word.width * zoom_factor)

            # Space before this word
            if not current_words and not output_lines:
                # Very first word on the page
                space = 0
            elif not current_words:
                # First word on a new output line
                space = 0
            elif word_idx == 0:
                if prev_line_ends_with_hyphen:
                    # Continuation of a hyphenated word — no inter-word gap,
                    # trim the leading blank so it joins the stripped word, and
                    # align its baseline/cap-height to the joined word so the
                    # glyphs sit at the same height (the continuation comes from
                    # a different original line with its own baseline stats).
                    space = 0
                    word = _trim_leading_blank(word, original_image)
                    scaled_w = int(word.width * zoom_factor)
                    if prev_hyphen_word is not None:
                        joined_above = prev_hyphen_word.height - prev_hyphen_word.bl
                        new_bl = max(0, word.height - joined_above)
                        word = Word(word.xmin, word.ymin, word.xmax, word.ymax,
                                    bl=new_bl, above=joined_above)
                else:
                    # First word of a new original line — use standard word space
                    space = avg_word_space
            else:
                # Within the same original line — use actual gap
                prev_word = sorted_words[word_idx - 1]
                space = _inter_word_gap(prev_word, word, zoom_factor, avg_word_space)

            # Paragraph indent on first word of a paragraph line
            indent = 0
            if is_para_start_line and word_idx == 0 and not is_title:
                indent = int(avg_word_w * zoom_factor * 0.5)  # ~half word width

            effective_available = available_width - indent

            would_overflow = (
                current_words
                and current_width + space + scaled_w > effective_available
            )

            if would_overflow:
                # Reserve space for the hyphen glyph so it fits inside the
                # right margin instead of being clipped by the renderer.
                hyphen_img = _synthesize_hyphen(word, zoom_factor, background_color)
                hyphen_w = hyphen_img.shape[1]
                remaining = effective_available - current_width - space - hyphen_w
                left_half, right_half = _split_word(
                    word, remaining, zoom_factor,
                    original_image, find_rects_fn, use_binarization,
                    lang=lang,
                )

                if left_half is not None:
                    # Place left half at end of current line, followed by a
                    # synthesized hyphen glyph to signal word continuation.
                    left_scaled_w = int(left_half.width * zoom_factor)
                    current_words.append(_PlacedWord(
                        word=left_half,
                        space_before=space,
                        is_split_half=True,
                    ))
                    current_width += space + left_scaled_w
                    # Append hyphen glyph with zero space (immediately after left half)
                    current_words.append(_PlacedWord(
                        word=left_half,   # dummy reference for baseline/height
                        space_before=0,
                        is_split_half=True,
                        synth_image=hyphen_img,
                    ))
                    current_width += hyphen_w
                    _flush(is_para_start_line and word_idx == 0)
                    # Right half starts the next line
                    if right_half is not None:
                        current_words.append(_PlacedWord(
                            word=right_half,
                            space_before=0,
                            is_split_half=True,
                        ))
                        current_width = int(right_half.width * zoom_factor)
                else:
                    # Nothing fit — flush current line, put whole word on next
                    _flush(is_para_start_line and word_idx == 0)
                    current_words.append(_PlacedWord(word=word, space_before=0))
                    current_width = scaled_w
            else:
                # Word fits normally
                effective_space = space + indent if not current_words else space
                current_words.append(_PlacedWord(word=word, space_before=effective_space))
                current_width += effective_space + scaled_w

        # Update hyphen state for the next original line
        prev_line_ends_with_hyphen = this_line_ends_with_hyphen

    # Flush last line
    _flush(False)

    if not output_lines:
        page = np.ones((empty_h, new_page_width, 3), dtype=np.uint8)
        page[:] = background_color
        return page

    # ------------------------------------------------------------------
    # Line height calculation
    # Use 95th percentile of scaled word heights × 1.5 (same as reflow.py)
    # ------------------------------------------------------------------
    all_scaled_heights = [
        int(pw.word.above * zoom_factor)
        for ol in output_lines
        for pw in ol['words']
        if pw.word.above > 0
    ]

    if all_scaled_heights:
        p95 = int(np.percentile(all_scaled_heights, 95))
        line_height = int(p95 * 1.3)
        logger.info("  [reflow_words] Word above p95=%dpx, line_height=%dpx, %d lines", p95, line_height, len(output_lines))
    else:
        line_height = 60
        logger.info("  [reflow_words] Fallback line_height=%dpx", line_height)

    para_spacing = int(line_height * 0.5)

    # ------------------------------------------------------------------
    # Compute total page height
    # ------------------------------------------------------------------
    total_height = top_margin
    prev_para = False
    for ol in output_lines:
        if ol['para_start'] and total_height > top_margin:
            total_height += para_spacing
        total_height += line_height
    total_height += bottom_margin

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    page = np.ones((total_height, new_page_width, 3), dtype=np.uint8)
    page[:] = background_color

    current_y = top_margin

    for ol in output_lines:
        if ol['para_start'] and current_y > top_margin:
            current_y += para_spacing

        # Baseline position: use the shared per-line above-baseline height stored
        # in word.above (set by words_to_wordlines from robust line statistics).
        # All words on the same original line share the same above value, so
        # max_above is stable and baseline_y doesn't shift due to OCR box jitter.
        above_vals = [
            int(pw.word.above * zoom_factor)
            for pw in ol['words']
        ]
        max_above = max(above_vals)
        baseline_y = current_y + max_above

        current_x = left_margin

        for pw in ol['words']:
            current_x += pw.space_before

            w = pw.word

            # --- Synthesized glyph (e.g. hyphen after a split word) ---
            if pw.synth_image is not None:
                si = pw.synth_image
                si_h, si_w = si.shape[:2]
                # Align to baseline using the reference word's geometry
                scaled_bl = int(w.bl * zoom_factor)
                word_above = int(w.height * zoom_factor) - scaled_bl
                y_start = baseline_y - word_above
                # Centre the glyph vertically within the word's above-baseline span
                y_offset = max(0, (word_above - si_h) // 2)
                y_start = y_start + y_offset
                y_end = y_start + si_h
                x_start = current_x
                x_end = current_x + si_w
                y_start_c = max(0, y_start)
                y_end_c = min(total_height, y_end)
                x_start_c = max(0, x_start)
                x_end_c = min(new_page_width - right_margin, x_end)
                if y_end_c > y_start_c and x_end_c > x_start_c:
                    si_y0 = y_start_c - y_start
                    si_y1 = si_y0 + (y_end_c - y_start_c)
                    si_x0 = x_start_c - x_start
                    si_x1 = si_x0 + (x_end_c - x_start_c)
                    page[y_start_c:y_end_c, x_start_c:x_end_c] = si[si_y0:si_y1, si_x0:si_x1]
                current_x += si_w
                continue

            scaled_w = int(w.width * zoom_factor)
            scaled_h = int(w.height * zoom_factor)

            if scaled_w <= 0 or scaled_h <= 0:
                continue

            # Crop from original image
            crop = original_image[w.ymin:w.ymax, w.xmin:w.xmax]
            if crop.size == 0:
                continue

            crop = _remove_padding_fragments(crop, word_padding)

            resized = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            # Place so word baseline aligns with baseline_y.
            # above-baseline part of this word = height - bl (in original pixels).
            # y_start is where the top of the word crop lands on the output page.
            scaled_bl = int(w.bl * zoom_factor)
            word_above = int(w.height * zoom_factor) - scaled_bl
            y_start = baseline_y - word_above
            y_end = y_start + scaled_h
            x_start = current_x
            x_end = current_x + scaled_w

            # Clamp to page bounds
            y_start_c = max(0, y_start)
            y_end_c = min(total_height, y_end)
            x_start_c = max(0, x_start)
            x_end_c = min(new_page_width - right_margin, x_end)

            if y_end_c > y_start_c and x_end_c > x_start_c:
                crop_y0 = y_start_c - y_start
                crop_y1 = crop_y0 + (y_end_c - y_start_c)
                crop_x0 = x_start_c - x_start
                crop_x1 = crop_x0 + (x_end_c - x_start_c)
                page[y_start_c:y_end_c, x_start_c:x_end_c] = resized[crop_y0:crop_y1, crop_x0:crop_x1]

            current_x += scaled_w

        current_y += line_height

    return page


# ---------------------------------------------------------------------------
# Helper: inter-word gap in scaled pixels
# ---------------------------------------------------------------------------

def _inter_word_gap(
    prev_word: Word,
    curr_word: Word,
    zoom_factor: float,
    avg_word_space: int,
) -> int:
    """
    Compute the scaled inter-word gap between two adjacent words on the same
    original line.  Falls back to avg_word_space if the gap is non-positive.
    """
    gap = curr_word.xmin - prev_word.xmax
    if gap > 0:
        return int(gap * zoom_factor)
    if gap >= -10:
        # Small overlap caused by the ±5px word-box padding expansion in main.py.
        # Two adjacent words that originally had 0–5px between them now overlap
        # by up to 10px.  Map this range to a small but non-zero space rather
        # than falling back to the full avg_word_space.
        return max(1, int((gap + 10) * zoom_factor // 2))
    return avg_word_space


# ---------------------------------------------------------------------------
# Utility: convert doctr word tuples → List[List[Word]]
# ---------------------------------------------------------------------------

def _robust_linear_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    epsilon: float,
) -> Optional[Tuple[float, float]]:
    """
    Fit y = a*x + b to (xs, ys) with one round of outlier rejection.

    Points whose residual from the initial fit exceeds `epsilon` are excluded,
    then the fit is redone on the inliers.  This mirrors the support/repel
    scoring in the interline space model (Kim & Oh): points within ε of the
    line support it; points further away are treated as outliers (descenders
    for ymax fits, diacritics/ascenders for ymin fits).

    Returns (a, b) of the final fit, or None if fewer than 2 inliers remain.
    """
    if len(xs) < 2:
        return None
    # Initial fit
    a, b = np.polyfit(xs, ys, 1)
    residuals = np.abs(ys - (a * xs + b))
    inlier_mask = residuals <= epsilon
    if inlier_mask.sum() < 2:
        return None
    # Refit on inliers
    a, b = np.polyfit(xs[inlier_mask], ys[inlier_mask], 1)
    return float(a), float(b)


def words_to_wordlines(
    lines: List[List[Tuple[int, int, int, int]]],
) -> List[List[Word]]:
    """
    Convert the line structure produced by main.py's line-grouping code
    (list of lists of (xmin, ymin, xmax, ymax) tuples) into List[List[Word]].

    For lines with ≥ 4 words, attempts a skew-aware per-word baseline by
    fitting linear models to the floor points (ymax → baseline) and ceil
    points (ymin → cap-height line) as a function of x_center.  This follows
    the interline space model of Kim & Oh: floor points define the baseline
    line, ceil points define the cap-height line, both parameterised by slope
    (skew angle θ) and intercept.

    Outlier rejection uses tolerance ε = 0.3 × median_word_height, matching
    the paper's support region width (roughly one descender depth).

    The skew fit is accepted only when the predicted baseline range across the
    line exceeds 0.15 × median_word_height — below that the slope is noise, not
    real skew, and the scalar fallback is used instead.

    Scalar fallback (lines < 4 words, or fit rejected):
      baseline_ymax = 10th percentile of ymaxes (or min for short lines)
      ref_ymin      = 10th percentile of ymins  (or min for short lines)
      above         = baseline_ymax − ref_ymin  (shared scalar per line)

    Each word's bl field = max(0, min(word.ymax − fitted_baseline(x_center),
                                      height // 2))
    Each word's above field = max(1, fitted_baseline(x_center)
                                     − fitted_capheight(x_center))
    """
    result = []
    for line_idx, line in enumerate(lines):
        if not line:
            result.append([])
            continue

        raw_ymaxes = np.array([ymax for (_, _, _, ymax) in line], dtype=float)
        raw_ymins  = np.array([ymin for (_, ymin, _, _) in line], dtype=float)
        raw_xctrs  = np.array([(xmin + xmax) / 2.0 for (xmin, _, xmax, _) in line], dtype=float)
        raw_heights = raw_ymaxes - raw_ymins

        median_h = float(np.median(raw_heights))

        # Exclude standalone em-dashes (thin wide strokes) from the baseline /
        # cap-height statistics — their small y-extents would skew the 10th
        # percentile baseline downward and misplace the whole line.
        dash_mask = np.array([
            (ymax - ymin) <= 0.35 * median_h and (xmax - xmin) >= 3 * (ymax - ymin)
            for (xmin, ymin, xmax, ymax) in line
        ])
        text_idx = ~dash_mask
        if not text_idx.any():
            text_idx = np.ones(len(line), dtype=bool)
        text_pos = np.cumsum(text_idx) - 1  # line index -> filtered index

        ymaxes = raw_ymaxes[text_idx]
        ymins  = raw_ymins[text_idx]
        xctrs  = raw_xctrs[text_idx]
        heights = ymaxes - ymins

        median_h = float(np.median(heights))
        epsilon = 0.3 * median_h

        use_fit = False
        bl_fit = cap_fit = None

        if len(line) >= 4:
            bl_fit  = _robust_linear_fit(xctrs, ymaxes, epsilon)
            cap_fit = _robust_linear_fit(xctrs, ymins,  epsilon)

            if bl_fit is not None and cap_fit is not None:
                x_range = float(xctrs.max() - xctrs.min())
                predicted_range = abs(bl_fit[0]) * x_range
                if predicted_range > 0.15 * median_h:
                    use_fit = True
                    logger.debug(
                        f"Line {line_idx}: skew fit accepted, slope={bl_fit[0]:.4f}, "
                        f"predicted_range={predicted_range:.1f}px, median_h={median_h:.1f}px"
                    )
                else:
                    logger.debug(
                        f"Line {line_idx}: skew fit rejected (range {predicted_range:.1f} "
                        f"< 0.15×{median_h:.1f}), using scalar fallback"
                    )

        if len(ymaxes) >= 4:
            baseline_ymax = int(np.percentile(ymaxes, 10))
        else:
            baseline_ymax = int(ymaxes.min())

        if not use_fit:
            # Scalar fallback — same logic as before
            if len(ymins) >= 4:
                ref_ymin = int(np.percentile(ymins, 10))
            else:
                ref_ymin = int(ymins.min())
            line_above = max(1, baseline_ymax - ref_ymin)

        words = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(line):
            height = ymax - ymin
            width = xmax - xmin
            if height <= 0.35 * median_h and width >= 3 * height:
                # Standalone em-dash: a thin wide stroke. Place it at its actual
                # height above the baseline, not at the line's cap height.
                # bl = height - above (negative) so the render loop's
                # (height - bl) equals the true above-baseline span.
                if use_fit:
                    xc = raw_xctrs[i]
                    fitted_bl = bl_fit[0] * xc + bl_fit[1]
                    word_above = max(1, int(round(fitted_bl - ymin)))
                else:
                    word_above = max(1, baseline_ymax - ymin)
                word_bl = height - word_above
            elif use_fit:
                j = int(text_pos[i])
                xc = xctrs[j]
                fitted_bl  = bl_fit[0]  * xc + bl_fit[1]
                fitted_cap = cap_fit[0] * xc + cap_fit[1]
                word_above = max(1, int(round(fitted_bl - fitted_cap)))
                word_bl    = max(0, min(int(round(ymax - fitted_bl)), height // 2))
            else:
                word_above = line_above
                word_bl    = max(0, min(ymax - baseline_ymax, height // 2))
            words.append(Word(xmin, ymin, xmax, ymax, bl=word_bl, above=word_above))
        result.append(words)
    return result
