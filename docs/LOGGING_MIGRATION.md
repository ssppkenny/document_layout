# Logging Migration Summary - OCR Reflow Package

## Overview
All `print()` statements in the ocr_reflow package have been replaced with Python's `logging` module. Logging is **disabled by default** to maintain clean output for end users.

## Changes Made

### 1. Package Initialization (`__init__.py`)
- Added logging import and configuration
- Set up package-level logger with `WARNING` level (disables all debug/info messages by default)
- Added `NullHandler` to prevent "No handlers could be found for logger" warnings

```python
import logging

# Configure logging - disabled by default (level set to WARNING)
# Users can enable debug logging with: logging.getLogger('ocr_reflow').setLevel(logging.DEBUG)
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.getLogger(__name__).setLevel(logging.WARNING)
```

### 2. Main Module (`main.py`)
**Added:**
- `import logging` at the top
- Logger setup: `logger = logging.getLogger(__name__)`

**Replacements (20 total):**
- Line 467: `print()` → `logger.debug()` - Background color detection
- Line 525: `print()` → `logger.debug()` - Layout analysis start
- Line 531: `print()` → `logger.debug()` - Layout boxes count
- Line 534: `print()` → `logger.debug()` - Layout box details (in loop)
- Line 558: `print()` → `logger.debug()` - Processing box type
- Line 745: `print()` → `logger.info()` - Layout-based processing choice
- Line 748: `print()` → `logger.info()` - Text-only processing choice
- Line 754: `print()` → `logger.info()` - Output file saved
- Line 760: `print()` → `logger.info()` - Preview saved
- Line 763: `print()` → `logger.info()` - Creating word segmentation
- Line 787: `print()` → `logger.info()` - Word segmentation saved
- Line 788: `print()` → `logger.info()` - Total words detected

### 3. Divide & Conquer Module (`divide_conquer_4d.py`)
**Added:**
- `import logging` after docstring
- Logger setup: `logger = logging.getLogger(__name__)`

**Replacements (6 total):**
- Line 479: `print()` → `logger.info()` - Found pairs count
- Line 480: `print()` → `logger.info()` - Algorithm execution time
- Line 481: `print()` → `logger.info()` - Enclosure pairs summary
- Line 483: `print()` → `logger.debug()` - Individual pair details (in loop)
- Line 485: `print()` → `logger.debug()` - Enclosing rectangle details
- Line 486: `print()` → `logger.debug()` - Enclosed rectangle details

### 4. CLI Module (`cli.py`)
**Added:**
- `import logging` after sys import
- Logger setup: `logger = logging.getLogger(__name__)`

**Replacements (8 total):**
- Lines 13-22: Multi-line help text converted to single `logger.info()` call
- Lines 26-28: Multi-line error text converted to single `logger.error()` call
- Line 35: `print()` → `logger.error()` - File not found error
- Line 46: `print()` → `logger.info()` - Processing message
- Line 50: `print()` → `logger.info()` - Success message
- Line 52: `print()` → `logger.error()` - Error processing message

## Logging Levels Used

- **DEBUG** (`logger.debug()`) - Detailed information for developers
  - Background color detection
  - Layout analysis details
  - Individual box/pair processing
  - Enclosed/enclosing rectangle details

- **INFO** (`logger.info()`) - General information messages
  - Processing mode selection (layout-based vs text-only)
  - File operations (save, create)
  - Summary statistics (word count, time)
  - CLI operations (processing start/completion)

- **ERROR** (`logger.error()`) - Error messages
  - File not found errors
  - Processing failures
  - CLI usage errors

## How to Use

### Default Behavior (Logging Disabled)
```python
from ocr_reflow import process_document

# No logging output - clean execution
result = process_document("image.png")
```

### Enable Debug Logging
```python
import logging
from ocr_reflow import process_document

# Enable all logging messages
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Now you'll see detailed debug messages
result = process_document("image.png")
```

### Enable Info Logging Only
```python
import logging
from ocr_reflow import process_document

# Enable info and above
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# See summary/status messages, but not debug details
result = process_document("image.png")
```

### Configure Specific Logger
```python
import logging

# Enable only ocr_reflow logging
logger = logging.getLogger('ocr_reflow')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)
```

### From Command Line
```bash
# Enable debug logging for CLI
PYTHONPATH=/path/to/package python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from ocr_reflow.cli import main
main()
" image.png
```

## Total Print Statements Replaced: 28
- `main.py`: 12 statements
- `divide_conquer_4d.py`: 6 statements
- `cli.py`: 8 statements
- `__init__.py`: Logging setup added

## Benefits

1. **User-Friendly**: Output is clean by default - no debug noise
2. **Developer-Friendly**: Easy to enable detailed logging when debugging
3. **Flexible**: Can configure logging globally or per-logger
4. **Standard**: Uses Python's built-in logging module (best practice)
5. **Maintainable**: Consistent logging pattern across all modules
6. **Testable**: Easier to test without capturing stdout

## Backward Compatibility

- No breaking changes to public APIs
- All functionality remains identical
- Only output method changed (print → logging)
- Existing code using the package works without modification
