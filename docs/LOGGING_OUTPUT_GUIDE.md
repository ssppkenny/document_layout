# Logging Output Guide

## Problem Fixed ✅

Your logging statements were not visible because Python's logging system needs to be configured with a handler before any log messages will appear.

## Solution Applied

Added `logging.basicConfig()` to the `if __name__ == "__main__":` block in `main.py`.

## How to Use

### 1. Normal Output (INFO level - shows important messages):
```bash
pixi run python src/ocr_reflow/main.py images/dvurog_p087.png --layout
```

**You'll now see:**
- INFO: Using layout-based processing...
- INFO: Output saved to: output_reflowed.png
- INFO: Preview saved to: output_reflowed_preview.png
- INFO: Creating word segmentation visualization...
- INFO: Word segmentation visualization saved to: ...

### 2. Detailed Output (DEBUG level - shows everything):

To see the device detection and model loading details, edit line 794 in `main.py`:

Change:
```python
level=logging.INFO,  # Change to DEBUG for more detailed output
```

To:
```python
level=logging.DEBUG,  # Change to DEBUG for more detailed output
```

**Then run:**
```bash
pixi run python src/ocr_reflow/main.py images/dvurog_p087.png --layout
```

**You'll see additional DEBUG messages like:**
- DEBUG: device for YOLOv10 determined mps
- DEBUG: Using device for YOLOv10: mps
- DEBUG: Detected background color (BGR): [...]
- DEBUG: Running layout analysis...
- DEBUG: Detected 5 layout boxes:
- DEBUG: Processing plain text box at y=123
- And much more!

### 3. Quick Debug Toggle (without editing files):

Create a file `run_debug.sh`:
```bash
#!/bin/bash
export PYTHONUNBUFFERED=1
pixi run python -c "
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
import sys
sys.argv = ['main.py'] + sys.argv[1:]
exec(open('src/ocr_reflow/main.py').read())
" "$@"
```

Make it executable:
```bash
chmod +x run_debug.sh
```

Run with:
```bash
./run_debug.sh images/dvurog_p087.png --layout
```

## Logging Levels Explained

- **DEBUG**: Most verbose - shows device detection, model loading, every processing step
- **INFO**: Normal output - shows major steps and results
- **WARNING**: Only issues that might need attention
- **ERROR**: Only errors

## Current Configuration

✅ Logging is now configured in `main.py` 
✅ Default level: `INFO` (shows important messages)
✅ Easy to change to `DEBUG` for troubleshooting

## What You'll See Now

When you run:
```bash
pixi run python src/ocr_reflow/main.py images/dvurog_p087.png --layout
```

You'll get output like:
```
INFO: Using layout-based processing...
INFO: Output saved to: output_reflowed.png
INFO: Preview saved to: output_reflowed_preview.png
INFO: Creating word segmentation visualization...
INFO: Word segmentation visualization saved to: output_word_segmentation.png
INFO:   Total words detected: 245
```

Much better than silence! 🎉
