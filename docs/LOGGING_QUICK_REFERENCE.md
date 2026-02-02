# Quick Logging Reference - OCR Reflow

## Enable Logging for Development

Add this to the beginning of your script:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Logging Hierarchy

```
ocr_reflow (package level - WARNING by default)
├── ocr_reflow.main
├── ocr_reflow.cli
├── ocr_reflow.divide_conquer_4d
└── ocr_reflow.layout (if applicable)
```

## Common Logging Patterns

### In Your Code
```python
import logging

logger = logging.getLogger(__name__)

# This replaces: print("Processing started")
logger.info("Processing started")

# This replaces: print(f"Debug info: {var}")
logger.debug(f"Debug info: {var}")

# This replaces: print("Error occurred")
logger.error("Error occurred")
```

### Testing with Logging
```python
import logging
from ocr_reflow import process_document

# Configure logging for test
logging.basicConfig(level=logging.DEBUG)

# Now debug output will show
result = process_document("test.png")
```

## What Gets Logged

| Module | What | Level |
|--------|------|-------|
| main.py | Background color detection | DEBUG |
| main.py | Layout analysis details | DEBUG |
| main.py | Box processing | DEBUG |
| main.py | Processing mode (layout/text) | INFO |
| main.py | File operations (save) | INFO |
| main.py | Statistics (word count) | INFO |
| divide_conquer_4d.py | Algorithm timing | INFO |
| divide_conquer_4d.py | Pair details | DEBUG |
| cli.py | File operations | INFO |
| cli.py | Errors | ERROR |
| cli.py | Help text | INFO |

## Disable Specific Loggers

```python
import logging

# Silence debug from a specific module
logging.getLogger('ocr_reflow.main').setLevel(logging.WARNING)

# Silence info-level messages from divide_conquer
logging.getLogger('ocr_reflow.divide_conquer_4d').setLevel(logging.ERROR)
```

## Log to File

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_reflow.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
```

## Environment Variable Override

```bash
# Set logging level via environment
LOGLEVEL=DEBUG python your_script.py
```

Then in code:
```python
import os
import logging

log_level = os.getenv('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=log_level)
```
