#!/usr/bin/env python3
"""
Wrapper script to run main.py with logging enabled
Usage: pixi run python run_with_logging.py images/dvurog_p087.png --layout
"""

import sys
import logging

# Configure logging to show INFO and DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO for less detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Set specific loggers to appropriate levels
logging.getLogger('ocr_reflow').setLevel(logging.DEBUG)
logging.getLogger('ocr_reflow.layout').setLevel(logging.DEBUG)
logging.getLogger('ocr_reflow.device_utils').setLevel(logging.DEBUG)

# Now run the main script
if __name__ == "__main__":
    # Import and run main
    sys.path.insert(0, 'src/ocr_reflow')

    # Import the main module
    import main
