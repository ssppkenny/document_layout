"""Centralized logging configuration for ocr_reflow.

Call setup_logging() in each entry point to configure the root logger.
"""

import logging
import sys
from pathlib import Path


def setup_logging(*, log_path: str | None = None, level: int = logging.INFO):
    """Configure the root logger to write all messages to a file (or stderr).

    Args:
        log_path: Path to the log file.  If None, logs go to stderr.
        level:    Minimum logging level (default INFO).
    """
    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8"
        )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)

    # Ensure the ocr_reflow package logger doesn't silently drop messages
    logging.getLogger("ocr_reflow").setLevel(logging.INFO)
