"""Structured logging for the overlay application."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def _has_handler(root: logging.Logger, handler_type: type[logging.Handler]) -> bool:
    return any(isinstance(handler, handler_type) for handler in root.handlers)


def setup_logging(level: int = logging.INFO, *, log_dir: Path | None = None) -> None:
    """Configure root logger once for the desktop client."""
    root = logging.getLogger()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not _has_handler(root, logging.StreamHandler):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"chessy_{datetime.now():%Y%m%d}.log"
        existing_files = [
            getattr(handler, "baseFilename", None)
            for handler in root.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        if str(log_file) not in existing_files:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
