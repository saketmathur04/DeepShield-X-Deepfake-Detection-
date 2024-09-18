"""
DeepShield-X — Logging Utilities

Structured logging with console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "deepshield",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure structured logging with console and optional file handlers.

    Args:
        name: Logger name.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        log_file: Specific log file name.
        console: Whether to log to console.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_dir or log_file:
        if log_file:
            fpath = Path(log_file)
        else:
            fpath = Path(log_dir) / f"{name}.log"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(fpath, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "deepshield") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)
