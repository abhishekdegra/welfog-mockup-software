"""
Application logging setup — info / warning / error channels.

Call `configure_logging()` once at startup. Modules then use
`logging.getLogger(__name__)`.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from ..config import get_config


_CONFIGURED = False


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure the root `mockup` logger with console + optional rotating file.

    Safe to call more than once; subsequent calls only adjust the level.
    """
    global _CONFIGURED
    cfg = get_config()
    resolved = (level or cfg.log_level or "INFO").upper()
    numeric = getattr(logging, resolved, logging.INFO)

    logger = logging.getLogger("mockup")
    logger.setLevel(numeric)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not _CONFIGURED:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(numeric)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if cfg.log_to_file:
            try:
                log_file = cfg.resolved_log_dir() / "app.log"
                file_handler = RotatingFileHandler(
                    str(log_file),
                    maxBytes=int(cfg.log_max_bytes),
                    backupCount=int(cfg.log_backup_count),
                    encoding="utf-8",
                )
                file_handler.setLevel(numeric)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError as exc:
                logger.warning("Could not open log file: %s", exc)

        _CONFIGURED = True
    else:
        for handler in logger.handlers:
            handler.setLevel(numeric)

    logger.debug("Logging configured at %s", resolved)
    return logger


def get_logger(name: str = "mockup") -> logging.Logger:
    """Return a child logger under the mockup namespace."""
    if not name.startswith("mockup"):
        name = f"mockup.{name}"
    return logging.getLogger(name)
