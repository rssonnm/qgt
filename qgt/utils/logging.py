"""
═══════════════════════════════════════════════════════════════════════════════
  qgt.utils.logging — Structured Logging for QGT Research
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Create a structured logger with consistent formatting.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    level : int, optional
        Logging level. Defaults to INFO.
    
    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(f"qgt.{name}")
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(name)s | %(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level or logging.INFO)
    return logger
