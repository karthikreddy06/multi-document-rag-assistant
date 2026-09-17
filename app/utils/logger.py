"""
Logging Configuration Module.
Provides structured and standardized logging across all application components.
"""

import logging
import sys
from app.config import settings


def setup_logger(name: str = "rag_app") -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        level_name = settings.log_level.upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
