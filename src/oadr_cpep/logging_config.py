# src/oadr_cpep/logging_config.py

import logging
import sys


def setup_logger(name: str = "oadr_cpep") -> logging.Logger:
    """Configure logger for oadr-cpep."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger
