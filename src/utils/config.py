# src/utils/config.py
"""
Shared config loading and logging setup utilities.
"""

import logging
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
