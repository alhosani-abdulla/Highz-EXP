"""Shared argparse formatter utilities for CLI scripts.

Provides a single formatter class that uses rich-argparse when available,
and falls back to standard argparse formatter classes otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys

try:
    from rich_argparse import (  # type: ignore[import-not-found]
        ArgumentDefaultsRichHelpFormatter,
        RawDescriptionRichHelpFormatter,
    )
except ImportError:
    ArgumentDefaultsRichHelpFormatter = argparse.ArgumentDefaultsHelpFormatter
    RawDescriptionRichHelpFormatter = argparse.RawDescriptionHelpFormatter


class RichHelpFormatter(ArgumentDefaultsRichHelpFormatter, RawDescriptionRichHelpFormatter):
    """Formatter combining defaults and raw description formatting."""


def setup_cli_logging(
    *,
    verbose: bool = False,
    logger_name: str | None = None,
    default_level: int = logging.WARNING,
) -> logging.Logger:
    """Configure CLI logging with RichHandler fallback and return a logger.

    Parameters
    ----------
    verbose : bool, optional
        If True, use INFO level. Otherwise use ``default_level``.
    logger_name : str or None, optional
        Logger name to return. If None, return root logger.
    default_level : int, optional
        Logging level used when ``verbose`` is False.
    """
    log_level = logging.INFO if verbose else default_level

    try:
        from rich.logging import RichHandler  # type: ignore[import-not-found]
    except ImportError:
        RichHandler = None

    if RichHandler is not None:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    return logging.getLogger(logger_name)
