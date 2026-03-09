"""Shared argparse formatter utilities for CLI scripts.

Provides a single formatter class that uses rich-argparse when available,
and falls back to standard argparse formatter classes otherwise.
"""

from __future__ import annotations

import argparse

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
