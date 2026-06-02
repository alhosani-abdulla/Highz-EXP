"""Shared argparse formatter utilities for CLI scripts.

Provides a single formatter class that uses rich-argparse when available,
and falls back to standard argparse formatter classes otherwise.
"""

from __future__ import annotations

import argparse
import os
import logging
import platform
import shutil
import subprocess
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


def _is_wsl() -> bool:
    """Return True when running inside WSL."""
    return "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ


def _windows_path_to_wsl(path: str) -> str:
    """Convert a Windows path to a WSL path when possible."""
    wslpath = shutil.which("wslpath")
    if wslpath is None:
        return path

    completed = subprocess.run(
        [wslpath, "-u", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or path


def select_file_path(
    *,
    title: str = "Select a file",
    initialdir: str | None = None,
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Open a file picker and return the selected file path.

    In WSL, this prefers a Windows file picker so the dialog can still appear
    when the Python process is running inside Linux.
    """
    if filetypes is None:
        filetypes = [("All files", "*.*")]

    if _is_wsl() and shutil.which("powershell.exe") is not None:
        filter_value = "|".join(f"{label}|{pattern}" for label, pattern in filetypes)
        safe_title = title.replace("'", "''")
        safe_filter = filter_value.replace("'", "''")
        safe_initialdir = (initialdir or "").replace("'", "''")
        powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = '{safe_title}'
$dialog.Filter = '{safe_filter}'
if ('{safe_initialdir}' -ne '') {{
    $dialog.InitialDirectory = '{safe_initialdir}'
}}
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.FileName
}}
"""
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Sta", "-Command", powershell_script],
            check=False,
            capture_output=True,
            text=True,
        )
        selected_path = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        if not selected_path:
            return None
        return _windows_path_to_wsl(selected_path)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is required to open a file dialog on this system") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected_path = filedialog.askopenfilename(
            title=title,
            initialdir=initialdir,
            filetypes=filetypes,
        )
    finally:
        root.destroy()

    return selected_path or None


def select_save_path(
    *,
    title: str = "Save file as",
    initialdir: str | None = None,
    initialfile: str | None = None,
    defaultextension: str = "",
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Open a save dialog and return the selected output file path.

    This lets the user choose a directory and type the filename in the same window.
    """
    if filetypes is None:
        filetypes = [("All files", "*.*")]

    if _is_wsl() and shutil.which("powershell.exe") is not None:
        filter_value = "|".join(f"{label}|{pattern}" for label, pattern in filetypes)
        safe_title = title.replace("'", "''")
        safe_filter = filter_value.replace("'", "''")
        safe_initialdir = (initialdir or "").replace("'", "''")
        safe_initialfile = (initialfile or "").replace("'", "''")
        safe_defaultextension = defaultextension.replace("'", "''")
        powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.SaveFileDialog
$dialog.Title = '{safe_title}'
$dialog.Filter = '{safe_filter}'
$dialog.DefaultExt = '{safe_defaultextension}'
if ('{safe_initialdir}' -ne '') {{
    $dialog.InitialDirectory = '{safe_initialdir}'
}}
if ('{safe_initialfile}' -ne '') {{
    $dialog.FileName = '{safe_initialfile}'
}}
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.FileName
}}
"""
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Sta", "-Command", powershell_script],
            check=False,
            capture_output=True,
            text=True,
        )
        selected_path = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        if not selected_path:
            return None
        return _windows_path_to_wsl(selected_path)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("tkinter is required to open a file dialog on this system") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected_path = filedialog.asksaveasfilename(
            title=title,
            initialdir=initialdir,
            initialfile=initialfile,
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
    finally:
        root.destroy()

    return selected_path or None


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


