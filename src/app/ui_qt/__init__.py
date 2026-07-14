"""PySide6/Qt desktop client for Chessy (Chess.com-style UI)."""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Launch the Qt overlay client."""
    from src.app.ui_qt.app_entry import run

    run()
