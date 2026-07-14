"""Qt application bootstrap for the Chessy overlay client."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication


def run() -> None:
    # Pass fractional scale factors through so logical/physical mapping stays
    # predictable; we capture screen regions in physical pixels via Win32.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Chessy")
    app.setApplicationDisplayName("Chessy")

    from src.app.ui_qt.app_window import AppWindow

    window = AppWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
