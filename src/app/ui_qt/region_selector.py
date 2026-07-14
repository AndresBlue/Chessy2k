"""Translucent full-desktop region picker (DPI-correct, physical pixels)."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from src.app.region_coords import region_from_drag
from src.app.screen_capture import ScreenRegion
from src.app.ui_qt.theme import ChessPalette


def _physical_cursor_pos(fallback_global: QPoint, dpr: float) -> tuple[int, int]:
    """Cursor position in physical screen pixels (matches mss/Win32 capture)."""
    if sys.platform == "win32":
        point = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return int(point.x), int(point.y)
    return int(round(fallback_global.x() * dpr)), int(round(fallback_global.y() * dpr))


class RegionSelectorOverlay(QWidget):
    """Drag a rectangle over the desktop; emits a ScreenRegion or None."""

    finished = Signal(object)

    def __init__(self, palette: ChessPalette) -> None:
        super().__init__(None)
        self._palette = palette
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        screen = QGuiApplication.primaryScreen()
        self._dpr = screen.devicePixelRatio() if screen else 1.0
        if screen is not None:
            self.setGeometry(screen.virtualGeometry())

        self._dragging = False
        self._emitted = False
        self._start_local = QPoint()
        self._cur_local = QPoint()
        self._start_phys: tuple[int, int] = (0, 0)
        self._cur_phys: tuple[int, int] = (0, 0)

    def show_selector(self) -> None:
        # Cover the entire virtual desktop (all monitors), not a single screen.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.virtualGeometry())
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        self.grabKeyboard()

    # -- events ----------------------------------------------------------
    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.key() == Qt.Key.Key_Escape:
            self._emit(None)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        self._start_local = event.position().toPoint()
        self._cur_local = self._start_local
        self._start_phys = _physical_cursor_pos(event.globalPosition().toPoint(), self._dpr)
        self._cur_phys = self._start_phys
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if not self._dragging:
            return
        self._cur_local = event.position().toPoint()
        self._cur_phys = _physical_cursor_pos(event.globalPosition().toPoint(), self._dpr)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        end_phys = _physical_cursor_pos(event.globalPosition().toPoint(), self._dpr)
        region = region_from_drag(
            self._start_phys[0],
            self._start_phys[1],
            end_phys[0],
            end_phys[1],
        ).clamp_to_virtual_screen()
        self._emit(region if region.is_valid() else None)

    def _emit(self, region: ScreenRegion | None) -> None:
        if self._emitted:
            return
        self._emitted = True
        try:
            self.releaseKeyboard()
        except Exception:
            pass
        self.finished.emit(region)
        self.close()

    # -- painting --------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        veil = QColor(0, 0, 0, 110)
        painter.fillRect(self.rect(), veil)

        accent = QColor(self._palette.accent)
        painter.setPen(QColor("#ffffff"))
        font = QFont(self.font())
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QRect(0, 28, self.width(), 40),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            "Arrastra sobre el tablero   |   ESC para cancelar",
        )

        if self._dragging:
            rect = QRect(self._start_local, self._cur_local).normalized()
            # Punch a clearer window inside the veil.
            clear = QColor(0, 0, 0, 0)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, clear)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(accent, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            label = f"{rect.width()} x {rect.height()}"
            painter.setPen(QColor("#ffffff"))
            painter.drawText(rect.adjusted(0, -26, 0, 0).topLeft() + QPoint(2, 0), label)
        painter.end()
