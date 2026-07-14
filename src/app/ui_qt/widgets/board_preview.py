"""Board preview widget: shows the annotated capture plus an animated timer."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from src.app.ui_qt.image_utils import bgr_to_pixmap, clamp01
from src.app.ui_qt.theme import ChessPalette

_TICK_MS = 16  # ~60 fps for a fluid countdown.


class BoardPreview(QWidget):
    """Renders the captured board and a full-board pie countdown overlay."""

    def __init__(self, palette: ChessPalette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._pixmap: QPixmap | None = None
        self._placeholder = (
            "La captura con la jugada sugerida\naparecera aqui"
        )
        self.setMinimumSize(360, 360)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        self._think_duration_s: float = 0.0
        self._think_deadline: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)

    def set_palette(self, palette: ChessPalette) -> None:
        self._palette = palette
        self.update()

    def set_image_bgr(self, image_bgr: np.ndarray | None) -> None:
        if image_bgr is None or image_bgr.size == 0:
            self._pixmap = None
        else:
            self._pixmap = bgr_to_pixmap(image_bgr)
        self.update()

    def clear_image(self) -> None:
        self._pixmap = None
        self.update()

    # -- think countdown -------------------------------------------------
    def start_think(self, duration_ms: int) -> None:
        self.stop_think()
        if duration_ms <= 0:
            return
        self._think_duration_s = duration_ms / 1000.0
        self._think_deadline = time.monotonic() + self._think_duration_s
        self._timer.start()
        self.update()

    def stop_think(self) -> None:
        self._timer.stop()
        self._think_deadline = None
        self._think_duration_s = 0.0
        self.update()

    @property
    def is_thinking(self) -> bool:
        return self._think_deadline is not None

    def _on_tick(self) -> None:
        if self._think_deadline is None:
            self._timer.stop()
            return
        if time.monotonic() >= self._think_deadline:
            self._think_deadline = None
            self._think_duration_s = 0.0
            self._timer.stop()
        self.update()

    # -- painting --------------------------------------------------------
    def _image_rect(self) -> QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF()
        avail_w = self.width()
        avail_h = self.height()
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return QRectF()
        scale = min(avail_w / pw, avail_h / ph)
        scale = max(scale, 0.01)
        draw_w = pw * scale
        draw_h = ph * scale
        x = (avail_w - draw_w) / 2.0
        y = (avail_h - draw_h) / 2.0
        return QRectF(x, y, draw_w, draw_h)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), QColor(self._palette.preview_bg))

        if self._pixmap is None or self._pixmap.isNull():
            self._paint_placeholder(painter)
            painter.end()
            return

        rect = self._image_rect()
        painter.drawPixmap(rect.toRect(), self._pixmap)
        self._paint_think_pie(painter, rect)
        painter.end()

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor(self._palette.text_muted))
        font = QFont(self.font())
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignCenter),
            self._placeholder,
        )

    def _paint_think_pie(self, painter: QPainter, rect: QRectF) -> None:
        if self._think_deadline is None or self._think_duration_s <= 0:
            return
        remaining = max(0.0, self._think_deadline - time.monotonic())
        fraction = clamp01(remaining / self._think_duration_s)
        if fraction <= 0:
            return

        diameter = min(rect.width(), rect.height()) * 0.96
        cx = rect.center().x()
        cy = rect.center().y()
        circle = QRectF(cx - diameter / 2.0, cy - diameter / 2.0, diameter, diameter)

        accent = QColor(self._palette.accent)

        # Faint full track ring for boundary definition.
        track_pen = QColor(self._palette.text)
        track_pen.setAlpha(28)
        painter.setPen(QPen(track_pen, max(2.0, diameter / 150.0)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(circle)

        # Remaining-time wedge: low opacity so the board stays readable.
        fill = QColor(accent)
        fill.setAlpha(46)
        painter.setBrush(fill)
        edge = QColor(accent)
        edge.setAlpha(120)
        painter.setPen(QPen(edge, max(1.5, diameter / 220.0)))
        start_angle = 90 * 16
        span_angle = int(-360 * fraction * 16)
        painter.drawPie(circle, start_angle, span_angle)
