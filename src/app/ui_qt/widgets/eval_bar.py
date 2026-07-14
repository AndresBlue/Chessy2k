"""A slim, animated vertical evaluation bar (Chess.com style)."""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from src.app.ui_qt.image_utils import clamp01
from src.app.ui_qt.theme import ChessPalette


class EvalBar(QWidget):
    """Vertical bar where fill height reflects the player's human/engine edge."""

    def __init__(self, palette: ChessPalette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self._fraction = 0.5
        self._post_fraction: float | None = None
        self.setFixedWidth(14)
        self.setMinimumHeight(120)
        self._anim = QPropertyAnimation(self, b"fraction", self)
        self._anim.setDuration(420)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._post_anim = QPropertyAnimation(self, b"postFraction", self)
        self._post_anim.setDuration(420)
        self._post_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_palette(self, palette: ChessPalette) -> None:
        self._palette = palette
        self.update()

    def animate_to(self, fraction: float) -> None:
        target = clamp01(fraction)
        self._post_anim.stop()
        self._post_fraction = None
        self._anim.stop()
        self._anim.setStartValue(self._fraction)
        self._anim.setEndValue(target)
        self._anim.start()

    def animate_dual(self, current: float, post: float) -> None:
        """Animate current vs after-line human WDL with green delta bands."""
        cur = clamp01(current)
        pst = clamp01(post)
        self._anim.stop()
        self._post_anim.stop()
        self._anim.setStartValue(self._fraction)
        self._anim.setEndValue(cur)
        self._post_anim.setStartValue(self._post_fraction if self._post_fraction is not None else cur)
        self._post_anim.setEndValue(pst)
        self._anim.start()
        self._post_anim.start()

    def reset(self) -> None:
        self._anim.stop()
        self._post_anim.stop()
        self._fraction = 0.5
        self._post_fraction = None
        self.update()

    def get_fraction(self) -> float:
        return self._fraction

    def set_fraction(self, value: float) -> None:
        self._fraction = clamp01(value)
        self.update()

    fraction = Property(float, get_fraction, set_fraction)

    def get_post_fraction(self) -> float:
        return self._post_fraction if self._post_fraction is not None else self._fraction

    def set_post_fraction(self, value: float) -> None:
        self._post_fraction = clamp01(value)
        self.update()

    postFraction = Property(float, get_post_fraction, set_post_fraction)

    def _draw_fill(
        self,
        painter: QPainter,
        full: QRectF,
        radius: float,
        fraction: float,
        color: QColor,
    ) -> None:
        fill_h = self.height() * fraction
        if fill_h <= 0:
            return
        fill_rect = QRectF(0, self.height() - fill_h, self.width(), fill_h)
        painter.setBrush(color)
        painter.setClipRect(fill_rect)
        painter.drawRoundedRect(full, radius, radius)
        painter.setClipping(False)

    def _draw_band(
        self,
        painter: QPainter,
        full: QRectF,
        radius: float,
        bottom_fraction: float,
        top_fraction: float,
        color: QColor,
    ) -> None:
        bottom = clamp01(bottom_fraction)
        top = clamp01(top_fraction)
        if top <= bottom:
            return
        y_top = self.height() * (1.0 - top)
        band_h = self.height() * (top - bottom)
        if band_h <= 0:
            return
        band_rect = QRectF(0, y_top, self.width(), band_h)
        painter.setBrush(color)
        painter.setClipRect(band_rect)
        painter.drawRoundedRect(full, radius, radius)
        painter.setClipping(False)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        radius = self.width() / 2.0
        full = QRectF(0, 0, self.width(), self.height())

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._palette.eval_track))
        painter.drawRoundedRect(full, radius, radius)

        current = self._fraction
        after = self._post_fraction
        base_green = QColor(self._palette.accent)
        light_green = QColor(self._palette.accent_hover)
        dark_green = QColor(self._palette.accent_pressed)

        if after is not None:
            lo = min(current, after)
            self._draw_fill(painter, full, radius, lo, base_green)
            if after > current:
                self._draw_band(painter, full, radius, current, after, light_green)
            elif current > after:
                self._draw_band(painter, full, radius, after, current, dark_green)
        else:
            self._draw_fill(painter, full, radius, current, base_green)

        mid_y = self.height() / 2.0
        marker = QColor(self._palette.text)
        marker.setAlpha(70)
        painter.setPen(marker)
        painter.drawLine(0, int(mid_y), self.width(), int(mid_y))
        painter.end()
