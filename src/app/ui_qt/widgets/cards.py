"""Card-style container widgets with soft elevation shadows."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from src.app.ui_qt.theme import ChessPalette, apply_shadow


class Card(QFrame):
    """A rounded surface panel with an optional title and drop shadow."""

    def __init__(
        self,
        title: str | None = None,
        *,
        inner: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CardInner" if inner else "Card")
        self._layout = QVBoxLayout(self)
        margin = 12 if inner else 16
        self._layout.setContentsMargins(margin, margin, margin, margin)
        self._layout.setSpacing(10)
        self._title_label: QLabel | None = None
        if title is not None:
            self._title_label = QLabel(title.upper())
            self._title_label.setObjectName("CardTitle")
            self._layout.addWidget(self._title_label)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget, *, stretch: int = 0, align: Qt.AlignmentFlag | None = None) -> None:
        if align is not None:
            self._layout.addWidget(widget, stretch, align)
        else:
            self._layout.addWidget(widget, stretch)

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_stretch(self, stretch: int = 1) -> None:
        self._layout.addStretch(stretch)

    def elevate(self, palette: ChessPalette, *, blur: int = 30, dy: int = 8) -> None:
        apply_shadow(self, palette, blur=blur, dy=dy)


def section_title(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("CardTitle")
    return label
