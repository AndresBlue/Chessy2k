"""Pure helpers for the Qt UI: image conversion, geometry and eval mapping.

These functions are deliberately free of any QWidget/state so they can be unit
tested without a running QApplication (except the QImage builder, which only
needs the lightweight QtGui types).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def clamp01(value: float) -> float:
    """Clamp a float to the inclusive [0, 1] range."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def bgr_to_qimage(image_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV BGR ndarray into a standalone QImage (owns its data)."""
    if image_bgr is None or image_bgr.size == 0:
        return QImage()
    if image_bgr.ndim == 2:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif image_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    bytes_per_line = 3 * width
    qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    # copy() detaches from the temporary numpy buffer.
    return qimage.copy()


def bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR ndarray into a QPixmap."""
    return QPixmap.fromImage(bgr_to_qimage(image_bgr))


def think_fraction(remaining_s: float, duration_s: float) -> float:
    """Fraction of the recommended think time still remaining (1.0 -> 0.0)."""
    if duration_s <= 0:
        return 0.0
    return clamp01(remaining_s / duration_s)


def eval_to_fraction(score_str: str, *, scale: float = 0.42) -> float:
    """Map a Stockfish score string to a 0..1 advantage for the side to move.

    0.5 is balanced, 1.0 means winning, 0.0 means losing. ``score_str`` follows
    the engine convention: ``+1.25`` / ``-0.30`` in pawns or ``#+3`` / ``#-3``
    for forced mates, always from the perspective of the player to move.
    """
    if not score_str:
        return 0.5
    text = score_str.strip()
    if text.startswith("#"):
        body = text[1:].replace("+", "")
        try:
            mate = int(body)
        except ValueError:
            mate = 0
        if mate == 0:
            return 0.5
        return 0.99 if mate > 0 else 0.01
    try:
        pawns = float(text)
    except ValueError:
        return 0.5
    # Logistic squashing keeps small edges near the centre and saturates slowly.
    return clamp01(1.0 / (1.0 + math.exp(-pawns * scale)))
