"""Screen capture via mss (thread-local instances, Windows-safe)."""

from __future__ import annotations

import ctypes
import sys
import threading
from dataclasses import dataclass

import cv2
import mss
import numpy as np

from src.app.errors import CaptureError

_tls = threading.local()


def _get_mss() -> mss.mss:
    """One mss instance per thread (required by the library)."""
    inst = getattr(_tls, "mss_instance", None)
    if inst is None:
        inst = mss.mss()
        _tls.mss_instance = inst
    return inst


def virtual_screen_bounds() -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) of the virtual desktop."""
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        left = int(user32.GetSystemMetrics(76))
        top = int(user32.GetSystemMetrics(77))
        width = int(user32.GetSystemMetrics(78))
        height = int(user32.GetSystemMetrics(79))
        return left, top, left + width, top + height
    with mss.mss() as sct:
        mon = sct.monitors[0]
        return mon["left"], mon["top"], mon["left"] + mon["width"], mon["top"] + mon["height"]


@dataclass
class ScreenRegion:
    """Screen rectangle in absolute pixel coordinates (left, top, right, bottom)."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def as_bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def as_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScreenRegion":
        return cls(
            left=int(data["left"]),
            top=int(data["top"]),
            right=int(data["right"]),
            bottom=int(data["bottom"]),
        )

    def is_valid(self) -> bool:
        return self.width >= 32 and self.height >= 32

    def clamp_to_virtual_screen(self) -> "ScreenRegion":
        """Clip region to visible desktop bounds."""
        vl, vt, vr, vb = virtual_screen_bounds()
        left = max(vl, min(self.left, vr))
        top = max(vt, min(self.top, vb))
        right = max(left, min(self.right, vr))
        bottom = max(top, min(self.bottom, vb))
        return ScreenRegion(left, top, right, bottom)


def _bgra_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    raise CaptureError(f"Unexpected frame shape: {frame.shape}")


def capture_region(region: ScreenRegion) -> np.ndarray:
    """Grab a screen region and return an OpenCV BGR image."""
    if not region.is_valid():
        raise CaptureError("Invalid capture region")

    clipped = region.clamp_to_virtual_screen()
    if not clipped.is_valid():
        raise CaptureError("Capture region outside visible screen")

    monitor = {
        "left": clipped.left,
        "top": clipped.top,
        "width": clipped.width,
        "height": clipped.height,
    }
    try:
        shot = _get_mss().grab(monitor)
        return _bgra_to_bgr(np.asarray(shot))
    except Exception as exc:
        raise CaptureError(f"Screen capture failed: {exc}") from exc


def capture_virtual_desktop() -> tuple[np.ndarray, int, int]:
    """Capture the full virtual desktop; returns (BGR image, origin_x, origin_y)."""
    left, top, right, bottom = virtual_screen_bounds()
    region = ScreenRegion(left, top, right, bottom)
    return capture_region(region), left, top
