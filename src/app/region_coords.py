"""Coordinate helpers for the lightweight region selector."""

from __future__ import annotations

from src.app.screen_capture import ScreenRegion


def region_from_drag(
    start_root_x: int,
    start_root_y: int,
    end_root_x: int,
    end_root_y: int,
) -> ScreenRegion:
    """Build a screen region from absolute Tk pointer coordinates."""
    left = min(start_root_x, end_root_x)
    top = min(start_root_y, end_root_y)
    right = max(start_root_x, end_root_x)
    bottom = max(start_root_y, end_root_y)
    return ScreenRegion(left, top, right, bottom)
