"""Tests for region selector coordinate helpers."""

from src.app.region_coords import region_from_drag
from src.app.screen_capture import ScreenRegion


def test_region_from_drag_normalizes_direction():
    region = region_from_drag(300, 250, 100, 50)
    assert region == ScreenRegion(100, 50, 300, 250)
    assert region.is_valid()


def test_region_from_drag_rejects_tiny_selection():
    region = region_from_drag(10, 10, 20, 20)
    assert not region.is_valid()
