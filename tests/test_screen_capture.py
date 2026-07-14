"""Tests for mss-based screen capture."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app.errors import CaptureError
from src.app.screen_capture import ScreenRegion, capture_region, virtual_screen_bounds


def test_screen_region_validity():
    assert ScreenRegion(0, 0, 100, 100).is_valid()
    assert not ScreenRegion(0, 0, 10, 10).is_valid()


def test_virtual_screen_bounds():
    left, top, right, bottom = virtual_screen_bounds()
    assert right > left
    assert bottom > top


def test_capture_region_invalid():
    with pytest.raises(CaptureError):
        capture_region(ScreenRegion(0, 0, 10, 10))


def test_capture_region_mss():
    fake = np.zeros((64, 64, 4), dtype=np.uint8)
    mock_grab = MagicMock(return_value=MagicMock(__array__=lambda: fake))
    mock_mss = MagicMock()
    mock_mss.grab = mock_grab

    with patch("src.app.screen_capture._get_mss", return_value=mock_mss):
        img = capture_region(ScreenRegion(0, 0, 64, 64))
    assert img.shape == (64, 64, 3)
    mock_grab.assert_called_once()
