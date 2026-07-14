"""Tests for capture service UI-thread delivery."""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.app.capture_service import CaptureResult, CaptureService
from src.app.screen_capture import ScreenRegion


def test_capture_delivers_via_result_queue():
    ui_queue: queue.Queue = queue.Queue()
    service = CaptureService(result_queue=ui_queue)
    fake = np.zeros((64, 64, 3), dtype=np.uint8)
    received: list[CaptureResult] = []

    def on_result(result: CaptureResult) -> None:
        received.append(result)

    with patch("src.app.capture_service.capture_region", return_value=fake):
        service.capture_region_async(ScreenRegion(0, 0, 64, 64), on_result)
        callback, result = ui_queue.get(timeout=5.0)
        assert callback is on_result
        assert result.image is fake
        callback(result)
        assert len(received) == 1

    service.shutdown()


def test_capture_never_calls_callback_from_worker_directly_when_queue_set():
    """Worker thread must enqueue; main thread invokes callback."""
    ui_queue: queue.Queue = queue.Queue()
    service = CaptureService(result_queue=ui_queue)
    fake = np.zeros((32, 32, 3), dtype=np.uint8)
    thread_ids: list[int] = []

    def on_result(_result: CaptureResult) -> None:
        thread_ids.append(threading.get_ident())

    with patch("src.app.capture_service.capture_region", return_value=fake):
        service.capture_region_async(ScreenRegion(0, 0, 32, 32), on_result)
        time.sleep(0.5)
        callback, result = ui_queue.get(timeout=2.0)
        callback(result)

    service.shutdown()
    assert len(thread_ids) == 1
