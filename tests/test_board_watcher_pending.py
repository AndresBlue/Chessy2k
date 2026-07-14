"""Tests that duplicate ready events are suppressed after reference sync."""

import numpy as np

from src.app.board_watcher import BoardStableDetector


def test_no_duplicate_ready_for_same_position():
    det = BoardStableDetector(settle_frames=2)
    base = np.full((200, 200, 3), 100, dtype=np.uint8)
    det.seed(base)

    moved = base.copy()
    moved[50:120, 50:120] = 200
    moved[130:180, 130:180] = 210
    det.poll(moved)
    ready = None
    for _ in range(5):
        result = det.poll(moved)
        if result.kind == "ready":
            ready = result
            break
    assert ready is not None

    dup = det.poll(moved)
    assert dup.kind == "none"

    det.mark_analyzed(ready.image)
    again = det.poll(moved)
    assert again.kind == "none"


def test_next_move_detectable_without_mark_analyzed():
    """After ready, a new move must be detectable even if analysis is still running."""
    det = BoardStableDetector(settle_frames=2)
    base = np.full((200, 200, 3), 90, dtype=np.uint8)
    det.seed(base)

    first = base.copy()
    first[40:100, 40:100] = 180
    first[120:180, 120:180] = 190
    det.poll(first)
    ready = None
    for _ in range(5):
        result = det.poll(first)
        if result.kind == "ready":
            ready = result
            break
    assert ready is not None

    second = first.copy()
    second[10:60, 150:200] = 30
    second[150:200, 10:60] = 35
    det.poll(second)
    opp_ready = None
    for _ in range(8):
        result = det.poll(second)
        if result.kind == "ready":
            opp_ready = result
            break
    assert opp_ready is not None


def test_discard_pending_allows_redetect():
    det = BoardStableDetector(settle_frames=2)
    base = np.full((200, 200, 3), 80, dtype=np.uint8)
    det.seed(base)

    moved = base.copy()
    moved[40:100, 40:100] = 180
    moved[120:180, 120:180] = 190
    det.poll(moved)
    ready = None
    for _ in range(5):
        result = det.poll(moved)
        if result.kind == "ready":
            ready = result
            break
    assert ready is not None

    det.discard_pending()
    det.poll(moved)
    retry = None
    for _ in range(5):
        result = det.poll(moved)
        if result.kind == "ready":
            retry = result
            break
    assert retry is None
