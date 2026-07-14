"""Tests for board change / settle detection."""

import numpy as np

from src.app.board_watcher import BoardStableDetector


def _board(base: int = 100) -> np.ndarray:
    img = np.full((256, 256, 3), base, dtype=np.uint8)
    return img


def test_first_poll_seeds_without_ready():
    det = BoardStableDetector()
    result = det.poll(_board())
    assert result.kind == "none"
    assert det.has_baseline


def test_default_polling_is_fast_for_responsive_auto():
    det = BoardStableDetector()
    assert det.poll_interval_ms <= 150
    det.seed(_board())
    moved = _board()
    moved[80:160, 80:160] = 200
    moved[40:80, 180:220] = 210
    assert det.poll(moved).kind == "motion"
    assert det.poll_interval_ms <= 70


def test_no_trigger_without_change():
    det = BoardStableDetector()
    det.seed(_board())
    result = det.poll(_board())
    assert result.kind == "none"


def test_ignores_single_cell_flicker():
    det = BoardStableDetector()
    base = _board(90)
    det.seed(base)
    flicker = base.copy()
    flicker[60:100, 60:100] = 120
    assert det.poll(flicker).kind == "none"


def test_waits_for_settle_before_ready():
    det = BoardStableDetector(settle_frames=2)
    base = _board(80)
    det.seed(base)

    moved = base.copy()
    moved[80:160, 80:160] = 200
    moved[40:80, 180:220] = 210

    assert det.poll(moved).kind == "motion"
    assert det.poll(moved).kind == "motion"
    final = det.poll(moved)
    assert final.kind == "ready"
    assert final.image is not None


def test_opponent_move_detected_after_player_move():
    det = BoardStableDetector(settle_frames=2)
    base = _board(90)
    det.seed(base)

    player = base.copy()
    player[60:100, 60:100] = 150
    player[180:220, 40:80] = 155
    det.poll(player)
    ready = None
    for _ in range(6):
        result = det.poll(player)
        if result.kind == "ready":
            ready = result
            break
    assert ready is not None
    det.mark_analyzed(ready.image)

    opponent = player.copy()
    opponent[180:220, 180:220] = 30
    opponent[40:80, 140:180] = 35
    det.poll(opponent)
    opp_ready = None
    for _ in range(6):
        result = det.poll(opponent)
        if result.kind == "ready":
            opp_ready = result
            break
    assert opp_ready is not None
