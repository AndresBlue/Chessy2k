"""Regression: placement inference must not hang on unreachable positions."""

import time

from src.chess_core.state_tracker import GameStateTracker
from src.chess_core.turn_infer import infer_side_to_move_from_placement

# Vision misread / inverted placement from user log (unreachable in opening BFS).
_BAD_PLACEMENT = "RNBKQBNR/PPP1PPPP/3P4/8/8/8/pppppppp/rnbkqbnr"


def test_unreachable_placement_bfs_returns_quickly():
    t0 = time.perf_counter()
    result = infer_side_to_move_from_placement(_BAD_PLACEMENT, max_nodes=10_000)
    elapsed = time.perf_counter() - t0
    assert result is None
    assert elapsed < 2.0


def test_tracker_initial_uses_side_hint_without_bfs():
    matrix = GameStateTracker._placement_to_matrix(_BAD_PLACEMENT)
    tracker = GameStateTracker()
    t0 = time.perf_counter()
    result = tracker.update_from_vision(matrix, side_hint="white")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5
    assert result.turn == "w"
    assert result.status == "initial"
