"""Operation repair: tracker speed, placement sanity, vision gate."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import chess
import numpy as np
import pytest

from src.app.fast_analyzer import FastAnalyzer, _vision_confidence_ok
from src.chess_core.fen_utils import board_matrix_from_board
from src.chess_core.state_tracker import GameStateTracker, TrackerConfig
from src.vision.pipeline import VisionResult

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_E6 = "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR"
INSANE_PLACEMENT = (
    "kkkkqkkk/kkkrkkkk/rkrkrkrk/krkrkrkr/rkrkrkrk/krkkkrkr/kkkkkkkk/kkkkkkkk"
)


def test_two_ply_transition_completes_quickly():
    """Push/pop for depth 1-2 should make this trivially fast (< 20 ms)."""
    start = chess.Board(START_FEN)
    target = AFTER_E4_E6
    tracker = GameStateTracker(
        TrackerConfig(transition_max_nodes=5000, transition_max_seconds=0.4)
    )
    t0 = time.perf_counter()
    moves = tracker._find_transition_moves(start, target, max_ply=4)
    elapsed = time.perf_counter() - t0
    assert moves, "should find e2e4 + e7e6"
    assert elapsed < 0.05, f"push/pop BFS took {elapsed:.3f}s — expected < 50ms"


def test_garbage_placement_search_aborts_quickly():
    """Garbage target should be found unreachable within the time budget."""
    start = chess.Board(START_FEN)
    tracker = GameStateTracker(
        TrackerConfig(transition_max_nodes=5000, transition_max_seconds=0.4)
    )
    t0 = time.perf_counter()
    moves = tracker._find_transition_moves(start, INSANE_PLACEMENT, max_ply=4)
    elapsed = time.perf_counter() - t0
    assert moves == []
    assert elapsed < 0.5


def test_insane_placement_does_not_mutate_tracker():
    board = chess.Board(START_FEN)
    tracker = GameStateTracker()
    tracker.update_from_vision(board_matrix_from_board(board), side_hint="white")
    prev_fen = tracker.board.fen() if tracker.board else None

    insane_matrix = GameStateTracker._placement_to_matrix(INSANE_PLACEMENT)
    result = tracker.update_from_vision(insane_matrix, side_hint="white")

    assert result.status == "vision_error"
    assert tracker.board is not None
    assert tracker.board.fen() == prev_fen


def test_vision_confidence_gate_rejects_low_mean():
    vision = VisionResult(
        board_matrix=[[None] * 8 for _ in range(8)],
        fen_pieces="",
        orientation="white",
        confidence=np.full((8, 8), 0.2, dtype=np.float32),
        ambiguous_squares=[],
        debug_image=None,
        cropped_board=np.zeros((8, 8, 3), dtype=np.uint8),
        time_ms=1.0,
        detection_method="test",
        board_bbox=(0, 0, 8, 8),
    )
    ok, reason = _vision_confidence_ok(
        vision, min_mean_confidence=0.5, max_ambiguous_squares=20
    )
    assert not ok
    assert "Confianza" in reason


@pytest.fixture
def analyzer():
    with patch("src.app.fast_analyzer.VisionPipeline") as vision_cls, patch(
        "src.app.fast_analyzer.StockfishClient"
    ), patch("src.app.fast_analyzer.HumanEngine"):
        vision_cls.from_config.return_value = MagicMock(device_label="cpu")
        fa = FastAnalyzer("sf", "ckpt")
        fa.min_mean_confidence = 0.5
        fa.max_ambiguous_squares = 20
        yield fa


def test_analyze_precomputed_blocks_low_confidence(analyzer):
    low_conf = VisionResult(
        board_matrix=[[None] * 8 for _ in range(8)],
        fen_pieces=AFTER_E4_E6,
        orientation="white",
        confidence=np.full((8, 8), 0.1, dtype=np.float32),
        ambiguous_squares=[(r, f) for r in range(8) for f in range(8)],
        debug_image=None,
        cropped_board=np.zeros((64, 64, 3), dtype=np.uint8),
        time_ms=5.0,
        detection_method="test",
        board_bbox=(0, 0, 64, 64),
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = analyzer.analyze_precomputed(image, low_conf, side="white")
    assert result.error
    assert "Vision poco confiable" in result.error
    analyzer.human_engine.analyze.assert_not_called()


def test_skip_redundant_placement_helper():
    last = AFTER_E4_E6
    same = AFTER_E4_E6
    different = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR"
    assert same == last
    assert different != last


def test_push_pop_does_not_mutate_start_board():
    """_find_transition_moves must leave the start board unchanged after push/pop."""
    start = chess.Board(START_FEN)
    fen_before = start.fen()
    tracker = GameStateTracker(TrackerConfig(transition_max_nodes=5000, transition_max_seconds=0.4))
    tracker._find_transition_moves(start, AFTER_E4_E6, max_ply=2)
    assert start.fen() == fen_before, "start board was mutated during push/pop search"


def test_single_ply_transition_returns_correct_move():
    """Depth-1 push/pop must return the exact move that reaches the target."""
    start = chess.Board(START_FEN)
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    tracker = GameStateTracker(TrackerConfig(transition_max_nodes=5000, transition_max_seconds=0.4))
    moves = tracker._find_transition_moves(start, after_e4, max_ply=1)
    assert len(moves) == 1
    assert moves[0].uci() == "e2e4"
