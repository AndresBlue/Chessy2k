"""Tests for legal validation and GameStateTracker."""

import chess

from src.chess_core.legal_validator import validate_fen, validate_placement_only
from src.chess_core.state_tracker import GameStateTracker
from src.chess_core.fen_utils import board_matrix_from_board


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_valid_starting_position():
    result = validate_fen(START_FEN)
    assert result.is_valid
    assert result.legal_move_count == 20


def test_invalid_fen():
    result = validate_fen("invalid")
    assert not result.is_valid
    assert len(result.errors) > 0


def test_placement_only():
    placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    result = validate_placement_only(placement)
    assert result.is_valid


def test_initial_from_vision():
    board = chess.Board(START_FEN)
    matrix = board_matrix_from_board(board)
    tracker = GameStateTracker()
    result = tracker.update_from_vision(matrix, orientation="white", side_hint="white")
    assert result.status == "initial"
    assert result.placement == board.board_fen()


def test_single_move_transition():
    board = chess.Board(START_FEN)
    matrix0 = board_matrix_from_board(board)
    tracker = GameStateTracker()
    tracker.update_from_vision(matrix0, side_hint="white")

    board.push(chess.Move.from_uci("e2e4"))
    matrix1 = board_matrix_from_board(board)
    result = tracker.update_from_vision(matrix1, side_hint="white")
    assert result.status == "ok"
    assert "e2e4" in result.candidate_moves


def test_set_fen():
    tracker = GameStateTracker()
    result = tracker.set_fen(START_FEN)
    assert result.status == "ok"
    assert tracker.has_history
