"""Tests for black/white board view orientation."""

import chess

from src.chess_core.fen_utils import (
    board_matrix_from_board,
    matrix_from_screen_view,
    screen_matrix_from_fen_matrix,
    uci_square_to_screen,
)
from src.chess_core.legal_validator import validate_fen
from src.chess_core.state_tracker import GameStateTracker
from src.vision.visualizer import draw_best_move_arrow
import numpy as np


START_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_screen_fen_roundtrip_black_view():
    fen_matrix = board_matrix_from_board(chess.Board(START_FEN))
    screen = screen_matrix_from_fen_matrix(fen_matrix, "black")
    restored = matrix_from_screen_view(screen, "black")
    assert restored == fen_matrix


def test_tracker_black_view_turn_and_valid_fen():
    board = chess.Board(START_FEN)
    screen = screen_matrix_from_fen_matrix(board_matrix_from_board(board), "black")
    tracker = GameStateTracker()
    result = tracker.update_from_vision(screen, side_hint="black")
    assert result.turn == "b"
    assert validate_fen(result.fen).is_valid


def test_uci_to_screen_black_view_corners():
    assert uci_square_to_screen("h1", "black") == (0, 0)
    assert uci_square_to_screen("a8", "black") == (7, 7)
    assert uci_square_to_screen("a1", "white") == (7, 0)
    assert uci_square_to_screen("h8", "white") == (0, 7)


def test_tracker_start_position_white_to_move_with_black_view():
    """User plays black (flipped view) but starting position is still white to move."""
    start = chess.Board()
    screen_white = screen_matrix_from_fen_matrix(board_matrix_from_board(start), "white")
    screen_black = screen_matrix_from_fen_matrix(board_matrix_from_board(start), "black")

    tracker = GameStateTracker()
    tracker.update_from_vision(screen_white, side_hint="white", orientation="white")
    result = tracker.update_from_vision(screen_black, side_hint="black", orientation="black")

    assert result.turn == "w"
    assert result.fen.split()[1] == "w"


def test_tracker_after_e4_black_to_move_with_black_view():
    """After 1. e4 the board turn is black regardless of user view orientation."""
    fen = "rnbqkb1r/pppppppp/5n2/8/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 0 1"
    board = chess.Board(fen)
    screen = screen_matrix_from_fen_matrix(board_matrix_from_board(board), "black")

    tracker = GameStateTracker()
    tracker.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    result = tracker.update_from_vision(screen, side_hint="black", orientation="black")

    assert result.turn == "b"
    assert result.fen.split()[1] == "b"


def test_arrow_black_view_uses_screen_coords():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    white_arrow = draw_best_move_arrow(img, "e7e5", orientation="white")
    black_arrow = draw_best_move_arrow(img, "e7e5", orientation="black")
    assert not np.array_equal(white_arrow, black_arrow)
