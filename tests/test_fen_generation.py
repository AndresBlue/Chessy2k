"""Tests for FEN generation and parsing."""

import chess
import pytest

from src.chess_core.fen_utils import (
    board_matrix_from_board,
    fen_from_matrix,
    matrix_to_placement,
    parse_fen,
    placement_from_class_matrix,
    PIECE_TO_CLASS,
)
import numpy as np


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_parse_starting_position():
    board = parse_fen(START_FEN)
    assert board.fen() == START_FEN


def test_matrix_roundtrip():
    board = chess.Board(START_FEN)
    matrix = board_matrix_from_board(board)
    placement = matrix_to_placement(matrix)
    assert placement == board.board_fen()


def test_fen_from_matrix():
    board = chess.Board(START_FEN)
    matrix = board_matrix_from_board(board)
    fen = fen_from_matrix(matrix, turn="w", castling="KQkq", en_passant="-", halfmove=0, fullmove=1)
    assert fen == START_FEN


def test_class_matrix_to_placement():
    board = chess.Board(START_FEN)
    matrix = board_matrix_from_board(board)
    class_matrix = np.zeros((8, 8), dtype=np.int64)
    for r in range(8):
        for f in range(8):
            piece = matrix[r][f]
            class_matrix[r, f] = PIECE_TO_CLASS[piece] if piece else 0
    placement = placement_from_class_matrix(class_matrix)
    assert placement == board.board_fen()
