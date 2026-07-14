"""Tests for zobrist hashing."""

import chess

from src.chess_core.zobrist import zobrist_hash


def test_zobrist_same_position():
    board = chess.Board()
    assert zobrist_hash(board) == zobrist_hash(chess.Board())


def test_zobrist_changes_after_move():
    board = chess.Board()
    h0 = zobrist_hash(board)
    board.push(chess.Move.from_uci("e2e4"))
    assert zobrist_hash(board) != h0
