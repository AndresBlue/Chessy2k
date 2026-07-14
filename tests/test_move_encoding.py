"""Tests for AlphaZero move encoding."""

import chess
import numpy as np

from src.chess_core.move_encoder import (
    board_to_tensor,
    legal_moves_flat_mask,
    move_to_index,
    index_to_move,
    INPUT_CHANNELS,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_board_tensor_shape():
    board = chess.Board(START_FEN)
    tensor = board_to_tensor(board)
    assert tensor.shape == (18, 8, 8)


def test_move_roundtrip():
    board = chess.Board(START_FEN)
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move, board)
    decoded = index_to_move(idx, board)
    assert decoded == move


def test_legal_mask_has_moves():
    board = chess.Board(START_FEN)
    mask = legal_moves_flat_mask(board)
    assert mask.sum() == 20


def test_all_legal_moves_decodable():
    board = chess.Board(START_FEN)
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        decoded = index_to_move(idx, board)
        assert decoded == move, f"Failed for {move.uci()}"


def test_straight_underpromotion_roundtrip():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("a7a8r")
    idx = move_to_index(move, board)
    assert index_to_move(idx, board) == move
