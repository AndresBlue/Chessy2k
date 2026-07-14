"""Zobrist hashing for transposition tables and caches."""

from __future__ import annotations

import random

import chess

# Deterministic seed for reproducible hashes across sessions
_RANDOM = random.Random(0xC055EE)


def _rand64() -> int:
    return _RANDOM.getrandbits(64)


# Piece keys: [color][piece_type][square]
PIECE_KEYS: list[list[list[int]]] = [
    [[_rand64() for _ in range(64)] for _ in range(7)] for _ in range(2)
]

SIDE_KEY = _rand64()
CASTLING_KEYS = [_rand64() for _ in range(4)]
EN_PASSANT_KEYS = [_rand64() for _ in range(8)]


def zobrist_hash(board: chess.Board) -> int:
    """Compute Zobrist hash for a python-chess Board."""
    h = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            h ^= PIECE_KEYS[piece.color][piece.piece_type][square]

    if board.turn == chess.BLACK:
        h ^= SIDE_KEY

    cr = board.castling_rights
    if cr & chess.BB_H1:
        h ^= CASTLING_KEYS[0]
    if cr & chess.BB_A1:
        h ^= CASTLING_KEYS[1]
    if cr & chess.BB_H8:
        h ^= CASTLING_KEYS[2]
    if cr & chess.BB_A8:
        h ^= CASTLING_KEYS[3]

    if board.ep_square is not None:
        h ^= EN_PASSANT_KEYS[chess.square_file(board.ep_square)]

    return h


class ZobristHasher:
    """Incremental Zobrist hasher."""

    def __init__(self, board: chess.Board | None = None):
        self.hash = zobrist_hash(board) if board else 0

    def update_move(self, board: chess.Board, move: chess.Move) -> int:
        """Update hash after making a move (board must already be pushed)."""
        self.hash = zobrist_hash(board)
        return self.hash

    def reset(self, board: chess.Board) -> int:
        self.hash = zobrist_hash(board)
        return self.hash
