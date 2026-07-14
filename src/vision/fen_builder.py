"""Build FEN piece placement from classified board matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.chess_core.fen_utils import matrix_to_placement, fen_from_matrix


@dataclass
class FenBuildResult:
    placement: str
    fen_pieces: str
    board_matrix: list[list[str | None]]
    orientation: str
    confidence: np.ndarray
    ambiguous_squares: list[tuple[int, int]]


def detect_orientation(board_matrix: list[list[str | None]]) -> str:
    """
    Detect board orientation from piece placement.

    White pieces typically on ranks 6-7 (rows 1-2 from bottom in standard view),
    black on ranks 0-1. For digital screenshots we check king positions.
    """
    white_king_row = -1
    black_king_row = -1
    for rank, row in enumerate(board_matrix):
        for piece in row:
            if piece == "K":
                white_king_row = rank
            elif piece == "k":
                black_king_row = rank

    if white_king_row >= 0 and black_king_row >= 0:
        return "white" if white_king_row > black_king_row else "black"

    # Default: white on bottom (rank 7 in matrix = row index 7)
    white_pieces_low = sum(
        1 for row in board_matrix[4:] for p in row if p and p.isupper()
    )
    white_pieces_high = sum(
        1 for row in board_matrix[:4] for p in row if p and p.isupper()
    )
    return "white" if white_pieces_low >= white_pieces_high else "black"


def build_fen(
    board_matrix: list[list[str | None]],
    confidence: np.ndarray,
    ambiguous_squares: list[tuple[int, int]],
    turn: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> FenBuildResult:
    """Build FEN from classified board."""
    orientation = detect_orientation(board_matrix)
    placement = matrix_to_placement(board_matrix)

    return FenBuildResult(
        placement=placement,
        fen_pieces=placement,
        board_matrix=board_matrix,
        orientation=orientation,
        confidence=confidence,
        ambiguous_squares=ambiguous_squares,
    )


def build_full_fen(
    board_matrix: list[list[str | None]],
    turn: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> str:
    return fen_from_matrix(board_matrix, turn, castling, en_passant, halfmove, fullmove)
