"""FEN parsing, building, and piece placement utilities."""

from __future__ import annotations

import re
from typing import Literal

import chess

# 13-class vision labels
PIECE_CLASSES: list[str] = [
    "empty",
    "P", "N", "B", "R", "Q", "K",
    "p", "n", "b", "r", "q", "k",
]

CLASS_TO_PIECE: dict[int, str | None] = {
    0: None,
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
}

PIECE_TO_CLASS: dict[str, int] = {v: k for k, v in CLASS_TO_PIECE.items() if v is not None}

Orientation = Literal["white", "black"]


def piece_placement_from_board(board: chess.Board) -> str:
    """Return only the piece-placement segment of FEN."""
    return board.board_fen()


def board_matrix_from_fen(fen: str) -> list[list[str | None]]:
    """Parse piece placement into 8x8 matrix (rank 8 at row 0)."""
    placement = fen.split()[0] if " " in fen else fen
    matrix: list[list[str | None]] = []
    for rank_str in placement.split("/"):
        row: list[str | None] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend([None] * int(ch))
            else:
                row.append(ch)
        matrix.append(row)
    return matrix


def board_matrix_from_board(board: chess.Board) -> list[list[str | None]]:
    """Extract 8x8 piece matrix from python-chess Board (rank 8 = row 0)."""
    matrix: list[list[str | None]] = []
    for rank in range(7, -1, -1):
        row: list[str | None] = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row.append(piece.symbol() if piece else None)
        matrix.append(row)
    return matrix


def fen_from_matrix(
    matrix: list[list[str | None]],
    turn: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> str:
    """Build full FEN from 8x8 piece matrix."""
    placement = matrix_to_placement(matrix)
    return f"{placement} {turn} {castling} {en_passant} {halfmove} {fullmove}"


def fen_side_to_move(fen: str) -> str | None:
    """Return ``w``/``b`` from a FEN string, or ``None`` if missing."""
    parts = fen.split()
    return parts[1] if len(parts) > 1 else None


def matrix_to_placement(matrix: list[list[str | None]]) -> str:
    """Convert 8x8 matrix to FEN piece-placement string."""
    ranks: list[str] = []
    for row in matrix:
        rank_str = ""
        empty = 0
        for cell in row:
            if cell is None:
                empty += 1
            else:
                if empty:
                    rank_str += str(empty)
                    empty = 0
                rank_str += cell
        if empty:
            rank_str += str(empty)
        ranks.append(rank_str)
    return "/".join(ranks)


def placement_from_class_matrix(class_matrix) -> str:
    """Convert 8x8 class indices to FEN placement."""
    import numpy as np

    matrix: list[list[str | None]] = []
    for row in class_matrix:
        fen_row: list[str | None] = []
        for cls in row:
            fen_row.append(CLASS_TO_PIECE[int(cls)])
        matrix.append(fen_row)
    return matrix_to_placement(matrix)


def parse_fen(fen: str) -> chess.Board:
    """Parse FEN into python-chess Board; raises ValueError if invalid."""
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {fen}") from exc
    return board


def default_castling_for_placement(placement: str) -> str:
    """Infer conservative castling rights from piece placement."""
    try:
        board = chess.Board(f"{placement} w - - 0 1")
    except ValueError:
        return "-"

    matrix = board_matrix_from_fen(placement)
    from src.vision.fen_repair import infer_castling_conservative

    return infer_castling_conservative(matrix)


def normalize_fen(fen: str) -> str:
    """Normalize FEN string via python-chess."""
    board = parse_fen(fen)
    return board.fen()


def count_pieces(matrix: list[list[str | None]]) -> dict[str, int]:
    """Count pieces in matrix."""
    counts: dict[str, int] = {}
    for row in matrix:
        for cell in row:
            if cell:
                counts[cell] = counts.get(cell, 0) + 1
    return counts


FEN_PATTERN = re.compile(
    r"^([pnbrqkPNBRQK1-8]+/){7}[pnbrqkPNBRQK1-8]+ [wb] [KQkq\-]+ [a-h3-6\-] \d+ \d+$"
)


def is_valid_fen_format(fen: str) -> bool:
    """Quick regex check for FEN format."""
    return bool(FEN_PATTERN.match(fen.strip()))


def matrix_from_screen_view(
    screen_matrix: list[list[str | None]],
    view: Orientation,
) -> list[list[str | None]]:
    """
    Convert vision matrix to standard FEN matrix (rank 8 = row 0).

    Chess.com with user playing black flips the board 180° (rank 1 top, h-file left).
    """
    if view == "black":
        return [list(reversed(row)) for row in reversed(screen_matrix)]
    return [row[:] for row in screen_matrix]


def screen_matrix_from_fen_matrix(
    fen_matrix: list[list[str | None]],
    view: Orientation,
) -> list[list[str | None]]:
    """Map standard FEN matrix to screen layout (inverse of ``matrix_from_screen_view``)."""
    if view == "black":
        return [list(reversed(row)) for row in reversed(fen_matrix)]
    return [row[:] for row in fen_matrix]


def uci_square_to_screen(square: str, view: Orientation) -> tuple[int, int]:
    """
    UCI square → screen grid for overlay drawing.

    Returns ``(screen_rank, screen_file)`` with 0 = top / left of the capture.
    """
    file_idx = ord(square[0]) - ord("a")
    rank_idx = int(square[1]) - 1
    if view == "black":
        return rank_idx, 7 - file_idx
    return 7 - rank_idx, file_idx
