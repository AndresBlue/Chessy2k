"""Split normalized board into 64 square patches."""

from __future__ import annotations

import numpy as np

from src.vision.board_cropper import BOARD_SIZE

SQUARE_SIZE = BOARD_SIZE // 8  # 64


def segment_squares(board_image: np.ndarray) -> np.ndarray:
    """
    Split 512x512 board into 64 square patches.

    Returns array of shape (64, SQUARE_SIZE, SQUARE_SIZE, 3) in row-major order
    from a8 (index 0) to h1 (index 63) for white-oriented boards.
    """
    if board_image.shape[0] != BOARD_SIZE or board_image.shape[1] != BOARD_SIZE:
        raise ValueError(f"Expected {BOARD_SIZE}x{BOARD_SIZE} board image")

    squares = np.zeros((64, SQUARE_SIZE, SQUARE_SIZE, board_image.shape[2]), dtype=board_image.dtype)
    idx = 0
    for rank in range(8):
        for file in range(8):
            y0 = rank * SQUARE_SIZE
            x0 = file * SQUARE_SIZE
            squares[idx] = board_image[y0 : y0 + SQUARE_SIZE, x0 : x0 + SQUARE_SIZE]
            idx += 1
    return squares


def square_index_to_coords(index: int) -> tuple[int, int]:
    """Convert flat index (0-63) to (rank, file) with rank 0 = top."""
    rank = index // 8
    file = index % 8
    return rank, file


def coords_to_square_index(rank: int, file: int) -> int:
    return rank * 8 + file
