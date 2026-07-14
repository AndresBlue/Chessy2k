"""Preprocess square patches before classification."""

from __future__ import annotations

import numpy as np


def mask_chesscom_coordinates(squares: np.ndarray) -> np.ndarray:
    """
    Mask Chess.com coordinate glyphs in square corners.

    Rank labels sit top-left on the a-file; file labels bottom-right on rank 1.
    Matrix layout: rank 0 = 8th rank, rank 7 = 1st rank.
    """
    masked = squares.copy()
    n, h, w = masked.shape[0], masked.shape[1], masked.shape[2]
    margin_y = max(6, h // 6)
    margin_x = max(6, w // 6)
    center = masked[:, h // 2 : h // 2 + 1, w // 2 : w // 2 + 1, :]

    for idx in range(n):
        rank, file = idx // 8, idx % 8
        patch = masked[idx]

        if file == 0:
            patch[:margin_y, :margin_x] = center[idx, 0, 0]
        if rank == 7:
            patch[h - margin_y :, w - margin_x :] = center[idx, 0, 0]
            patch[h - margin_y :, :margin_x] = center[idx, 0, 0]

        masked[idx] = patch

    return masked


def crop_square_center(squares: np.ndarray, ratio: float = 0.76) -> np.ndarray:
    """Keep central region of each patch to reduce border noise."""
    n, h, w, c = squares.shape
    ch, cw = int(h * ratio), int(w * ratio)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = squares[:, y0 : y0 + ch, x0 : x0 + cw, :]
    out = np.zeros_like(squares)
    oy = (h - ch) // 2
    ox = (w - cw) // 2
    out[:, oy : oy + ch, ox : ox + cw, :] = cropped
    return out
