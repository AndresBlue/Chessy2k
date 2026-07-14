"""Detect which board squares changed between captures."""

from __future__ import annotations

import numpy as np

# Mean absolute pixel diff above this counts as a changed square.
DEFAULT_SQUARE_DIFF_THRESHOLD = 10.0
# Fall back to full-board inference when too many squares differ.
MAX_INCREMENTAL_SQUARES = 12


def diff_square_indices(
    previous: np.ndarray,
    current: np.ndarray,
    threshold: float = DEFAULT_SQUARE_DIFF_THRESHOLD,
) -> list[int]:
    """
    Return flat indices (0-63) of squares that changed between captures.

    Args:
        previous: (64, H, W, 3) BGR patches from last analysis
        current: (64, H, W, 3) BGR patches from new capture
    """
    if previous.shape != current.shape:
        return list(range(64))

    changed: list[int] = []
    for idx in range(previous.shape[0]):
        diff = float(
            np.mean(
                np.abs(
                    previous[idx].astype(np.int16) - current[idx].astype(np.int16)
                )
            )
        )
        if diff > threshold:
            changed.append(idx)
    return changed
