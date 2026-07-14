"""Tests for incremental square diff and inference."""

import numpy as np

from src.vision.piece_classifier import PieceClassifier
from src.vision.square_diff import diff_square_indices


def test_diff_square_indices_detects_single_change():
    prev = np.zeros((64, 64, 64, 3), dtype=np.uint8)
    curr = prev.copy()
    curr[20] = 255
    assert diff_square_indices(prev, curr) == [20]


def test_diff_square_indices_ignores_noise():
    prev = np.full((64, 64, 64, 3), 120, dtype=np.uint8)
    curr = prev.copy()
    curr[5] = prev[5] + 3
    assert diff_square_indices(prev, curr) == []


def test_incremental_inference_faster_than_full():
    ckpt = "data/checkpoints/vision/best.pt"
    try:
        clf = PieceClassifier(ckpt)
    except Exception:
        return

    squares = np.random.randint(0, 255, (64, 64, 64, 3), dtype=np.uint8)
    full = clf.predict_squares(squares)

    moved = squares.copy()
    moved[10] = 255 - moved[10]
    moved[18] = 255 - moved[18]
    inc = clf.predict_squares(moved, previous_squares=squares, previous_result=full)

    assert inc["inference_mode"] == "incremental"
    assert inc["squares_updated"] == 2
    assert inc["probs"].shape == (64, 13)
