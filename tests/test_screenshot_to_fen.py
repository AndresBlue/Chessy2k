"""Tests for screenshot to FEN vision pipeline."""

import tempfile
from pathlib import Path

import cv2
import chess
import numpy as np
import pytest

from src.chess_core.fen_utils import board_matrix_from_board, matrix_to_placement
from src.vision.board_cropper import crop_board
from src.vision.screenshot_detector import BoardDetection, detect_board
from src.vision.square_segmenter import segment_squares
from src.vision.synthetic_renderer import SyntheticBoardRenderer
from src.vision.fen_builder import build_fen, detect_orientation


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_render_and_detect():
    renderer = SyntheticBoardRenderer(theme="green")
    img = renderer.render(START_FEN)
    detection = detect_board(img)
    assert detection.bbox[2] > 0
    assert detection.bbox[3] > 0


def test_crop_and_segment():
    renderer = SyntheticBoardRenderer(board_size=512)
    img = renderer.render(START_FEN)
    det = BoardDetection(bbox=(0, 0, 512, 512), corners=None, confidence=1.0, method="test")
    cropped = crop_board(img, det)
    assert cropped.shape == (512, 512, 3)
    squares = segment_squares(cropped)
    assert squares.shape == (64, 64, 64, 3)


def test_orientation_detection():
    board = chess.Board(START_FEN)
    matrix = board_matrix_from_board(board)
    orientation = detect_orientation(matrix)
    assert orientation == "white"


def test_synthetic_pipeline_roundtrip():
    """Full synthetic render -> save -> detect -> classify (untrained) -> FEN."""
    import tempfile
    import os

    renderer = SyntheticBoardRenderer(theme="lichess", board_size=512)
    img = renderer.render(START_FEN)

    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "board.png")
        cv2.imwrite(img_path, img)

        from src.vision.pipeline import VisionPipeline
        pipeline = VisionPipeline()  # untrained model
        result = pipeline.process(img_path)

        assert result.fen_pieces is not None
        assert result.orientation in ("white", "black")
        assert result.confidence.shape == (8, 8)
