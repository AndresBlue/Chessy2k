"""Tests for FEN repair heuristics."""

import numpy as np

from src.chess_core.fen_utils import board_matrix_from_fen
from src.chess_core.legal_validator import validate_fen, validate_fen_for_analysis
from src.vision.fen_repair import infer_castling_conservative, repair_board_fen

START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _uniform_probs() -> np.ndarray:
    return np.full((64, 13), 1.0 / 13, dtype=np.float32)


def test_infer_castling_start_position():
    matrix = board_matrix_from_fen(START_PLACEMENT)
    assert infer_castling_conservative(matrix) == "KQkq"


def test_valid_board_not_marked_repaired():
    matrix = board_matrix_from_fen(START_PLACEMENT)
    result = repair_board_fen(matrix, _uniform_probs(), side_hint="white")
    assert validate_fen(result.fen).is_valid
    assert not result.repaired


def test_fix_extra_white_king():
    matrix = board_matrix_from_fen(START_PLACEMENT)
    matrix[4][4] = "K"
    probs = _uniform_probs()
    probs[4 * 8 + 4, 0] = 0.9
    probs[4 * 8 + 4, 6] = 0.05
    probs[7 * 8 + 4, 6] = 0.95

    result = repair_board_fen(matrix, probs, side_hint="white")
    assert validate_fen(result.fen).is_valid
    assert result.repaired
    assert sum(1 for row in result.board_matrix for cell in row if cell == "K") == 1


def test_fix_pawn_on_first_rank():
    matrix = board_matrix_from_fen(START_PLACEMENT)
    matrix[7][3] = "P"

    result = repair_board_fen(matrix, _uniform_probs(), side_hint="white")
    assert validate_fen(result.fen).is_valid
    assert result.repaired
    assert matrix[7][3] == "P"
    assert result.board_matrix[7][3] is None


def test_fix_false_king_on_b1_when_square_empty():
    """Knight left b1: vision sometimes reads K; repair should clear to empty."""
    matrix = board_matrix_from_fen(START_PLACEMENT)
    matrix[7][1] = "K"
    matrix[7][0] = "R"
    matrix[7][2] = "B"
    matrix[7][3] = "Q"
    matrix[7][4] = "K"
    matrix[7][5] = "B"
    matrix[7][6] = "N"
    matrix[7][7] = "R"

    probs = _uniform_probs()
    probs[7 * 8 + 1, 6] = 0.7
    probs[7 * 8 + 1, 0] = 0.25
    probs[7 * 8 + 4, 6] = 0.95

    result = repair_board_fen(matrix, probs, side_hint="white")
    assert validate_fen_for_analysis(result.fen).is_valid
    assert result.board_matrix[7][1] is None
    assert sum(1 for row in result.board_matrix for cell in row if cell == "K") == 1


def test_relaxed_validation_allows_extra_minor_if_trimmed():
    placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBNKBNR"
    fen = f"{placement} w KQkq - 0 1"
    assert validate_fen_for_analysis(fen).is_valid


def test_mask_coordinates_preserves_shape():
    from src.vision.square_preprocess import crop_square_center, mask_chesscom_coordinates

    squares = np.random.randint(0, 255, (64, 64, 64, 3), dtype=np.uint8)
    masked = mask_chesscom_coordinates(squares)
    cropped = crop_square_center(masked)
    assert masked.shape == squares.shape
    assert cropped.shape == squares.shape
