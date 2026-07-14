"""Tests for strategic feature extraction."""

import chess

from src.engines.strategic_features import (
    STRATEGIC_FEATURE_DIM,
    complexity_score,
    extract_strategic_features,
    extract_strategic_vector,
)


START = chess.Board()


def test_vector_dimension():
    vec = extract_strategic_vector(START)
    assert vec.shape == (STRATEGIC_FEATURE_DIM,)
    assert vec.dtype.name == "float32"


def test_starting_position_sane():
    f = extract_strategic_features(START)
    assert f.legal_moves == 20
    assert f.mobility_white == 20
    assert f.mobility_black == 20
    assert 0.0 <= f.king_safety_white <= 1.0
    assert f.isolated_pawns_white == 0


def test_complexity_in_range():
    c = complexity_score(START)
    assert 0.0 <= c <= 1.5


def test_after_e4_opening():
    board = chess.Board()
    board.push_san("e4")
    after = extract_strategic_features(board)
    assert after.legal_moves == 20
    assert after.extended_center_white > 0
