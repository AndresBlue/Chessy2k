"""Tests for side-to-move inference from placement."""

from src.chess_core.turn_infer import infer_side_to_move_from_placement


def test_starting_position_white_to_move():
    placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    assert infer_side_to_move_from_placement(placement) == "white"


def test_after_e4_black_to_move():
    placement = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    assert infer_side_to_move_from_placement(placement) == "black"


def test_after_e4_e5_white_to_move():
    placement = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
    assert infer_side_to_move_from_placement(placement) == "white"
