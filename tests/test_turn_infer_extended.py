"""Tests for turn inference beyond opening BFS."""

from src.chess_core.turn_infer import infer_side_to_move_from_placement

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
MIDDLEGAME = "r1bq1rk1/ppp2ppp/2n1bn2/3pp3/2PP4/2N1BN2/PP3PPP/R1BQ1RK1"


def test_reachability_middlegame_ambiguous_without_hint():
    # Both w/b turns can pass syntactic is_valid(); overlay supplies side_hint instead.
    assert infer_side_to_move_from_placement(MIDDLEGAME) is None


def test_after_e4_black():
    assert infer_side_to_move_from_placement(AFTER_E4) == "black"


def test_start_white():
    assert infer_side_to_move_from_placement(START) == "white"
