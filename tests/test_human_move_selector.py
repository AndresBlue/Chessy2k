"""Tests for human move selector fallbacks."""

from src.engines.human_move_selector import select_human_move
from src.engines.stockfish_client import EngineMove


def test_empty_top_moves_fallback():
    move = select_human_move([], target_elo=2000)
    assert move.san == "(sin jugadas)"


def test_select_best_when_no_alternatives():
    best = EngineMove(uci="e2e4", san="e4", score_cp=40)
    chosen = select_human_move([best], target_elo=2000)
    assert chosen.uci == "e2e4"
