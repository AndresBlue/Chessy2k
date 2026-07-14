"""Tests for Stockfish UCI client (mocked)."""

import pytest
from unittest.mock import MagicMock, patch

from src.engines.stockfish_client import StockfishClient, EngineMove


def test_parse_info_line():
    line = "info depth 15 multipv 1 score cp 35 nodes 123456 nps 1000000 pv e2e4 e7e5 g1f3"
    parsed = StockfishClient._parse_info(line)
    assert parsed is not None
    assert parsed["score_cp"] == 35
    assert parsed["depth"] == 15
    assert parsed["pv"][0] == "e2e4"


def test_engine_move_score_str():
    m = EngineMove(uci="e2e4", san="e4", score_cp=50)
    assert m.score_str == "+0.50"

    m2 = EngineMove(uci="f7f8q", san="f8=Q", score_mate=3)
    assert "#+3" in m2.score_str or "+3" in m2.score_str


@pytest.mark.skipif(True, reason="Requires Stockfish binary")
def test_stockfish_integration():
    client = StockfishClient("./stockfish")
    with client:
        result = client.analyze(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            depth=5,
            multipv=3,
        )
    assert result.best_move.uci
