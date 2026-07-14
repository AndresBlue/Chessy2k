"""Tests for Stockfish client game-over and timeout handling."""

from unittest.mock import MagicMock, patch

import chess

from src.engines.stockfish_client import StockfishClient


def test_analyze_game_over_position():
    client = StockfishClient("/fake/stockfish")
    mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    with patch.object(client, "start"):
        result = client.analyze(mate_fen, depth=1)
    assert result.game_over
    assert result.best_move.san == "(fin)"


def test_build_top_moves_handles_none_bestmove():
    board = chess.Board()
    moves = StockfishClient._build_top_moves(board, "", {}, 1)
    assert len(moves) == 1
    assert moves[0].san == "(fin)"
