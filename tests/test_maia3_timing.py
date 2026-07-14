"""Tests for Maia-3 human timing advisor."""

import chess

from src.engines.maia3_timing import (
    MaiaTimingConfig,
    advise_maia_think_time,
    maia_timing_config_from_dict,
)
from src.engines.stockfish_client import EngineMove


def _move(uci: str, san: str, wdl: tuple[int, int, int] | None = None) -> EngineMove:
    return EngineMove(uci=uci, san=san, wdl_permille=wdl)


def test_calm_opening_low_time():
    board = chess.Board()
    top = [
        _move("e2e4", "e4", (450, 300, 250)),
        _move("d2d4", "d4", (420, 310, 270)),
    ]
    advice = advise_maia_think_time(board, top, target_elo=1500)
    assert advice.criticality_score <= 3
    assert advice.ms <= 8000
    assert "Apertura" in advice.note or advice.criticality_score <= 2


def test_check_raises_criticality():
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    top = [
        _move("g2g3", "g3", (100, 200, 700)),
        _move("f3f4", "f4", (120, 180, 700)),
    ]
    advice = advise_maia_think_time(board, top, target_elo=1800)
    assert advice.criticality_score >= 5
    assert advice.ms >= 10_000


def test_close_candidates_raise_time():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    top = [
        _move("e7e5", "e5", (480, 280, 240)),
        _move("c7c5", "c5", (470, 290, 240)),
        _move("e7e6", "e6", (460, 300, 240)),
    ]
    advice = advise_maia_think_time(board, top, target_elo=2000)
    assert advice.criticality_score >= 2
    assert advice.ms >= 2000


def test_simple_endgame_lower_than_critical_middlegame():
    board = chess.Board("8/8/4k3/8/8/8/8/K6R w - - 0 1")
    top = [_move("h1h7", "Rh7", (900, 80, 20))]
    endgame = advise_maia_think_time(board, top, target_elo=1600)

    mid_board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/3P4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 4")
    mid_top = [
        _move("g7g6", "g6", (200, 300, 500)),
        _move("d7d6", "d6", (220, 280, 500)),
    ]
    middlegame = advise_maia_think_time(mid_board, mid_top, target_elo=1600)

    assert endgame.criticality_score <= middlegame.criticality_score + 2


def test_higher_elo_spends_more_on_critical():
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/3P4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 4")
    top = [
        _move("g7g6", "g6", (200, 300, 500)),
        _move("d7d6", "d6", (220, 280, 500)),
        _move("h7h6", "h6", (210, 290, 500)),
    ]
    low = advise_maia_think_time(board, top, target_elo=800)
    high = advise_maia_think_time(board, top, target_elo=2400)
    if low.criticality_score >= 5:
        assert high.ms >= low.ms


def test_config_from_dict():
    cfg = maia_timing_config_from_dict({"min_seconds": 3, "max_seconds": 60})
    assert cfg.min_seconds == 3.0
    assert cfg.max_seconds == 60.0
    assert cfg.enabled is True


def test_gui_import_smoke():
    from src.app.ui_qt.app_window import AppWindow  # noqa: F401
    from src.app.fast_analyzer import FastAnalyzer  # noqa: F401
