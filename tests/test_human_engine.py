"""Tests for humanized engine components."""

from __future__ import annotations

import chess
import pytest

from src.app.elo_power import ELO_MAX_HUMAN, MAX_EFFORT_ELO
from src.engines.human_move_selector import (
    SelectionConfig,
    blunder_chance,
    inaccuracy_chance,
    practical_cp_slack,
    select_human_move,
)
from src.engines.position_profiler import (
    OpeningConfig,
    TimingConfig,
    detect_phase,
    is_calm_opening,
    profile_position,
    recommend_engine_budget_ms,
    recommend_think_time,
)
from src.engines.stockfish_client import EngineMove


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLEGAME_FEN = "r1bq1rk1/ppp2ppp/2n1bn2/3pp3/2PP4/2N1BN2/PP3PPP/R1BQ1RK1 w - - 0 15"


def test_opening_profile_low_criticality():
    board = chess.Board(START_FEN)
    profile = profile_position(
        board,
        opening=OpeningConfig(max_fullmove=12, movetime_ms=120),
        timing=TimingConfig(),
        target_elo=ELO_MAX_HUMAN,
    )
    assert profile.phase == "opening"
    assert profile.criticality == "low"
    assert profile.movetime_ms == 120
    assert profile.calm_opening


def test_developed_position_still_opening():
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4"):
        board.push(board.parse_san(san))
    assert detect_phase(board, opening=OpeningConfig()) == "opening"
    assert is_calm_opening(board, opening=OpeningConfig())


def test_opening_ignores_close_probe_scores():
    board = chess.Board(START_FEN)
    probe = [
        EngineMove(uci="e2e4", san="e4", score_cp=30),
        EngineMove(uci="d2d4", san="d4", score_cp=20),
    ]
    profile = profile_position(
        board,
        timing=TimingConfig(critical_cp_gap=25),
        probe_moves=probe,
        target_elo=ELO_MAX_HUMAN,
    )
    assert profile.phase == "opening"
    assert profile.criticality == "low"
    assert profile.movetime_ms == 120


def test_close_scores_raise_criticality_in_middlegame():
    board = chess.Board(MIDDLEGAME_FEN)
    probe = [
        EngineMove(uci="d4d5", san="d5", score_cp=30),
        EngineMove(uci="c4d5", san="cxd5", score_cp=10),
    ]
    profile = profile_position(
        board,
        timing=TimingConfig(critical_cp_gap=25),
        probe_moves=probe,
        target_elo=ELO_MAX_HUMAN,
    )
    assert profile.phase == "middlegame"
    assert profile.criticality == "high"
    assert profile.movetime_ms >= 9500


def test_gambit_opening_not_calm():
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 2"
    board = chess.Board(fen)
    assert detect_phase(board) == "opening"
    assert not is_calm_opening(board)


def test_practical_cp_slack_interpolation():
    config = SelectionConfig(practical_cp_slack_1900=45, practical_cp_slack_2100=25)
    from src.app.elo_power import ELO_MIN, ELO_MAX_HUMAN, MAX_EFFORT_ELO, STOCKFISH_UCI_ELO_MIN

    assert practical_cp_slack(ELO_MIN, config) == 260
    assert practical_cp_slack(STOCKFISH_UCI_ELO_MIN, config) == 100
    assert practical_cp_slack(1900, config) == 45
    assert practical_cp_slack(ELO_MAX_HUMAN, config) == 25
    assert practical_cp_slack(MAX_EFFORT_ELO, config) == 0
    assert practical_cp_slack(2000, config) == 35


def test_sub_stockfish_elo_weaker_than_floor():
    config = SelectionConfig()
    from src.app.elo_power import ELO_MIN, STOCKFISH_UCI_ELO_MIN

    assert blunder_chance(ELO_MIN, config) > blunder_chance(STOCKFISH_UCI_ELO_MIN, config)
    assert inaccuracy_chance(ELO_MIN, config) > inaccuracy_chance(STOCKFISH_UCI_ELO_MIN, config)
    assert practical_cp_slack(ELO_MIN, config) > practical_cp_slack(STOCKFISH_UCI_ELO_MIN, config)


def test_inaccuracy_decreases_with_elo():
    config = SelectionConfig()
    from src.app.elo_power import ELO_MIN, ELO_MAX_HUMAN

    assert inaccuracy_chance(ELO_MIN, config) > inaccuracy_chance(ELO_MAX_HUMAN, config)


def test_blunder_chance_higher_at_low_elo():
    config = SelectionConfig()
    from src.app.elo_power import ELO_MIN, ELO_MAX_HUMAN

    assert blunder_chance(ELO_MIN, config) > blunder_chance(ELO_MAX_HUMAN, config)


def test_select_human_move_prefers_best_by_default():
    top = [
        EngineMove(uci="e2e4", san="e4", score_cp=50),
        EngineMove(uci="d2d4", san="d4", score_cp=40),
    ]
    import random

    rng = random.Random(0)
    for _ in range(20):
        move = select_human_move(
            top,
            target_elo=MAX_EFFORT_ELO,
            config=SelectionConfig(),
            rng=rng,
        )
        assert move.uci == "e2e4"


def test_select_human_move_can_pick_alternative():
    top = [
        EngineMove(uci="e2e4", san="e4", score_cp=50),
        EngineMove(uci="d2d4", san="d4", score_cp=30),
    ]
    import random

    rng = random.Random(42)
    picks = {select_human_move(top, target_elo=1900, rng=rng).uci for _ in range(50)}
    assert "d2d4" in picks


def test_medium_middlegame_uses_normal_think_time():
    board = chess.Board(MIDDLEGAME_FEN)
    profile = profile_position(
        board,
        timing=TimingConfig(normal_movetime_ms=3500, critical_movetime_ms=9500),
        probe_moves=None,
        target_elo=ELO_MAX_HUMAN,
    )
    assert profile.phase == "middlegame"
    assert profile.criticality == "medium"
    assert profile.movetime_ms == 3500


def test_real_engine_budget_is_fast_and_capped():
    board = chess.Board(MIDDLEGAME_FEN)
    profile = profile_position(
        board,
        timing=TimingConfig(normal_movetime_ms=3500, critical_movetime_ms=9500),
        probe_moves=None,
        target_elo=2000,
    )
    budget = recommend_engine_budget_ms(profile, target_elo=2000)
    assert 180 <= budget <= 1000
    assert budget <= 500


def test_high_criticality_engine_budget_never_exceeds_one_second():
    board = chess.Board(MIDDLEGAME_FEN)
    probe = [
        EngineMove(uci="d4d5", san="d5", score_cp=30),
        EngineMove(uci="c4d5", san="cxd5", score_cp=10),
    ]
    profile = profile_position(
        board,
        timing=TimingConfig(critical_cp_gap=25),
        probe_moves=probe,
        target_elo=ELO_MAX_HUMAN,
    )
    assert profile.criticality == "high"
    assert recommend_engine_budget_ms(profile, target_elo=ELO_MAX_HUMAN) <= 1000


def test_endgame_phase_detection():
    fen = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"
    board = chess.Board(fen)
    profile = profile_position(board, opening=OpeningConfig(max_fullmove=12))
    assert profile.phase == "endgame"


def test_simple_endgame_suggests_quick_human_time():
    board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    profile = profile_position(board, opening=OpeningConfig(max_fullmove=12), target_elo=1900)
    advice = recommend_think_time(board, profile, target_elo=1900)
    assert profile.phase == "endgame"
    assert advice.ms <= 8000
    assert "Final" in advice.note


def test_mating_or_conversion_endgame_suggests_longer_time_for_strong_player():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1")
    profile = profile_position(board, opening=OpeningConfig(max_fullmove=12), target_elo=2100)
    advice = recommend_think_time(
        board,
        profile,
        target_elo=2100,
        top_moves=[
            EngineMove(uci="f1c1", san="Rc1", score_mate=5),
            EngineMove(uci="f1d1", san="Rd1", score_cp=200),
        ],
    )
    assert advice.ms >= 25000
    assert "mate" in advice.note.lower() or "conversion" in advice.note.lower()


@pytest.mark.skipif(True, reason="Requires Stockfish binary and opening book")
def test_human_engine_integration():
    from pathlib import Path

    from src.engines.human_engine import HumanEngine
    from src.engines.humanization_config import HumanizationConfig
    from src.engines.stockfish_client import StockfishClient
    from src.app.runtime_paths import resolve_runtime_paths

    paths = resolve_runtime_paths()
    client = StockfishClient(paths["stockfish"])
    client.start()
    engine = HumanEngine(client, HumanizationConfig(), project_root=paths["root"])
    result = engine.analyze(START_FEN, target_elo=2000)
    assert result.best_move_uci
    client.stop()
