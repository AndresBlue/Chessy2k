"""Tests for Maia-3 WDL evaluation mapping."""

import chess

from src.engines.maia_eval import (
    aggregate_position_wdl,
    analyze_maia_line,
    flip_wdl_perspective,
    format_maia_evaluation,
    maia_eval_fractions,
    wdl_to_equity,
    wdl_to_equity_fraction,
)
from src.engines.stockfish_client import AnalysisResult, EngineMove


class _MockMaiaClient:
    def __init__(self, responses: list[AnalysisResult]):
        self._responses = list(responses)
        self.elo_calls: list[tuple[int, int | None]] = []

    def configure_elos(self, self_elo: int, oppo_elo: int | None = None) -> None:
        self.elo_calls.append((self_elo, oppo_elo))

    def analyze(self, *, fen: str, multipv: int = 3, is_cancelled=None):
        del fen, multipv, is_cancelled
        if not self._responses:
            raise RuntimeError("no mock responses")
        return self._responses.pop(0)


def test_wdl_to_equity_balanced():
    assert abs(wdl_to_equity(333, 334, 333) - 0.5) < 0.05
    assert abs(wdl_to_equity_fraction(333, 334, 333) - 0.5) < 0.05


def test_wdl_to_equity_winning():
    assert wdl_to_equity(700, 200, 100) > 0.65
    assert wdl_to_equity_fraction(700, 200, 100) > 0.65


def test_flip_wdl_perspective():
    assert flip_wdl_perspective((600, 250, 150)) == (150, 250, 600)


def test_aggregate_position_wdl():
    moves = [
        EngineMove("e2e4", "e4", wdl_permille=(520, 280, 200)),
        EngineMove("d2d4", "d4", wdl_permille=(480, 300, 220)),
    ]
    agg = aggregate_position_wdl(moves)
    assert agg is not None
    w, d, l = agg
    assert w + d + l == 1000
    assert w > l


def test_maia_eval_fractions():
    moves = [
        EngineMove("e2e4", "e4", wdl_permille=(520, 280, 200)),
        EngineMove("d2d4", "d4", wdl_permille=(480, 300, 220)),
    ]
    cur, post, label = maia_eval_fractions(moves)
    assert post is not None
    assert cur > 0.45
    assert post > cur - 0.1
    assert "→" in label


def test_format_maia_evaluation():
    text = format_maia_evaluation((450, 300, 250), (520, 280, 200))
    assert "→" in text
    assert "W" in text


def test_analyze_maia_line_predicts_opponent_and_evaluates_after():
    start_fen = chess.Board().fen()
    client = _MockMaiaClient(
        [
            AnalysisResult(
                best_move=EngineMove("e2e4", "e4", wdl_permille=(520, 280, 200)),
                top_moves=[EngineMove("e2e4", "e4", wdl_permille=(520, 280, 200))],
                fen=start_fen,
                time_ms=10.0,
            ),
            AnalysisResult(
                best_move=EngineMove("e7e5", "e5", wdl_permille=(480, 300, 220)),
                top_moves=[EngineMove("e7e5", "e5", wdl_permille=(480, 300, 220))],
                fen="",
                time_ms=12.0,
            ),
            AnalysisResult(
                best_move=EngineMove("g1f3", "Nf3", wdl_permille=(500, 290, 210)),
                top_moves=[
                    EngineMove("g1f3", "Nf3", wdl_permille=(500, 290, 210)),
                    EngineMove("b1c3", "Nc3", wdl_permille=(490, 295, 215)),
                ],
                fen="",
                time_ms=11.0,
            ),
        ]
    )
    line = analyze_maia_line(client, start_fen, player_elo=1500, multipv=1)
    assert line.user_move_uci == "e2e4"
    assert line.opponent_move_uci == "e7e5"
    assert line.after_fraction is not None
    assert line.after_fraction > 0.45
    assert "→" in line.label
    assert client.elo_calls[0] == (1500, 1500)
    assert client.elo_calls[1] == (1500, 1500)
    assert client.elo_calls[2] == (1500, 1500)
    assert line.total_time_ms == 33.0