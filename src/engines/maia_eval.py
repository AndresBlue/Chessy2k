"""Map Maia-3 human WDL outputs to overlay evaluation display."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from src.engines.stockfish_client import AnalysisResult, EngineMove


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def wdl_to_equity(w: int, d: int, l: int) -> float:
    """Human win expectancy in [0, 1] from WDL permille (side to move)."""
    total = max(1, w + d + l)
    win = w / total
    draw = d / total
    return win + 0.5 * draw


def wdl_to_equity_fraction(w: int, d: int, l: int) -> float:
    """Map human WDL to 0..1 bar fraction (0.5 = equal)."""
    return _clamp01(wdl_to_equity(w, d, l))


def flip_wdl_perspective(wdl: tuple[int, int, int]) -> tuple[int, int, int]:
    """Swap win/loss — WDL from the other side's point of view."""
    w, d, l = wdl
    return _normalize_permille(l, d, w)


def aggregate_position_wdl(top_moves: list[EngineMove]) -> tuple[int, int, int] | None:
    """Expected human WDL from current position (policy-weighted over MultiPV).

    Maia-3 emits post-move WDL per candidate. Weighting by rank approximates the
    model's move distribution and yields a position-level human outlook.
    """
    wdls = [m.wdl_permille for m in top_moves if m.wdl_permille]
    if not wdls:
        return None
    weights = [1.0 / (i + 1) for i in range(len(wdls))]
    total_w = sum(weights)
    w = int(round(sum(w * weights[i] for i, (w, _d, _l) in enumerate(wdls)) / total_w))
    d = int(round(sum(d * weights[i] for i, (_w, d, _l) in enumerate(wdls)) / total_w))
    l = int(round(sum(l * weights[i] for i, (_w, _d, l) in enumerate(wdls)) / total_w))
    return _normalize_permille(w, d, l)


def _normalize_permille(w: int, d: int, l: int) -> tuple[int, int, int]:
    w = max(0, w)
    d = max(0, d)
    l = max(0, l)
    total = w + d + l
    if total == 0:
        return 333, 334, 333
    if total == 1000:
        return w, d, l
    scale = 1000.0 / total
    wi = int(w * scale)
    di = int(d * scale)
    li = int(l * scale)
    remainder = 1000 - wi - di - li
    if remainder > 0:
        wi += remainder
    return wi, di, li


def format_wdl_permille(wdl: tuple[int, int, int] | None) -> str:
    if not wdl:
        return "—"
    w, d, l = wdl
    return f"W{w / 10:.0f}% D{d / 10:.0f}% L{l / 10:.0f}%"


def format_maia_evaluation(
    current: tuple[int, int, int] | None,
    post: tuple[int, int, int] | None,
) -> str:
    if current and post:
        return f"{format_wdl_permille(current)} → {format_wdl_permille(post)}"
    if post:
        return format_wdl_permille(post)
    if current:
        return format_wdl_permille(current)
    return "—"


def maia_eval_fractions(top_moves: list[EngineMove]) -> tuple[float, float | None, str]:
    """Return (current_bar, post_bar, label) for overlay eval UI."""
    post_wdl = top_moves[0].wdl_permille if top_moves else None
    current_wdl = aggregate_position_wdl(top_moves)
    current_frac = (
        wdl_to_equity_fraction(*current_wdl) if current_wdl else 0.5
    )
    post_frac = (
        wdl_to_equity_fraction(*post_wdl) if post_wdl else None
    )
    label = format_maia_evaluation(current_wdl, post_wdl)
    return current_frac, post_frac, label


@dataclass
class MaiaLineAnalysis:
    user_move_uci: str
    user_move_san: str
    opponent_move_uci: str | None
    opponent_move_san: str | None
    top_moves: list[EngineMove]
    current_wdl: tuple[int, int, int] | None
    after_line_wdl: tuple[int, int, int] | None
    current_fraction: float
    after_fraction: float | None
    label: str
    total_time_ms: float


def _player_perspective_wdl(
    wdl: tuple[int, int, int] | None,
    *,
    side_to_move: chess.Color,
    player_color: chess.Color,
) -> tuple[int, int, int] | None:
    if wdl is None:
        return None
    if side_to_move == player_color:
        return wdl
    return flip_wdl_perspective(wdl)


def analyze_maia_line(
    client,
    fen: str,
    *,
    player_elo: int,
    opponent_elo: int | None = None,
    multipv: int = 3,
    is_cancelled=None,
) -> MaiaLineAnalysis:
    """Predict user move, opponent reply, then evaluate after both."""
    board = chess.Board(fen)
    player_color = board.turn
    elo = max(0, int(player_elo or 1500))
    oppo = max(0, int(opponent_elo if opponent_elo is not None else elo))
    total_ms = 0.0

    client.configure_elos(elo, oppo)
    user_analysis: AnalysisResult = client.analyze(
        fen=fen,
        multipv=multipv,
        is_cancelled=is_cancelled,
    )
    total_ms += user_analysis.time_ms
    user_move = user_analysis.best_move
    current_wdl = aggregate_position_wdl(user_analysis.top_moves)
    current_frac = wdl_to_equity_fraction(*current_wdl) if current_wdl else 0.5

    opponent_uci: str | None = None
    opponent_san: str | None = None
    after_wdl: tuple[int, int, int] | None = None
    after_frac: float | None = None

    if not user_analysis.game_over:
        user_chess_move = chess.Move.from_uci(user_move.uci)
        if user_chess_move in board.legal_moves:
            board.push(user_chess_move)
            if not board.is_game_over():
                client.configure_elos(oppo, elo)
                oppo_analysis: AnalysisResult = client.analyze(
                    fen=board.fen(),
                    multipv=multipv,
                    is_cancelled=is_cancelled,
                )
                total_ms += oppo_analysis.time_ms
                oppo_move = oppo_analysis.best_move
                if not oppo_analysis.game_over:
                    oppo_chess_move = chess.Move.from_uci(oppo_move.uci)
                    if oppo_chess_move in board.legal_moves:
                        opponent_uci = oppo_move.uci
                        opponent_san = oppo_move.san
                        board.push(oppo_chess_move)
                        if not board.is_game_over():
                            client.configure_elos(elo, oppo)
                            final_analysis: AnalysisResult = client.analyze(
                                fen=board.fen(),
                                multipv=multipv,
                                is_cancelled=is_cancelled,
                            )
                            total_ms += final_analysis.time_ms
                            raw_after = aggregate_position_wdl(final_analysis.top_moves)
                            after_wdl = _player_perspective_wdl(
                                raw_after,
                                side_to_move=board.turn,
                                player_color=player_color,
                            )
                            if after_wdl:
                                after_frac = wdl_to_equity_fraction(*after_wdl)

    label = format_maia_evaluation(current_wdl, after_wdl)
    return MaiaLineAnalysis(
        user_move_uci=user_move.uci,
        user_move_san=user_move.san,
        opponent_move_uci=opponent_uci,
        opponent_move_san=opponent_san,
        top_moves=user_analysis.top_moves,
        current_wdl=current_wdl,
        after_line_wdl=after_wdl,
        current_fraction=current_frac,
        after_fraction=after_frac,
        label=label,
        total_time_ms=total_ms,
    )
