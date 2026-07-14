"""Human think-time advisor for Maia-3 overlay analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from src.app.elo_power import ELO_MAX_HUMAN, ELO_MIN, is_max_effort
from src.engines.position_profiler import OpeningConfig, detect_phase, is_calm_opening
from src.engines.stockfish_client import EngineMove


@dataclass
class MaiaTimingConfig:
    enabled: bool = True
    min_seconds: float = 2.0
    max_seconds: float = 75.0
    elo_scale: bool = True
    criticality_10_point: bool = True


@dataclass
class MaiaTimingAdvice:
    ms: int
    criticality_score: int
    note: str
    phase: str = "middlegame"


def maia_timing_config_from_dict(raw: dict[str, Any] | None) -> MaiaTimingConfig:
    raw = raw or {}
    return MaiaTimingConfig(
        enabled=bool(raw.get("enabled", True)),
        min_seconds=float(raw.get("min_seconds", 2.0)),
        max_seconds=float(raw.get("max_seconds", 75.0)),
        elo_scale=bool(raw.get("elo_scale", True)),
        criticality_10_point=bool(raw.get("criticality_10_point", True)),
    )


def _elo_t(target_elo: int) -> float:
    if target_elo <= ELO_MIN:
        return 0.0
    if target_elo >= ELO_MAX_HUMAN:
        return 1.0
    return (target_elo - ELO_MIN) / max(1, ELO_MAX_HUMAN - ELO_MIN)


def _policy_entropy(top_moves: list[EngineMove]) -> float:
    """Normalized entropy over implied policy mass (rank-based proxy)."""
    if len(top_moves) <= 1:
        return 0.0
    n = len(top_moves)
    weights = [1.0 / (i + 1) for i in range(n)]
    total = sum(weights)
    probs = [w / total for w in weights]
    import math

    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(n) if n > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _wdl_swing(top_moves: list[EngineMove]) -> float:
    """How much human WDL changes between top candidates (0..1)."""
    wdls = [m.wdl_permille for m in top_moves[:3] if m.wdl_permille]
    if len(wdls) < 2:
        return 0.0
    swings: list[float] = []
    for i in range(len(wdls) - 1):
        w1, d1, l1 = wdls[i]
        w2, d2, l2 = wdls[i + 1]
        swing = (abs(w1 - w2) + abs(d1 - d2) + abs(l1 - l2)) / 1000.0
        swings.append(swing)
    return min(1.0, max(swings) if swings else 0.0)


def _tactical_pressure(board: chess.Board) -> float:
    """Simple tactical pressure score 0..1."""
    score = 0.0
    if board.is_check():
        score += 0.35
    legal = board.legal_moves.count()
    if legal <= 3:
        score += 0.35
    elif legal <= 5:
        score += 0.25
    elif legal <= 8:
        score += 0.12

    captures = 0
    promotions = 0
    for move in board.legal_moves:
        if board.is_capture(move):
            captures += 1
        if move.promotion:
            promotions += 1
    if captures >= 3:
        score += 0.15
    if promotions > 0:
        score += 0.12
    return min(1.0, score)


def _board_complexity(board: chess.Board, phase: str) -> float:
    piece_count = len(board.piece_map())
    legal = board.legal_moves.count()
    if phase == "endgame":
        if piece_count <= 6 and legal <= 10:
            return 0.1
        if piece_count <= 10:
            return 0.35
        return 0.5
    if legal >= 35:
        return 0.75
    if legal >= 25:
        return 0.55
    if piece_count >= 28:
        return 0.45
    return 0.3


def _phase_factor(board: chess.Board, phase: str, calm_opening: bool) -> float:
    if calm_opening:
        return 0.15
    if phase == "opening":
        return 0.45
    if phase == "endgame":
        return 0.4
    return 0.55


def _compute_criticality_raw(
    board: chess.Board,
    top_moves: list[EngineMove],
    *,
    phase: str,
    calm_opening: bool,
) -> float:
    """Raw criticality 0..1 before scaling to 1-10."""
    policy_unc = _policy_entropy(top_moves)
    wdl_sw = _wdl_swing(top_moves)
    tactical = _tactical_pressure(board)
    complexity = _board_complexity(board, phase)
    phase_f = _phase_factor(board, phase, calm_opening)

    raw = (
        0.28 * policy_unc
        + 0.22 * wdl_sw
        + 0.22 * tactical
        + 0.18 * complexity
        + 0.10 * phase_f
    )
    if calm_opening:
        raw *= 0.35
    return max(0.0, min(1.0, raw))


def _criticality_to_seconds(score: int, cfg: MaiaTimingConfig) -> float:
    """Map 1-10 criticality to seconds per plan bands."""
    bands = {
        1: (2, 5),
        2: (2, 5),
        3: (5, 10),
        4: (5, 10),
        5: (10, 20),
        6: (10, 20),
        7: (20, 40),
        8: (20, 40),
        9: (40, 75),
        10: (40, 75),
    }
    lo, hi = bands.get(score, (10, 20))
    t = (score - 1) / 9.0 if score > 1 else 0.0
    within = lo + t * (hi - lo)
    return max(cfg.min_seconds, min(cfg.max_seconds, within))


def _note_for_score(score: int, phase: str, calm_opening: bool) -> str:
    if calm_opening:
        return "Apertura conocida"
    if score <= 2:
        return "Jugada natural"
    if score <= 4:
        return "Decision rutinaria"
    if score <= 6:
        return "Decision fina"
    if score <= 8:
        return "Posicion critica"
    if phase == "endgame":
        return "Final con calculo"
    return "Momento decisivo"


def advise_maia_think_time(
    board: chess.Board,
    top_moves: list[EngineMove],
    *,
    target_elo: int,
    config: MaiaTimingConfig | None = None,
    opening: OpeningConfig | None = None,
) -> MaiaTimingAdvice:
    """Estimate human think time and 1-10 criticality from Maia-3 signals."""
    cfg = config or MaiaTimingConfig()
    opening = opening or OpeningConfig()

    if is_max_effort(target_elo):
        return MaiaTimingAdvice(ms=0, criticality_score=0, note="Sin simulacion humana")

    phase = detect_phase(board, opening=opening)
    calm = is_calm_opening(board, opening)

    raw = _compute_criticality_raw(board, top_moves, phase=phase, calm_opening=calm)
    score = max(1, min(10, int(round(1 + raw * 9))))

    seconds = _criticality_to_seconds(score, cfg)
    t = _elo_t(target_elo) if cfg.elo_scale else 0.5

    if score <= 3:
        seconds *= 0.75 + 0.25 * t
    elif score >= 7:
        seconds *= 0.85 + 0.35 * t
    else:
        seconds *= 0.80 + 0.30 * t

    seconds = max(cfg.min_seconds, min(cfg.max_seconds, seconds))
    note = _note_for_score(score, phase, calm)

    return MaiaTimingAdvice(
        ms=int(round(seconds * 1000)),
        criticality_score=score,
        note=note,
        phase=phase,
    )
