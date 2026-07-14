"""Pick human-practical moves from engine candidate lines."""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.app.elo_power import (
    ELO_MAX_HUMAN,
    ELO_MIN,
    STOCKFISH_UCI_ELO_MIN,
    is_max_effort,
)
from src.engines.stockfish_client import EngineMove


@dataclass
class SelectionConfig:
    multipv: int = 5
    practical_move_chance: float = 0.15
    practical_cp_slack_1900: int = 45
    practical_cp_slack_2100: int = 25
    # Elo-scaled inaccuracy knobs
    inaccuracy_chance_floor: float = 0.35
    inaccuracy_chance_ceiling: float = 0.06
    blunder_chance_floor: float = 0.08
    blunder_cp_min: int = 120


def _elo_span(elo: int, lo: int, hi: int) -> float:
    """Linear 0..1 position of *elo* between *lo* and *hi* (clamped)."""
    if hi <= lo:
        return 0.0
    if elo <= lo:
        return 0.0
    if elo >= hi:
        return 1.0
    return (elo - lo) / (hi - lo)


def practical_cp_slack(elo: int, config: SelectionConfig) -> int:
    """Interpolate centipawn slack across the full human Elo range."""
    if is_max_effort(elo):
        return 0
    if elo >= ELO_MAX_HUMAN:
        return config.practical_cp_slack_2100
    if elo >= 2100:
        t = _elo_span(elo, 2100, ELO_MAX_HUMAN)
        return int(round(config.practical_cp_slack_2100 + t * 0))
    if elo >= 1900:
        t = _elo_span(elo, 1900, 2100)
        return int(
            round(
                config.practical_cp_slack_1900
                + t * (config.practical_cp_slack_2100 - config.practical_cp_slack_1900)
            )
        )
    if elo >= STOCKFISH_UCI_ELO_MIN:
        t = _elo_span(elo, STOCKFISH_UCI_ELO_MIN, 1900)
        low = 100
        return int(round(low + t * (config.practical_cp_slack_1900 - low)))
    # Below Stockfish's UCI floor: accept much worse-looking moves.
    t = _elo_span(elo, ELO_MIN, STOCKFISH_UCI_ELO_MIN)
    slack_at_min = 260
    slack_at_sf_min = 100
    return int(round(slack_at_min + t * (slack_at_sf_min - slack_at_min)))


def _score_to_cp(move: EngineMove) -> int:
    if move.score_mate is not None:
        sign = 1 if move.score_mate > 0 else -1
        return sign * (100_000 - abs(move.score_mate))
    return move.score_cp or 0


def inaccuracy_chance(elo: int, config: SelectionConfig) -> float:
    if is_max_effort(elo) or elo >= ELO_MAX_HUMAN:
        return config.inaccuracy_chance_ceiling
    if elo <= ELO_MIN:
        return 0.58
    if elo <= STOCKFISH_UCI_ELO_MIN:
        t = _elo_span(elo, ELO_MIN, STOCKFISH_UCI_ELO_MIN)
        return 0.58 + t * (0.38 - 0.58)
    t = _elo_span(elo, STOCKFISH_UCI_ELO_MIN, ELO_MAX_HUMAN)
    return 0.38 + t * (config.inaccuracy_chance_ceiling - 0.38)


def blunder_chance(elo: int, config: SelectionConfig) -> float:
    if is_max_effort(elo) or elo >= ELO_MAX_HUMAN - 200:
        return 0.0
    if elo <= ELO_MIN:
        return 0.32
    if elo <= STOCKFISH_UCI_ELO_MIN:
        t = _elo_span(elo, ELO_MIN, STOCKFISH_UCI_ELO_MIN)
        return 0.32 + t * (0.14 - 0.32)
    t = _elo_span(elo, STOCKFISH_UCI_ELO_MIN, ELO_MAX_HUMAN - 200)
    return 0.14 + t * (config.blunder_chance_floor - 0.14)


def blunder_cp_threshold(elo: int, config: SelectionConfig) -> int:
    """How many cp worse a line must be to count as a blunder candidate."""
    if is_max_effort(elo):
        return config.blunder_cp_min
    if elo <= STOCKFISH_UCI_ELO_MIN:
        t = _elo_span(elo, ELO_MIN, STOCKFISH_UCI_ELO_MIN)
        return int(round(70 + t * (config.blunder_cp_min - 70)))
    return config.blunder_cp_min


def multipv_for_elo(elo: int, config: SelectionConfig) -> int:
    if is_max_effort(elo):
        return 1
    if elo <= 800:
        return max(config.multipv, 8)
    if elo <= STOCKFISH_UCI_ELO_MIN:
        return max(config.multipv, 7)
    if elo <= 1800:
        return max(config.multipv, 6)
    if elo >= ELO_MAX_HUMAN - 100:
        return max(3, config.multipv - 2)
    return config.multipv


def select_human_move(
    top_moves: list[EngineMove],
    *,
    target_elo: int,
    config: SelectionConfig | None = None,
    rng: random.Random | None = None,
) -> EngineMove:
    """
    Choose a move from MultiPV lines.

    Usually returns the engine best line; occasionally picks a close alternative
    within human-practical centipawn slack, with more mistakes at lower Elo.
    """
    if not top_moves:
        return EngineMove(uci="", san="(sin jugadas)")

    config = config or SelectionConfig()
    rng = rng or random.Random()

    best = top_moves[0]
    if not best.uci or is_max_effort(target_elo):
        return best

    slack = practical_cp_slack(target_elo, config)
    best_cp = _score_to_cp(best)
    blunder_min = blunder_cp_threshold(target_elo, config)

    alternatives: list[EngineMove] = []
    blunders: list[EngineMove] = []
    for move in top_moves[1:]:
        if not move.uci:
            continue
        gap = best_cp - _score_to_cp(move)
        if gap <= slack:
            alternatives.append(move)
        elif gap >= blunder_min:
            blunders.append(move)

    roll = rng.random()
    b_chance = blunder_chance(target_elo, config)
    if blunders and roll < b_chance:
        return rng.choice(blunders)

    i_chance = inaccuracy_chance(target_elo, config)
    if alternatives and roll < i_chance + b_chance:
        pool = alternatives
        weights = [max(1, slack - (best_cp - _score_to_cp(m))) for m in pool]
        return rng.choices(pool, weights=weights, k=1)[0]

    if not alternatives:
        return best

    chance = config.practical_move_chance
    if target_elo <= STOCKFISH_UCI_ELO_MIN:
        chance = min(0.45, chance * 2.0)
    elif target_elo <= 1900:
        chance = min(0.30, chance * 1.4)
    elif target_elo >= 2100:
        chance = max(0.06, chance * 0.7)

    if rng.random() >= chance:
        return best

    pool = [best] + alternatives
    weights = [3] + [1] * len(alternatives)
    return rng.choices(pool, weights=weights, k=1)[0]
