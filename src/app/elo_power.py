"""Elo power slider constants and persistence helpers."""

from __future__ import annotations

# User-facing minimum on the power slider (beginner / Chess.com ~400-800).
ELO_MIN = 600
# Stockfish LimitStrength only accepts UCI_Elo in this range.
STOCKFISH_UCI_ELO_MIN = 1320
STOCKFISH_UCI_ELO_MAX = 3190
ELO_MAX_HUMAN = 3000
MAX_EFFORT_ELO = 0  # sentinel: full-strength Stockfish, no humanization
SLIDER_STEPS = 100


def is_max_effort(elo: int) -> bool:
    return elo == MAX_EFFORT_ELO


def stockfish_uci_elo(requested: int) -> int:
    """Map the requested human Elo to a value Stockfish LimitStrength accepts."""
    if is_max_effort(requested):
        return STOCKFISH_UCI_ELO_MAX
    return max(STOCKFISH_UCI_ELO_MIN, min(STOCKFISH_UCI_ELO_MAX, requested))


def is_below_stockfish_floor(elo: int) -> bool:
    """True when we must weaken further on top of Stockfish's minimum strength."""
    return not is_max_effort(elo) and elo < STOCKFISH_UCI_ELO_MIN


def elo_to_slider(elo: int) -> float:
    """Map stored Elo to slider position 0..100."""
    if is_max_effort(elo):
        return float(SLIDER_STEPS)
    clamped = max(ELO_MIN, min(ELO_MAX_HUMAN, elo))
    if ELO_MAX_HUMAN == ELO_MIN:
        return 0.0
    return (clamped - ELO_MIN) / (ELO_MAX_HUMAN - ELO_MIN) * (SLIDER_STEPS - 1)


def slider_to_elo(value: float) -> int:
    """Map slider position to stored Elo (0 = max effort)."""
    if value >= SLIDER_STEPS - 0.5:
        return MAX_EFFORT_ELO
    if value <= 0:
        return ELO_MIN
    t = value / (SLIDER_STEPS - 1)
    return int(round(ELO_MIN + t * (ELO_MAX_HUMAN - ELO_MIN)))


def format_elo_label(elo: int) -> str:
    if is_max_effort(elo):
        return "Maximo esfuerzo"
    return f"~{elo} Elo"
