"""Classify chess positions for human-like think time and difficulty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import chess

from src.app.elo_power import ELO_MAX_HUMAN, ELO_MIN, is_max_effort
from src.engines.stockfish_client import EngineMove

Phase = Literal["opening", "middlegame", "endgame"]
Criticality = Literal["low", "medium", "high"]


@dataclass
class TimingConfig:
    probe_movetime_ms: int = 200
    normal_movetime_ms: int = 3500
    critical_movetime_ms: int = 9500
    critical_cp_gap: int = 25


@dataclass
class OpeningConfig:
    max_fullmove: int = 12
    movetime_ms: int = 120
    min_pieces: int = 24


@dataclass
class PositionProfile:
    phase: Phase
    criticality: Criticality
    movetime_ms: int
    in_book: bool = False
    calm_opening: bool = False
    piece_count: int = 32
    legal_moves: int = 0


@dataclass
class ThinkTimeAdvice:
    """Suggested human think time shown to the user, separate from engine time."""

    ms: int
    note: str


ENGINE_MIN_MS = 180
ENGINE_AVG_MS = 250
ENGINE_MAX_MS = 1000


def _score_to_cp(move: EngineMove) -> int:
    if move.score_mate is not None:
        sign = 1 if move.score_mate > 0 else -1
        return sign * (100_000 - abs(move.score_mate))
    return move.score_cp or 0


def detect_phase(
    board: chess.Board,
    *,
    opening: OpeningConfig | None = None,
) -> Phase:
    """Game phase from move number and material (not piece placement on back rank)."""
    opening = opening or OpeningConfig()
    piece_count = len(board.piece_map())
    if piece_count <= 12:
        return "endgame"
    if (
        board.fullmove_number <= opening.max_fullmove
        and piece_count >= opening.min_pieces
    ):
        return "opening"
    return "middlegame"


def is_calm_opening(board: chess.Board, opening: OpeningConfig | None = None) -> bool:
    """
    Mainline opening: respond quickly like an experienced player.

    False when the position is sharp or the opponent has broken normal development
    (checks, heavy trades, queen off, very few options).
    """
    opening = opening or OpeningConfig()
    if detect_phase(board, opening=opening) != "opening":
        return False
    if board.is_check():
        return False

    piece_count = len(board.piece_map())
    if piece_count < opening.min_pieces:
        return False

    if board.legal_moves.count() <= 6:
        return False

    if board.ep_square is not None and board.fullmove_number <= 8:
        return False

    if board.fullmove_number <= 10:
        white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
        if white_queens == 0 or black_queens == 0:
            return False

    return True


def scale_movetime_for_elo(movetime_ms: int, target_elo: int) -> int:
    """Lower Elo thinks faster; high Elo uses the full budget."""
    if movetime_ms <= 0 or is_max_effort(target_elo):
        return movetime_ms
    if target_elo <= ELO_MIN:
        return max(80, int(movetime_ms * 0.45))
    if target_elo >= ELO_MAX_HUMAN:
        return movetime_ms
    t = (target_elo - ELO_MIN) / max(1, ELO_MAX_HUMAN - ELO_MIN)
    factor = 0.45 + t * 0.55
    return max(80, int(movetime_ms * factor))


def recommend_think_time(
    board: chess.Board,
    profile: PositionProfile,
    *,
    target_elo: int,
    top_moves: list[EngineMove] | None = None,
    from_book: bool = False,
) -> ThinkTimeAdvice:
    """Estimate how long the user should appear to think before playing.

    This is intentionally not the Stockfish search budget. It models human rhythm:
    known openings are fast, quiet technical endgames are often intuitive, and
    forcing mate/draw conversion can justify a longer pause.
    """
    if is_max_effort(target_elo):
        return ThinkTimeAdvice(ms=0, note="Sin simulacion humana")
    if from_book or profile.in_book or profile.calm_opening:
        return ThinkTimeAdvice(ms=_seconds_ms(2 + 2 * _elo_t(target_elo)), note="Apertura conocida")

    t = _elo_t(target_elo)
    piece_count = profile.piece_count
    legal_moves = profile.legal_moves
    mate_seen = any(m.score_mate is not None for m in (top_moves or [])[:3])
    score_gap = _top_score_gap(top_moves)

    if profile.phase == "endgame":
        if mate_seen:
            seconds = 16 + 28 * t
            note = "Final: mate/conversion"
        elif piece_count <= 6 and legal_moves <= 10:
            seconds = 4 + 3 * t
            note = "Final simple"
        elif piece_count <= 10:
            seconds = 6 + 8 * t
            note = "Final tecnico"
        else:
            seconds = 8 + 10 * t
            note = "Final con calculo"
    elif profile.phase == "opening":
        seconds = 5 + 8 * t
        note = "Apertura fuera de teoria"
    else:
        if profile.criticality == "high" or mate_seen:
            seconds = 16 + 24 * t
            note = "Posicion critica"
        elif profile.criticality == "medium":
            seconds = 8 + 14 * t
            note = "Medio juego"
        else:
            seconds = 5 + 8 * t
            note = "Jugada natural"

    if board.is_check():
        seconds += 6 + 6 * t
        note = "En jaque"
    if legal_moves <= 5:
        seconds += 4 + 8 * t
        note = "Pocas defensas"
    elif legal_moves >= 35 and profile.phase != "endgame":
        seconds += 3 + 5 * t
        note = "Muchas opciones"
    if score_gap is not None and score_gap <= 25 and profile.phase != "opening":
        seconds += 6 + 12 * t
        note = "Decision fina"

    # Keep the suggestion believable: quick enough for normal play, long enough
    # to communicate a real pause in critical positions.
    clamped = max(2, min(65, round(seconds)))
    return ThinkTimeAdvice(ms=clamped * 1000, note=note)


def recommend_engine_budget_ms(
    profile: PositionProfile,
    *,
    target_elo: int,
) -> int:
    """Fast real engine budget, independent from the visual human think timer."""
    if is_max_effort(target_elo):
        return ENGINE_MAX_MS
    t = _elo_t(target_elo)

    if profile.calm_opening:
        base = 160
    elif profile.phase == "endgame":
        if profile.criticality == "high":
            base = 420
        elif profile.piece_count <= 8:
            base = 210
        else:
            base = 300
    elif profile.criticality == "high":
        base = 520
    elif profile.criticality == "medium":
        base = ENGINE_AVG_MS
    else:
        base = 200

    if profile.legal_moves >= 35 and profile.phase != "endgame":
        base += 90
    if profile.legal_moves <= 5:
        base += 120

    # Higher Elo gets a little more search, but remains capped at 1 second.
    budget = int(base * (0.85 + 0.35 * t))
    return max(ENGINE_MIN_MS, min(ENGINE_MAX_MS, budget))


def _seconds_ms(seconds: float) -> int:
    return int(round(seconds)) * 1000


def _elo_t(target_elo: int) -> float:
    if target_elo <= ELO_MIN:
        return 0.0
    if target_elo >= ELO_MAX_HUMAN:
        return 1.0
    return (target_elo - ELO_MIN) / max(1, ELO_MAX_HUMAN - ELO_MIN)


def _top_score_gap(top_moves: list[EngineMove] | None) -> int | None:
    if not top_moves or len(top_moves) < 2:
        return None
    return abs(_score_to_cp(top_moves[0]) - _score_to_cp(top_moves[1]))


def profile_position(
    board: chess.Board,
    *,
    opening: OpeningConfig | None = None,
    timing: TimingConfig | None = None,
    in_book: bool = False,
    probe_moves: list[EngineMove] | None = None,
    target_elo: int = 2000,
) -> PositionProfile:
    """Estimate game phase, criticality, and search budget."""
    opening = opening or OpeningConfig()
    timing = timing or TimingConfig()

    piece_count = len(board.piece_map())
    legal_moves = board.legal_moves.count()
    phase = detect_phase(board, opening=opening)
    calm = is_calm_opening(board, opening)
    criticality = _base_criticality(board, piece_count, legal_moves, phase, calm)

    # Close MultiPV scores are normal in openings (e4 vs d4); only sharpen outside calm theory.
    if probe_moves and len(probe_moves) >= 2 and not calm:
        gap = abs(_score_to_cp(probe_moves[0]) - _score_to_cp(probe_moves[1]))
        if gap <= timing.critical_cp_gap:
            criticality = "high"
        elif gap <= timing.critical_cp_gap * 2 and criticality != "high":
            criticality = "medium"

    if probe_moves and not calm:
        for move in probe_moves[:3]:
            if move.score_mate is not None:
                criticality = "high"
                break

    movetime_ms = _movetime_for(phase, criticality, calm, in_book, opening, timing)
    movetime_ms = scale_movetime_for_elo(movetime_ms, target_elo)
    return PositionProfile(
        phase=phase,
        criticality=criticality,
        movetime_ms=movetime_ms,
        in_book=in_book,
        calm_opening=calm,
        piece_count=piece_count,
        legal_moves=legal_moves,
    )


def _base_criticality(
    board: chess.Board,
    piece_count: int,
    legal_moves: int,
    phase: Phase,
    calm_opening: bool,
) -> Criticality:
    if calm_opening:
        return "low"
    if board.is_check():
        return "high"
    if legal_moves <= 5:
        return "high"
    if phase == "opening":
        return "medium"
    if phase == "endgame" and piece_count <= 8:
        return "medium"
    return "medium"


def _movetime_for(
    phase: Phase,
    criticality: Criticality,
    calm_opening: bool,
    in_book: bool,
    opening: OpeningConfig,
    timing: TimingConfig,
) -> int:
    if in_book:
        return 0
    if calm_opening:
        return opening.movetime_ms
    if phase == "opening":
        if criticality == "high":
            return timing.critical_movetime_ms
        if criticality == "medium":
            return timing.normal_movetime_ms
        return opening.movetime_ms
    if criticality == "high":
        return timing.critical_movetime_ms
    return timing.normal_movetime_ms
