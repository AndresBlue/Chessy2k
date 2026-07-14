"""Human-like chess play over any MultiPV UCI engine client.

Strength shaping for engines without UCI_Elo (e.g. Reckless) is done entirely
in Python: opening book, MultiPV sampling, and adaptive think-time hints.
Stockfish additionally receives UCI_LimitStrength / UCI_Elo when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chess

from src.app.elo_power import stockfish_uci_elo
from src.app.errors import InvalidFEN
from src.app.logging_config import get_logger
from src.engines.human_move_selector import multipv_for_elo, select_human_move
from src.engines.humanization_config import HumanizationConfig, resolve_book_path
from src.engines.position_profiler import (
    PositionProfile,
    is_calm_opening,
    profile_position,
    recommend_engine_budget_ms,
    recommend_think_time,
)
from src.engines.stockfish_client import AnalysisResult, EngineMove, StockfishClient
from src.search.opening_book import OpeningBook

log = get_logger(__name__)


@dataclass
class HumanAnalysisResult:
    fen: str
    best_move: EngineMove
    top_moves: list[EngineMove]
    evaluation: str
    phase: str
    criticality: str
    planned_ms: int
    engine_ms: float
    from_book: bool
    target_elo: int
    think_note: str = ""
    game_over: bool = False

    @property
    def best_move_uci(self) -> str:
        return self.best_move.uci

    @property
    def best_move_san(self) -> str:
        return self.best_move.san


class HumanEngine:
    """Wrap a MultiPV UCI client with book + humanized move selection."""

    def __init__(
        self,
        engine: StockfishClient,
        config: HumanizationConfig | None = None,
        project_root: Path | None = None,
    ):
        self.engine = engine
        # Back-compat alias used by older call sites / tests.
        self.stockfish = engine
        self.config = config or HumanizationConfig()
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self._book: OpeningBook | None = None
        self._load_book()

    def _load_book(self) -> None:
        if not self.config.use_book:
            self._book = None
            return
        book_path = resolve_book_path(self.config.book_path, self.project_root)
        if book_path.exists():
            self._book = OpeningBook(str(book_path))
            log.info("Opening book loaded: %s", book_path)
        else:
            self._book = OpeningBook(None)
            log.warning("Opening book not found: %s", book_path)

    def analyze(
        self,
        fen: str,
        *,
        target_elo: int | None = None,
        on_thinking: Callable[[int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> HumanAnalysisResult:
        elo = target_elo if target_elo is not None else self.config.target_elo
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise InvalidFEN(fen) from exc

        if board.is_game_over():
            move = EngineMove(uci="", san="(fin)")
            return HumanAnalysisResult(
                fen=fen,
                best_move=move,
                top_moves=[move],
                evaluation="fin de partida",
                phase="endgame",
                criticality="low",
                planned_ms=0,
                engine_ms=0.0,
                from_book=False,
                target_elo=elo,
                think_note="Fin de partida",
                game_over=True,
            )

        book_move = self._try_book_move(board)
        if book_move is not None:
            try:
                move = chess.Move.from_uci(book_move)
                if move not in board.legal_moves:
                    log.warning("Book move illegal, skipping: %s", book_move)
                else:
                    engine_move = EngineMove(uci=book_move, san=board.san(move))
                    return HumanAnalysisResult(
                        fen=fen,
                        best_move=engine_move,
                        top_moves=[engine_move],
                        evaluation="libro",
                        phase="opening",
                        criticality="low",
                        planned_ms=2000,
                        engine_ms=0.0,
                        from_book=True,
                        target_elo=elo,
                        think_note="Apertura conocida",
                    )
            except (ValueError, chess.IllegalMoveError) as exc:
                log.warning("Book move rejected: %s (%s)", book_move, exc)

        # Cap Stockfish via UCI when available; Reckless relies on MultiPV selection.
        self.engine.configure_strength(stockfish_uci_elo(elo), limit_strength=True)

        if is_calm_opening(board, self.config.opening):
            profile = PositionProfile(
                phase="opening",
                criticality="low",
                movetime_ms=self.config.opening.movetime_ms,
                calm_opening=True,
                piece_count=len(board.piece_map()),
                legal_moves=board.legal_moves.count(),
            )
            movetime = recommend_engine_budget_ms(profile, target_elo=elo)
            analysis = self.engine.analyze(
                fen,
                movetime_ms=movetime,
                multipv=3,
                is_cancelled=is_cancelled,
            )
            if analysis.game_over:
                return self._game_over_result(fen, analysis, elo)
            profile.movetime_ms = movetime
            return self._build_result(
                fen,
                analysis,
                profile,
                elo,
                from_book=False,
                engine_ms=analysis.time_ms,
            )

        if is_cancelled and is_cancelled():
            raise RuntimeError("Analysis cancelled")

        probe_movetime = min(self.config.timing.probe_movetime_ms, 100)
        probe = self.engine.analyze(
            fen,
            movetime_ms=probe_movetime,
            multipv=3,
            is_cancelled=is_cancelled,
        )
        if probe.game_over:
            return self._game_over_result(fen, probe, elo)

        profile = profile_position(
            board,
            opening=self.config.opening,
            timing=self.config.timing,
            in_book=False,
            probe_moves=probe.top_moves,
            target_elo=elo,
        )

        engine_budget = recommend_engine_budget_ms(profile, target_elo=elo)
        final_movetime = max(80, engine_budget - probe_movetime)
        profile.movetime_ms = engine_budget
        if on_thinking is not None and final_movetime > 1500:
            on_thinking(final_movetime)

        if is_cancelled and is_cancelled():
            raise RuntimeError("Analysis cancelled")

        analysis = self.engine.analyze(
            fen,
            movetime_ms=final_movetime,
            multipv=multipv_for_elo(elo, self.config.selection),
            is_cancelled=is_cancelled,
        )
        if analysis.game_over:
            return self._game_over_result(fen, analysis, elo)

        return self._build_result(
            fen,
            analysis,
            profile,
            elo,
            from_book=False,
            engine_ms=probe.time_ms + analysis.time_ms,
        )

    def _game_over_result(
        self,
        fen: str,
        analysis: AnalysisResult,
        elo: int,
    ) -> HumanAnalysisResult:
        move = analysis.best_move
        return HumanAnalysisResult(
            fen=fen,
            best_move=move,
            top_moves=analysis.top_moves,
            evaluation=analysis.game_over_reason or "fin de partida",
            phase="endgame",
            criticality="low",
            planned_ms=0,
            engine_ms=analysis.time_ms,
            from_book=False,
            target_elo=elo,
            think_note="Fin de partida",
            game_over=True,
        )

    def _try_book_move(self, board: chess.Board) -> str | None:
        if self._book is None or not self.config.use_book:
            return None
        if board.fullmove_number > self.config.opening.max_fullmove:
            return None
        return self._book.get_move(board, temperature=self.config.book_temperature)

    def _build_result(
        self,
        fen: str,
        analysis: AnalysisResult,
        profile: PositionProfile,
        elo: int,
        *,
        from_book: bool,
        engine_ms: float,
    ) -> HumanAnalysisResult:
        chosen = select_human_move(
            analysis.top_moves,
            target_elo=elo,
            config=self.config.selection,
        )
        think = recommend_think_time(
            chess.Board(fen),
            profile,
            target_elo=elo,
            top_moves=analysis.top_moves,
            from_book=from_book,
        )
        return HumanAnalysisResult(
            fen=fen,
            best_move=chosen,
            top_moves=analysis.top_moves,
            evaluation=chosen.score_str,
            phase=profile.phase,
            criticality=profile.criticality,
            planned_ms=think.ms,
            engine_ms=engine_ms,
            from_book=from_book,
            target_elo=elo,
            think_note=think.note,
        )
