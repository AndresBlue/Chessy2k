"""Game state tracker: reconcile vision output with legal chess state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import chess

from src.app.logging_config import get_logger
from src.chess_core.fen_utils import (
    board_matrix_from_board,
    fen_from_matrix,
    matrix_from_screen_view,
    matrix_to_placement,
    default_castling_for_placement,
)
from src.chess_core.legal_validator import validate_placement_sanity
from src.chess_core.zobrist import zobrist_hash

log = get_logger(__name__)

Orientation = Literal["white", "black"]


@dataclass
class TrackerConfig:
    infer_castling: bool = True
    infer_en_passant: bool = False
    max_search_ply: int = 4
    transition_max_nodes: int = 5_000
    transition_max_seconds: float = 0.4


@dataclass
class TrackerResult:
    fen: str
    placement: str
    turn: str
    castling: str
    en_passant: str
    halfmove: int
    fullmove: int
    status: str  # "ok", "ambiguous", "vision_error", "initial"
    candidate_moves: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    zobrist: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "placement": self.placement,
            "turn": self.turn,
            "castling": self.castling,
            "en_passant": self.en_passant,
            "status": self.status,
            "candidate_moves": self.candidate_moves,
            "warnings": self.warnings,
            "zobrist": self.zobrist,
        }


class GameStateTracker:
    """Track game state across successive screenshots."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.board: chess.Board | None = None
        self.fen_history: list[str] = []
        self.move_history: list[chess.Move] = []
        self.zobrist_hash: int = 0
        self.eval_cache: dict[int, float] = {}

    @property
    def has_history(self) -> bool:
        return self.board is not None

    def reset(self) -> None:
        self.board = None
        self.fen_history.clear()
        self.move_history.clear()
        self.zobrist_hash = 0
        self.eval_cache.clear()

    def set_fen(self, fen: str) -> TrackerResult:
        """Initialize tracker from a complete FEN."""
        board = chess.Board(fen)
        self.board = board
        self.fen_history = [fen]
        self.move_history.clear()
        self.zobrist_hash = zobrist_hash(board)
        return TrackerResult(
            fen=board.fen(),
            placement=board.board_fen(),
            turn="w" if board.turn == chess.WHITE else "b",
            castling=self._castling_str(board),
            en_passant=chess.square_name(board.ep_square) if board.ep_square else "-",
            halfmove=board.halfmove_clock,
            fullmove=board.fullmove_number,
            status="ok",
            zobrist=self.zobrist_hash,
        )

    def update_from_vision(
        self,
        board_matrix: list[list[str | None]],
        orientation: Orientation = "white",
        side_hint: str = "white",
        castling_hint: str | None = None,
        en_passant_hint: str | None = None,
    ) -> TrackerResult:
        """Update state from vision-detected board matrix."""
        view = side_hint if side_hint in ("white", "black") else orientation
        board_matrix = matrix_from_screen_view(board_matrix, view)

        placement = matrix_to_placement(board_matrix)
        sanity_errors = validate_placement_sanity(placement)
        if sanity_errors:
            return self._vision_error_result(
                placement,
                warnings=sanity_errors,
                preserve_board=True,
            )

        if not self.has_history:
            result = self._initial_from_placement(
                placement,
                castling_hint,
                en_passant_hint,
                side_hint=view,
            )
        else:
            result = self._update_from_transition(placement, board_matrix)

        return result

    def _vision_error_result(
        self,
        placement: str,
        *,
        warnings: list[str],
        preserve_board: bool,
    ) -> TrackerResult:
        """Return vision_error without mutating tracker state when preserve_board."""
        if preserve_board and self.board is not None:
            board = self.board
            return TrackerResult(
                fen=board.fen(),
                placement=placement,
                turn="w" if board.turn == chess.WHITE else "b",
                castling=self._castling_str(board),
                en_passant=chess.square_name(board.ep_square)
                if board.ep_square
                else "-",
                halfmove=board.halfmove_clock,
                fullmove=board.fullmove_number,
                status="vision_error",
                warnings=warnings,
                zobrist=self.zobrist_hash,
            )
        return TrackerResult(
            fen=f"{placement} w - - 0 1",
            placement=placement,
            turn="w",
            castling="-",
            en_passant="-",
            halfmove=0,
            fullmove=1,
            status="vision_error",
            warnings=warnings,
        )

    def _initial_from_placement(
        self,
        placement: str,
        castling_hint: str | None,
        en_passant_hint: str | None,
        *,
        side_hint: Orientation = "white",
    ) -> TrackerResult:
        from src.chess_core.turn_infer import infer_side_to_move_from_placement

        inferred = infer_side_to_move_from_placement(placement)
        if inferred in ("white", "black"):
            turn = "w" if inferred == "white" else "b"
        elif side_hint in ("white", "black"):
            # Fallback only for positions where placement alone cannot infer turn.
            turn = "w" if side_hint == "white" else "b"
        else:
            turn = "w"
        if castling_hint is not None:
            castling = castling_hint
        elif self.config.infer_castling:
            castling = default_castling_for_placement(placement)
        else:
            castling = "-"
        en_passant = en_passant_hint if en_passant_hint is not None else "-"
        if en_passant == "-" and not self.config.infer_en_passant:
            en_passant = "-"

        fen = fen_from_matrix(
            self._placement_to_matrix(placement),
            turn=turn,
            castling=castling,
            en_passant=en_passant,
        )

        try:
            board = chess.Board(fen)
        except ValueError:
            return TrackerResult(
                fen=fen,
                placement=placement,
                turn=turn,
                castling=castling,
                en_passant=en_passant,
                halfmove=0,
                fullmove=1,
                status="vision_error",
                warnings=["Could not construct valid board from detected pieces"],
            )

        self.board = board
        self.fen_history = [board.fen()]
        self.zobrist_hash = zobrist_hash(board)

        warnings: list[str] = []
        if castling_hint is None and self.config.infer_castling:
            warnings.append("Castling rights inferred from piece placement")
        if en_passant_hint is None and not self.config.infer_en_passant:
            warnings.append("En passant square unknown; set to '-'")

        return TrackerResult(
            fen=board.fen(),
            placement=placement,
            turn=turn,
            castling=castling,
            en_passant=en_passant,
            halfmove=board.halfmove_clock,
            fullmove=board.fullmove_number,
            status="initial",
            warnings=warnings,
            zobrist=self.zobrist_hash,
        )

    def _update_from_transition(
        self,
        new_placement: str,
        new_matrix: list[list[str | None]],
    ) -> TrackerResult:
        assert self.board is not None
        prev_board = self.board.copy()
        prev_placement = prev_board.board_fen()

        if new_placement == prev_placement:
            return TrackerResult(
                fen=prev_board.fen(),
                placement=new_placement,
                turn="w" if prev_board.turn == chess.WHITE else "b",
                castling=self._castling_str(prev_board),
                en_passant=chess.square_name(prev_board.ep_square)
                if prev_board.ep_square
                else "-",
                halfmove=prev_board.halfmove_clock,
                fullmove=prev_board.fullmove_number,
                status="ok",
                warnings=["No change detected"],
                zobrist=self.zobrist_hash,
            )

        candidate_moves = self._find_transition_moves(prev_board, new_placement)

        if len(candidate_moves) == 1:
            move = candidate_moves[0]
            self.board.push(move)
            self.move_history.append(move)
            self.fen_history.append(self.board.fen())
            self.zobrist_hash = zobrist_hash(self.board)
            return TrackerResult(
                fen=self.board.fen(),
                placement=new_placement,
                turn="w" if self.board.turn == chess.WHITE else "b",
                castling=self._castling_str(self.board),
                en_passant=chess.square_name(self.board.ep_square)
                if self.board.ep_square
                else "-",
                halfmove=self.board.halfmove_clock,
                fullmove=self.board.fullmove_number,
                status="ok",
                candidate_moves=[move.uci()],
                zobrist=self.zobrist_hash,
            )

        if len(candidate_moves) > 1:
            next_turn = "w" if prev_board.turn == chess.BLACK else "b"
            return TrackerResult(
                fen=fen_from_matrix(
                    new_matrix,
                    turn=next_turn,
                    castling=self._castling_str(prev_board),
                    en_passant="-",
                    halfmove=prev_board.halfmove_clock,
                    fullmove=prev_board.fullmove_number,
                ),
                placement=new_placement,
                turn=next_turn,
                castling=self._castling_str(prev_board),
                en_passant="-",
                halfmove=prev_board.halfmove_clock,
                fullmove=prev_board.fullmove_number,
                status="ambiguous",
                candidate_moves=[m.uci() for m in candidate_moves],
                warnings=[f"Multiple legal transitions ({len(candidate_moves)})"],
            )

        return self._vision_error_result(
            new_placement,
            warnings=["No legal transition from previous state"],
            preserve_board=True,
        )

    def _find_transition_moves(
        self,
        start: chess.Board,
        target_placement: str,
        max_ply: int | None = None,
        *,
        max_nodes: int | None = None,
        max_seconds: float | None = None,
    ) -> list[chess.Move]:
        """Find legal move sequences (1..max_ply) that reach target placement.

        Depth 1-2 use push/pop (zero board copies) for speed.
        Depth 3+ fall back to BFS with copies (uncommon, only for skipped moves).
        """
        limit = max_ply if max_ply is not None else self.config.max_search_ply
        node_limit = max_nodes if max_nodes is not None else self.config.transition_max_nodes
        time_limit = (
            max_seconds if max_seconds is not None else self.config.transition_max_seconds
        )
        t0 = time.monotonic()
        deadline = t0 + time_limit

        # ── Depth 1: push/pop, zero copies ────────────────────────────────
        direct: list[chess.Move] = []
        for move in start.legal_moves:
            start.push(move)
            if start.board_fen() == target_placement:
                direct.append(move)
            start.pop()
        if direct:
            return direct

        if limit < 2:
            return []

        # ── Depth 2: nested push/pop, zero copies ─────────────────────────
        seen: set[str] = {start.board_fen()}
        found_d2: dict[str, chess.Move] = {}
        abort = False
        for m1 in start.legal_moves:
            if time.monotonic() >= deadline:
                abort = True
                break
            start.push(m1)
            k1 = start.board_fen()
            if k1 not in seen:
                seen.add(k1)
                for m2 in start.legal_moves:
                    if time.monotonic() >= deadline:
                        abort = True
                        break
                    start.push(m2)
                    if start.board_fen() == target_placement:
                        found_d2[m1.uci()] = m1
                    start.pop()
            start.pop()
            if abort:
                break

        if found_d2:
            return list(found_d2.values())

        elapsed_ms = (time.monotonic() - t0) * 1000
        if abort:
            log.warning(
                "transition:abort depth=2 ms=%.0f target=%s",
                elapsed_ms,
                target_placement[:40],
            )
            return []

        if limit < 3:
            if elapsed_ms > 50:
                log.debug(
                    "transition:miss depth=2 ms=%.0f target=%s",
                    elapsed_ms,
                    target_placement[:40],
                )
            return []

        # ── Depth 3+: BFS with board copies ───────────────────────────────
        # Only reached when max_search_ply >= 3 (covers skipped moves in auto mode).
        # Rebuild depth-2 frontier from copies since push/pop left no persistent boards.
        nodes = len(seen)
        frontier: list[tuple[chess.Board, chess.Move]] = []
        for m1 in start.legal_moves:
            if time.monotonic() >= deadline or nodes >= node_limit:
                break
            c1 = start.copy()
            c1.push(m1)
            k1 = c1.board_fen()
            if k1 in seen:
                continue
            seen.add(k1)
            for m2 in c1.legal_moves:
                if time.monotonic() >= deadline or nodes >= node_limit:
                    break
                c2 = c1.copy()
                c2.push(m2)
                k2 = c2.board_fen()
                if k2 not in seen:
                    frontier.append((c2, m1))
                nodes += 1

        for depth in range(3, limit + 1):
            if not frontier:
                break
            if nodes >= node_limit or time.monotonic() >= deadline:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.warning(
                    "transition:abort depth=%s nodes=%s ms=%.0f target=%s",
                    depth,
                    nodes,
                    elapsed_ms,
                    target_placement[:40],
                )
                return []

            level_found: dict[str, chess.Move] = {}
            next_frontier: list[tuple[chess.Board, chess.Move]] = []

            for board, first_move in frontier:
                if nodes >= node_limit or time.monotonic() >= deadline:
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    log.warning(
                        "transition:abort depth=%s nodes=%s ms=%.0f target=%s",
                        depth,
                        nodes,
                        elapsed_ms,
                        target_placement[:40],
                    )
                    return list(level_found.values()) if level_found else []
                nodes += 1
                key = board.board_fen()
                if key in seen:
                    continue
                seen.add(key)
                if key == target_placement:
                    level_found[first_move.uci()] = first_move
                    continue
                if depth < limit:
                    for move in board.legal_moves:
                        if time.monotonic() >= deadline or nodes >= node_limit:
                            break
                        child = board.copy()
                        child.push(move)
                        next_frontier.append((child, first_move))

            if level_found:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.info(
                    "transition:found depth=%s nodes=%s ms=%.0f moves=%s",
                    depth,
                    nodes,
                    elapsed_ms,
                    list(level_found.keys())[:5],
                )
                return list(level_found.values())

            frontier = next_frontier

        elapsed_ms = (time.monotonic() - t0) * 1000
        if nodes > 100:
            log.info(
                "transition:miss nodes=%s ms=%.0f target=%s",
                nodes,
                elapsed_ms,
                target_placement[:40],
            )
        return []

    def _reconcile_board_placement(self, placement: str) -> None:
        """Reset internal board to detected placement preserving turn when possible."""
        assert self.board is not None
        turn = "w" if self.board.turn == chess.WHITE else "b"
        castling = self._castling_str(self.board)
        fen = f"{placement} {turn} {castling} - {self.board.halfmove_clock} {self.board.fullmove_number}"
        try:
            self.board = chess.Board(fen)
        except ValueError:
            fen = f"{placement} {turn} - - 0 1"
            self.board = chess.Board(fen)
        self.zobrist_hash = zobrist_hash(self.board)
        if not self.fen_history or self.fen_history[-1] != self.board.fen():
            self.fen_history.append(self.board.fen())

    @staticmethod
    def _castling_str(board: chess.Board) -> str:
        s = ""
        if board.has_kingside_castling_rights(chess.WHITE):
            s += "K"
        if board.has_queenside_castling_rights(chess.WHITE):
            s += "Q"
        if board.has_kingside_castling_rights(chess.BLACK):
            s += "k"
        if board.has_queenside_castling_rights(chess.BLACK):
            s += "q"
        return s or "-"

    @staticmethod
    def _placement_to_matrix(placement: str) -> list[list[str | None]]:
        matrix: list[list[str | None]] = []
        for rank_str in placement.split("/"):
            row: list[str | None] = []
            for ch in rank_str:
                if ch.isdigit():
                    row.extend([None] * int(ch))
                else:
                    row.append(ch)
            matrix.append(row)
        return matrix

    def get_board_matrix(self) -> list[list[str | None]] | None:
        if self.board is None:
            return None
        return board_matrix_from_board(self.board)
