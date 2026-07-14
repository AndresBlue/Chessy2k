"""Legal position validation using python-chess."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chess


@dataclass
class ValidationResult:
    is_valid: bool
    fen: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    board: chess.Board | None = None
    is_check: bool = False
    is_checkmate: bool = False
    is_stalemate: bool = False
    is_insufficient_material: bool = False
    legal_move_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "fen": self.fen,
            "errors": self.errors,
            "warnings": self.warnings,
            "is_check": self.is_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "is_insufficient_material": self.is_insufficient_material,
            "legal_move_count": self.legal_move_count,
        }


def validate_fen(fen: str) -> ValidationResult:
    """Validate a complete FEN string (strict: includes reachability check)."""
    return _validate_fen_internal(fen, strict=True)


def validate_fen_for_analysis(fen: str) -> ValidationResult:
    """
    Validate a FEN for engine analysis after vision.

    Skips python-chess reachability (``is_valid``) which rejects many
    positions that are fine for Stockfish (e.g. one misclassified minor piece).
    """
    return _validate_fen_internal(fen, strict=False)


def _sanity_piece_counts(board: chess.Board) -> list[str]:
    """Return errors for impossible piece totals on the board."""
    errors: list[str] = []
    limits = {
        chess.KING: 1,
        chess.QUEEN: 9,
        chess.ROOK: 10,
        chess.BISHOP: 10,
        chess.KNIGHT: 10,
        chess.PAWN: 8,
    }
    for color in (chess.WHITE, chess.BLACK):
        for piece_type, max_count in limits.items():
            count = len(board.pieces(piece_type, color))
            if count > max_count:
                name = chess.piece_symbol(piece_type)
                label = name.upper() if color == chess.WHITE else name
                errors.append(f"Too many {label} pieces: {count} (max {max_count})")
    return errors


def _validate_fen_internal(fen: str, *, strict: bool) -> ValidationResult:
    """Validate a complete FEN string."""
    errors: list[str] = []
    warnings: list[str] = []
    board: chess.Board | None = None

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return ValidationResult(is_valid=False, fen=fen, errors=[str(exc)])

    if strict and not board.is_valid():
        errors.append("Position is not legally reachable (python-chess is_valid check failed)")

    errors.extend(_sanity_piece_counts(board))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            if rank in (0, 7):
                errors.append("Pawn on back rank")

    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    if white_kings != 1:
        errors.append(f"Expected 1 white king, found {white_kings}")
    if black_kings != 1:
        errors.append(f"Expected 1 black king, found {black_kings}")

    if not strict and not board.is_valid():
        warnings.append("Position may not be legally reachable; analysis will proceed anyway")

    if board.is_check():
        warnings.append("Side to move is in check")

    if board.is_checkmate():
        warnings.append("Position is checkmate")
    elif board.is_stalemate():
        warnings.append("Position is stalemate")
    elif board.is_insufficient_material():
        warnings.append("Insufficient material draw")

    return ValidationResult(
        is_valid=len(errors) == 0,
        fen=board.fen(),
        errors=errors,
        warnings=warnings,
        board=board,
        is_check=board.is_check(),
        is_checkmate=board.is_checkmate(),
        is_stalemate=board.is_stalemate(),
        is_insufficient_material=board.is_insufficient_material(),
        legal_move_count=board.legal_moves.count(),
    )


def validate_placement_sanity(placement: str) -> list[str]:
    """
    Fast sanity check on piece placement only (no reachability).

    Returns a list of error strings; empty means placement counts look plausible.
    """
    try:
        board = chess.Board(f"{placement} w - - 0 1")
    except ValueError as exc:
        return [str(exc)]

    errors = _sanity_piece_counts(board)

    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    if white_kings != 1:
        errors.append(f"Expected 1 white king, found {white_kings}")
    if black_kings != 1:
        errors.append(f"Expected 1 black king, found {black_kings}")

    total_pieces = len(board.piece_map())
    if total_pieces > 32:
        errors.append(f"Too many pieces on board: {total_pieces} (max 32)")

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            if rank in (0, 7):
                errors.append("Pawn on back rank")

    return errors


def validate_placement_only(
    placement: str,
    turn: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> ValidationResult:
    """Validate piece placement with auxiliary FEN fields."""
    fen = f"{placement} {turn} {castling} {en_passant} {halfmove} {fullmove}"
    return validate_fen(fen)


def is_legal_move(board: chess.Board, uci: str) -> bool:
    """Check if UCI move is legal in position."""
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False
    return move in board.legal_moves
