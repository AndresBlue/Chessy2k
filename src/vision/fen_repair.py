"""Repair misclassified boards into legal FEN positions."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from src.chess_core.fen_utils import CLASS_TO_PIECE, PIECE_TO_CLASS, fen_from_matrix, matrix_to_placement
from src.chess_core.legal_validator import validate_fen_for_analysis

PIECE_LIMITS: dict[str, int] = {
    "K": 1,
    "Q": 9,
    "R": 10,
    "B": 10,
    "N": 10,
    "P": 8,
    "k": 1,
    "q": 9,
    "r": 10,
    "b": 10,
    "n": 10,
    "p": 8,
}


@dataclass
class RepairResult:
    board_matrix: list[list[str | None]]
    fen: str
    repaired: bool
    notes: list[str]


def infer_castling_conservative(matrix: list[list[str | None]]) -> str:
    """Conservative castling rights from king/rook squares only."""
    rights = ""

    white_king = matrix[7][4] == "K"
    black_king = matrix[0][4] == "k"
    if white_king and matrix[7][7] == "R":
        rights += "K"
    if white_king and matrix[7][0] == "R":
        rights += "Q"
    if black_king and matrix[0][7] == "r":
        rights += "k"
    if black_king and matrix[0][0] == "r":
        rights += "q"
    return rights or "-"


def _set_square(matrix: list[list[str | None]], rank: int, file: int, piece: str | None) -> None:
    matrix[rank][file] = piece


def _king_keep_score(rank: int, file: int, king_piece: str, prob: float) -> float:
    score = prob
    if king_piece == "K" and file == 4:
        score += 2.0
    if king_piece == "k" and file == 4:
        score += 2.0
    if king_piece == "K" and rank == 7:
        score += 0.15
    if king_piece == "k" and rank == 0:
        score += 0.15
    return score


def _fix_kings(matrix: list[list[str | None]], probs: np.ndarray, notes: list[str]) -> list[tuple[int, int]]:
    """Fix king counts. Returns squares that were demoted (for later search)."""
    demoted: list[tuple[int, int]] = []
    specs = [("K", 6), ("k", 12)]
    for king_piece, king_cls in specs:
        positions = [
            (r, f)
            for r, row in enumerate(matrix)
            for f, cell in enumerate(row)
            if cell == king_piece
        ]
        if len(positions) == 1:
            continue

        if len(positions) == 0:
            best_idx = int(np.argmax(probs[:, king_cls]))
            r, f = best_idx // 8, best_idx % 8
            _set_square(matrix, r, f, king_piece)
            notes.append(f"Inserted missing {king_piece} on square {r},{f}")
            continue

        best_pos = max(
            positions,
            key=lambda pos: _king_keep_score(
                pos[0],
                pos[1],
                king_piece,
                float(probs[pos[0] * 8 + pos[1], king_cls]),
            ),
        )
        for pos in positions:
            if pos == best_pos:
                continue
            r, f = pos
            _demote_extra_king(matrix, r, f, probs, king_piece, king_cls, notes)
            demoted.append((r, f))

    return demoted


def _demote_extra_king(
    matrix: list[list[str | None]],
    r: int,
    f: int,
    probs: np.ndarray,
    king_piece: str,
    king_cls: int,
    notes: list[str],
) -> None:
    idx = r * 8 + f
    white_piece = king_piece.isupper()
    king_prob = float(probs[idx, king_cls])
    empty_prob = float(probs[idx, 0])

    if empty_prob >= 0.18 or empty_prob >= king_prob * 0.45:
        _set_square(matrix, r, f, None)
        notes.append(f"Demoted extra {king_piece} at {r},{f} -> empty")
        return

    alt: str | None = None
    best_prob = -1.0
    for cls in np.argsort(probs[idx])[::-1]:
        piece = CLASS_TO_PIECE.get(int(cls))
        if piece is None:
            if float(probs[idx, cls]) > best_prob:
                best_prob = float(probs[idx, cls])
                alt = None
            continue
        if piece == king_piece:
            continue
        if white_piece and piece.islower():
            continue
        if not white_piece and piece.isupper():
            continue
        prob = float(probs[idx, cls])
        if prob > best_prob:
            best_prob = prob
            alt = piece

    _set_square(matrix, r, f, alt)
    notes.append(f"Demoted extra {king_piece} at {r},{f} -> {alt or 'empty'}")


def _fix_back_rank_pawns(matrix: list[list[str | None]], notes: list[str]) -> None:
    for f in range(8):
        if matrix[0][f] == "p":
            _set_square(matrix, 0, f, None)
            notes.append("Removed black pawn on 8th rank")
        if matrix[7][f] == "P":
            _set_square(matrix, 7, f, None)
            notes.append("Removed white pawn on 1st rank")


def _fix_piece_counts(matrix: list[list[str | None]], probs: np.ndarray, notes: list[str]) -> None:
    for piece, limit in PIECE_LIMITS.items():
        if piece in ("K", "k"):
            continue
        cls = PIECE_TO_CLASS[piece]
        positions = [
            (r, f)
            for r, row in enumerate(matrix)
            for f, cell in enumerate(row)
            if cell == piece
        ]
        while len(positions) > limit:
            worst = min(positions, key=lambda pos: float(probs[pos[0] * 8 + pos[1], cls]))
            _set_square(matrix, worst[0], worst[1], None)
            notes.append(f"Removed extra {piece} at {worst[0]},{worst[1]}")
            positions.remove(worst)


def _class_matrix_from_matrix(matrix: list[list[str | None]]) -> np.ndarray:
    out = np.zeros((8, 8), dtype=np.int64)
    for r in range(8):
        for f in range(8):
            piece = matrix[r][f]
            out[r, f] = PIECE_TO_CLASS.get(piece, 0) if piece else 0
    return out


def _matrix_from_class_matrix(class_matrix: np.ndarray) -> list[list[str | None]]:
    matrix: list[list[str | None]] = []
    for r in range(8):
        row: list[str | None] = []
        for f in range(8):
            row.append(CLASS_TO_PIECE[int(class_matrix[r, f])])
        matrix.append(row)
    return matrix


def _try_build_valid_fen(
    matrix: list[list[str | None]],
    side_hint: str,
) -> str | None:
    turns = ["w", "b"] if side_hint == "white" else ["b", "w"]
    castlings = [infer_castling_conservative(matrix), "-"]

    for turn, castling in itertools.product(turns, castlings):
        fen = fen_from_matrix(matrix, turn=turn, castling=castling, en_passant="-")
        result = validate_fen_for_analysis(fen)
        if result.is_valid:
            return result.fen
    return None


def _search_alternatives(
    matrix: list[list[str | None]],
    probs: np.ndarray,
    side_hint: str,
    priority_squares: list[tuple[int, int]] | None = None,
    max_squares: int = 6,
) -> str | None:
    conf: list[tuple[float, int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for r, f in priority_squares or []:
        idx = r * 8 + f
        conf.append((float(probs[idx].max()), r, f, idx))
        seen.add((r, f))

    for r in range(8):
        for f in range(8):
            if (r, f) in seen:
                continue
            idx = r * 8 + f
            conf.append((float(probs[idx].max()), r, f, idx))
    conf.sort(key=lambda item: item[0])

    candidates = conf[:max_squares]
    option_lists: list[list[tuple[int, int, int]]] = []
    for _, r, f, idx in candidates:
        top = np.argsort(probs[idx])[::-1][:4]
        option_lists.append([(r, f, int(cls)) for cls in top])

    for combo in itertools.product(*option_lists):
        class_matrix = _class_matrix_from_matrix(matrix)
        for r, f, cls in combo:
            class_matrix[r, f] = cls
        trial = _matrix_from_class_matrix(class_matrix)
        fen = _try_build_valid_fen(trial, side_hint)
        if fen:
            return fen
    return None


def repair_board_fen(
    board_matrix: list[list[str | None]],
    probs: np.ndarray,
    side_hint: str = "white",
) -> RepairResult:
    """Try to repair a classified board into a legal FEN."""
    matrix = [row[:] for row in board_matrix]
    notes: list[str] = []

    fen = _try_build_valid_fen(matrix, side_hint)
    if fen:
        return RepairResult(matrix, fen, False, notes)

    demoted = _fix_kings(matrix, probs, notes)
    _fix_back_rank_pawns(matrix, notes)
    _fix_piece_counts(matrix, probs, notes)
    fen = _try_build_valid_fen(matrix, side_hint)
    if fen:
        notes.append("Applied king/pawn/piece-count fixes")
        return RepairResult(matrix, fen, True, notes)

    fen = _search_alternatives(matrix, probs, side_hint, priority_squares=demoted)
    if fen:
        from src.chess_core.fen_utils import board_matrix_from_fen

        notes.append("Resolved via alternative classes on uncertain squares")
        repaired_matrix = board_matrix_from_fen(fen)
        return RepairResult(repaired_matrix, fen, True, notes)

    fen = _search_alternatives(matrix, probs, side_hint)
    if fen:
        from src.chess_core.fen_utils import board_matrix_from_fen

        notes.append("Resolved via global alternative search")
        repaired_matrix = board_matrix_from_fen(fen)
        return RepairResult(repaired_matrix, fen, True, notes)

    placement = matrix_to_placement(matrix)
    turn = "w" if side_hint == "white" else "b"
    fallback_fen = fen_from_matrix(matrix, turn=turn, castling="-", en_passant="-")
    return RepairResult(matrix, fallback_fen, False, notes + ["Could not repair to legal FEN"])
