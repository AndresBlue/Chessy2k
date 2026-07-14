"""Strategic feature extraction from chess.Board (CPU, no GPU)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np

STRATEGIC_FEATURE_DIM = 33

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


@dataclass
class StrategicFeatures:
    center_control_white: float
    center_control_black: float
    extended_center_white: float
    extended_center_black: float
    king_safety_white: float
    king_safety_black: float
    king_attacks_white: float
    king_attacks_black: float
    king_zone_pressure_white: int
    king_zone_pressure_black: int
    isolated_pawns_white: int
    isolated_pawns_black: int
    doubled_pawns_white: int
    doubled_pawns_black: int
    passed_pawns_white: int
    passed_pawns_black: int
    mobility_white: int
    mobility_black: int
    mobility_ratio: float
    pieces_under_attack_white: int
    pieces_under_attack_black: int
    hanging_pieces_white: int
    hanging_pieces_black: int
    pawn_tension: int
    semi_open_files_white: int
    semi_open_files_black: int
    legal_moves: int
    capture_moves: int
    check_moves: int
    complexity_score: float

    def to_vector(self) -> np.ndarray:
        return extract_strategic_vector_from_board_features(self)


def _king_zone_squares(king_sq: int) -> set[int]:
    zone: set[int] = {king_sq}
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
        zone.add(sq)
    return zone


def _count_attacks_on_squares(board: chess.Board, squares: set[int], by_color: chess.Color) -> int:
    return sum(1 for sq in squares if board.is_attacked_by(by_color, sq))


def _file_pawns(board: chess.Board, color: chess.Color, file_idx: int) -> list[int]:
    pawns: list[int] = []
    for rank in range(8):
        sq = chess.square(file_idx, rank)
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            pawns.append(sq)
    return pawns


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    count = 0
    for file_idx in range(8):
        pawns = _file_pawns(board, color, file_idx)
        if not pawns:
            continue
        has_neighbor = False
        for adj in (file_idx - 1, file_idx + 1):
            if 0 <= adj < 8 and _file_pawns(board, color, adj):
                has_neighbor = True
                break
        if not has_neighbor:
            count += len(pawns)
    return count


def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    doubled = 0
    for file_idx in range(8):
        n = len(_file_pawns(board, color, file_idx))
        if n > 1:
            doubled += n - 1
    return doubled


def _is_passed_pawn(board: chess.Board, sq: int, color: chess.Color) -> bool:
    file_idx = chess.square_file(sq)
    rank = chess.square_rank(sq)
    enemy = not color
    for adj_file in range(max(0, file_idx - 1), min(8, file_idx + 2)):
        for r in range(8):
            piece = board.piece_at(chess.square(adj_file, r))
            if piece and piece.piece_type == chess.PAWN and piece.color == enemy:
                if color == chess.WHITE and r > rank:
                    return False
                if color == chess.BLACK and r < rank:
                    return False
    return True


def _passed_pawns(board: chess.Board, color: chess.Color) -> int:
    return sum(
        1
        for sq in chess.SQUARES
        if (p := board.piece_at(sq))
        and p.piece_type == chess.PAWN
        and p.color == color
        and _is_passed_pawn(board, sq, color)
    )


def _semi_open_files(board: chess.Board, color: chess.Color) -> int:
    enemy = not color
    open_count = 0
    for file_idx in range(8):
        own = bool(_file_pawns(board, color, file_idx))
        opp = bool(_file_pawns(board, enemy, file_idx))
        if not own and opp:
            open_count += 1
    return open_count


def _pieces_under_attack(board: chess.Board, color: chess.Color) -> int:
    enemy = not color
    count = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and board.is_attacked_by(enemy, sq):
            count += 1
    return count


def _hanging_pieces(board: chess.Board, color: chess.Color) -> int:
    enemy = not color
    hanging = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color or piece.piece_type == chess.KING:
            continue
        if board.is_attacked_by(enemy, sq) and not board.is_attacked_by(color, sq):
            hanging += 1
    return hanging


def _pawn_tension_count(board: chess.Board) -> int:
    tension = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.piece_type != chess.PAWN:
            continue
        attacks = chess.BB_PAWN_ATTACKS[piece.color][sq]
        for target in chess.SquareSet(attacks):
            target_piece = board.piece_at(target)
            if target_piece and target_piece.piece_type == chess.PAWN:
                tension += 1
    return tension // 2


def _center_control(board: chess.Board, color: chess.Color, squares: set[int]) -> float:
    controlled = sum(1 for sq in squares if board.is_attacked_by(color, sq))
    return controlled / max(len(squares), 1)


def _mobility(board: chess.Board, color: chess.Color) -> int:
    if board.turn == color:
        return board.legal_moves.count()
    b = board.copy()
    b.turn = color
    return b.legal_moves.count()


def extract_strategic_features(board: chess.Board) -> StrategicFeatures:
    """Compute strategic features from the current position."""
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    w_zone = _king_zone_squares(white_king) if white_king is not None else set()
    b_zone = _king_zone_squares(black_king) if black_king is not None else set()

    w_mob = _mobility(board, chess.WHITE)
    b_mob = _mobility(board, chess.BLACK)
    legal = board.legal_moves.count()
    captures = sum(1 for m in board.legal_moves if board.is_capture(m))
    checks = sum(1 for m in board.legal_moves if board.gives_check(m))

    complexity = (
        math.log1p(legal) * 0.3
        + math.log1p(captures) * 0.25
        + math.log1p(checks) * 0.2
        + (_pieces_under_attack(board, chess.WHITE) + _pieces_under_attack(board, chess.BLACK)) * 0.05
        + _pawn_tension_count(board) * 0.1
    )

    return StrategicFeatures(
        center_control_white=_center_control(board, chess.WHITE, CENTER_SQUARES),
        center_control_black=_center_control(board, chess.BLACK, CENTER_SQUARES),
        extended_center_white=_center_control(board, chess.WHITE, EXTENDED_CENTER),
        extended_center_black=_center_control(board, chess.BLACK, EXTENDED_CENTER),
        king_safety_white=1.0 - _count_attacks_on_squares(board, w_zone, chess.BLACK) / max(len(w_zone), 1),
        king_safety_black=1.0 - _count_attacks_on_squares(board, b_zone, chess.WHITE) / max(len(b_zone), 1),
        king_attacks_white=_count_attacks_on_squares(board, b_zone, chess.WHITE) / max(len(b_zone), 1),
        king_attacks_black=_count_attacks_on_squares(board, w_zone, chess.BLACK) / max(len(w_zone), 1),
        king_zone_pressure_white=_count_attacks_on_squares(board, w_zone, chess.BLACK),
        king_zone_pressure_black=_count_attacks_on_squares(board, b_zone, chess.WHITE),
        isolated_pawns_white=_isolated_pawns(board, chess.WHITE),
        isolated_pawns_black=_isolated_pawns(board, chess.BLACK),
        doubled_pawns_white=_doubled_pawns(board, chess.WHITE),
        doubled_pawns_black=_doubled_pawns(board, chess.BLACK),
        passed_pawns_white=_passed_pawns(board, chess.WHITE),
        passed_pawns_black=_passed_pawns(board, chess.BLACK),
        mobility_white=w_mob,
        mobility_black=b_mob,
        mobility_ratio=w_mob / max(w_mob + b_mob, 1),
        pieces_under_attack_white=_pieces_under_attack(board, chess.WHITE),
        pieces_under_attack_black=_pieces_under_attack(board, chess.BLACK),
        hanging_pieces_white=_hanging_pieces(board, chess.WHITE),
        hanging_pieces_black=_hanging_pieces(board, chess.BLACK),
        pawn_tension=_pawn_tension_count(board),
        semi_open_files_white=_semi_open_files(board, chess.WHITE),
        semi_open_files_black=_semi_open_files(board, chess.BLACK),
        legal_moves=legal,
        capture_moves=captures,
        check_moves=checks,
        complexity_score=complexity,
    )


def extract_strategic_vector(board: chess.Board) -> np.ndarray:
    """Return normalized strategic feature vector of shape (STRATEGIC_FEATURE_DIM,)."""
    return extract_strategic_features(board).to_vector()


def extract_strategic_vector_from_board_features(f: StrategicFeatures) -> np.ndarray:
    """Flatten StrategicFeatures to a normalized vector."""
    stm_white = f.mobility_white >= f.mobility_black
    stm_mob = f.mobility_white if stm_white else f.mobility_black
    opp_mob = f.mobility_black if stm_white else f.mobility_white
    stm_adv = (stm_mob - opp_mob) / max(stm_mob + opp_mob, 1)
    open_files = (f.semi_open_files_white + f.semi_open_files_black) / 16.0
    attack_balance = (f.pieces_under_attack_black - f.pieces_under_attack_white) / 16.0

    raw = np.array(
        [
            f.center_control_white,
            f.center_control_black,
            f.extended_center_white,
            f.extended_center_black,
            f.king_safety_white,
            f.king_safety_black,
            f.king_attacks_white,
            f.king_attacks_black,
            f.king_zone_pressure_white / 8.0,
            f.king_zone_pressure_black / 8.0,
            f.isolated_pawns_white / 8.0,
            f.isolated_pawns_black / 8.0,
            f.doubled_pawns_white / 8.0,
            f.doubled_pawns_black / 8.0,
            f.passed_pawns_white / 8.0,
            f.passed_pawns_black / 8.0,
            f.mobility_white / 50.0,
            f.mobility_black / 50.0,
            f.mobility_ratio,
            f.pieces_under_attack_white / 16.0,
            f.pieces_under_attack_black / 16.0,
            f.hanging_pieces_white / 8.0,
            f.hanging_pieces_black / 8.0,
            f.pawn_tension / 10.0,
            f.semi_open_files_white / 8.0,
            f.semi_open_files_black / 8.0,
            math.log1p(f.legal_moves) / math.log1p(50),
            f.capture_moves / max(f.legal_moves, 1),
            f.check_moves / max(f.legal_moves, 1),
            min(f.complexity_score / 5.0, 1.0),
            stm_adv,
            open_files,
            attack_balance,
        ],
        dtype=np.float32,
    )
    assert raw.shape[0] == STRATEGIC_FEATURE_DIM, raw.shape
    return raw


def complexity_score(board: chess.Board) -> float:
    """Scalar complexity in [0, ~1] for reward/MCTS."""
    return float(extract_strategic_features(board).complexity_score / 5.0)
