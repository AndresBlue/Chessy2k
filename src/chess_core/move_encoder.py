"""AlphaZero-style move encoding (8x8x73)."""

from __future__ import annotations

import numpy as np
import chess

# 73 move types per from-square:
# 56 queen-like rays (8 directions x 7 distances)
# 8 knight moves
# 9 underpromotions (3 directions x 3 piece types)

INPUT_CHANNELS = 18
NUM_POLICY_PLANES = 73
POLICY_SHAPE = (8, 8, NUM_POLICY_PLANES)
POLICY_SIZE = 8 * 8 * NUM_POLICY_PLANES

_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]

_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

_UNDERPROMO_FILE_DELTAS = [-1, 0, 1]
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _ray_index(df: int, dr: int, distance: int) -> int:
    """Map direction + distance to plane index 0-55."""
    for di, (d_file, d_rank) in enumerate(_DIRECTIONS):
        if d_file == df and d_rank == dr:
            return di * 7 + (distance - 1)
    raise ValueError(f"Invalid ray direction ({df}, {dr})")


def _knight_index(df: int, dr: int) -> int:
    for ki, (kf, kr) in enumerate(_KNIGHT_DELTAS):
        if kf == df and kr == dr:
            return 56 + ki
    raise ValueError(f"Invalid knight delta ({df}, {dr})")


def _underpromo_index(df: int, promotion_piece: int) -> int:
    try:
        di = _UNDERPROMO_FILE_DELTAS.index(df)
    except ValueError as exc:
        raise ValueError(f"Invalid underpromotion ({df}, {promotion_piece})") from exc
    try:
        pi = _UNDERPROMO_PIECES.index(promotion_piece)
    except ValueError as exc:
        raise ValueError(f"Invalid underpromotion ({df}, {promotion_piece})") from exc
    return 64 + di * 3 + pi


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode move to flat index in [0, 8*8*73)."""
    from_sq = move.from_square
    to_sq = move.to_square
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)

    df = to_file - from_file
    dr = to_rank - from_rank

    if move.promotion and move.promotion != chess.QUEEN:
        plane = _underpromo_index(df, move.promotion)
    elif abs(df) * abs(dr) == 2 and abs(df) + abs(dr) == 3:
        plane = _knight_index(df, dr)
    else:
        if df == 0 and dr == 0:
            raise ValueError(f"Cannot encode move {move}")
        distance = max(abs(df), abs(dr))
        ndf = 0 if df == 0 else df // abs(df)
        ndr = 0 if dr == 0 else dr // abs(dr)
        plane = _ray_index(ndf, ndr, distance)

    return (from_rank * 8 + from_file) * NUM_POLICY_PLANES + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Decode flat index to Move; returns None if not legal."""
    from_flat = index // NUM_POLICY_PLANES
    plane = index % NUM_POLICY_PLANES
    from_file = from_flat % 8
    from_rank = from_flat // 8
    from_sq = chess.square(from_file, from_rank)

    if plane < 56:
        direction = plane // 7
        distance = plane % 7 + 1
        df, dr = _DIRECTIONS[direction]
        to_file = from_file + df * distance
        to_rank = from_rank + dr * distance
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            return None
        to_sq = chess.square(to_file, to_rank)
        promotion = None
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN and to_rank in (0, 7):
            promotion = chess.QUEEN
        move = chess.Move(from_sq, to_sq, promotion=promotion)
    elif plane < 64:
        ki = plane - 56
        df, dr = _KNIGHT_DELTAS[ki]
        to_file = from_file + df
        to_rank = from_rank + dr
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            return None
        move = chess.Move(from_sq, chess.square(to_file, to_rank))
    else:
        promo_idx = plane - 64
        di = promo_idx // 3
        pi = promo_idx % 3
        df = _UNDERPROMO_FILE_DELTAS[di]
        to_file = from_file + df
        to_rank = 7 if board.turn == chess.WHITE else 0
        if not (0 <= to_file < 8):
            return None
        move = chess.Move(
            from_sq,
            chess.square(to_file, to_rank),
            promotion=_UNDERPROMO_PIECES[pi],
        )

    return move if move in board.legal_moves else None


def legal_moves_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (8, 8, 73) for legal moves."""
    mask = np.zeros(POLICY_SHAPE, dtype=bool)
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        from_sq = move.from_square
        plane = idx % NUM_POLICY_PLANES
        rank = chess.square_rank(from_sq)
        file = chess.square_file(from_sq)
        mask[rank, file, plane] = True
    return mask


def legal_moves_flat_mask(board: chess.Board) -> np.ndarray:
    """Flat boolean mask of shape (8*8*73,)."""
    return legal_moves_mask(board).reshape(-1)


def board_to_tensor(
    board: chess.Board,
    history_planes: int = 0,
) -> np.ndarray:
    """Convert board to Nx8x8 tensor for neural network input."""
    # 12 piece planes + 1 side + 4 castling + 1 en passant = 18
    planes = 18 + history_planes
    tensor = np.zeros((planes, 8, 8), dtype=np.float32)

    piece_map = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_map[(piece.piece_type, piece.color)]
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[plane, rank, file] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        tensor[17, :, ep_file] = 1.0

    return tensor
