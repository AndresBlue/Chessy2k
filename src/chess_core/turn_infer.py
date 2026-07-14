"""Infer side to move from piece placement."""

from __future__ import annotations

from collections import deque
from typing import Literal

import chess

from src.chess_core.fen_utils import default_castling_for_placement

Side = Literal["white", "black"]

_START_PLACEMENT = chess.Board().board_fen()


def _placement_square_diff(a: str, b: str) -> int:
    """Count squares where two placements differ (ignoring rank separators)."""

    def cells(p: str) -> list[str]:
        out: list[str] = []
        for ch in p.replace("/", ""):
            if ch.isdigit():
                out.extend(["."] * int(ch))
            else:
                out.append(ch)
        return out

    ca, cb = cells(a), cells(b)
    if len(ca) != 64 or len(cb) != 64:
        return 64
    return sum(1 for x, y in zip(ca, cb) if x != y)


def infer_side_to_move_from_placement(
    placement: str,
    *,
    max_ply: int = 4,
    max_nodes: int = 10_000,
) -> Side | None:
    """
    Guess who moves next from piece placement.

    1. Reachability check (fast; works in middlegame).
    2. Bounded BFS from the standard start (opening positions only).
    """
    target = placement.strip()
    if not target:
        return None
    if target == _START_PLACEMENT:
        return "white"

    by_reach = _infer_by_reachability(target)
    if by_reach is not None:
        return by_reach
    if _placement_square_diff(_START_PLACEMENT, target) > max_ply * 2:
        return None
    return _infer_from_start_bfs(target, max_ply=max_ply, max_nodes=max_nodes)


def _infer_from_start_bfs(
    target: str,
    *,
    max_ply: int,
    max_nodes: int,
) -> Side | None:
    start = chess.Board()
    seen: set[str] = set()
    queue: deque[tuple[chess.Board, int]] = deque([(start, 0)])
    nodes = 0

    while queue:
        if nodes >= max_nodes:
            return None
        board, depth = queue.popleft()
        nodes += 1
        if depth > max_ply:
            continue
        key = board.board_fen()
        if key in seen:
            continue
        seen.add(key)
        if key == target:
            return "white" if board.turn == chess.WHITE else "black"
        if depth == max_ply:
            continue
        for move in board.legal_moves:
            next_board = board.copy()
            next_board.push(move)
            queue.append((next_board, depth + 1))
    return None


def _infer_by_reachability(placement: str) -> Side | None:
    """Pick the turn for which a minimal FEN is a reachable chess position."""
    castling = default_castling_for_placement(placement)
    valid: list[Side] = []
    for turn in ("w", "b"):
        fen = f"{placement} {turn} {castling} - 0 1"
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.is_valid():
            valid.append("white" if turn == "w" else "black")
    if len(valid) == 1:
        return valid[0]
    return None
