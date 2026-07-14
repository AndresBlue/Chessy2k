"""Simple opening book from Polyglot .bin or move list."""

from __future__ import annotations

import random
import struct
from pathlib import Path

import chess


class OpeningBook:
    """Simple opening book from Polyglot .bin or move list."""

    def __init__(self, book_path: str | None = None):
        self.entries: dict[int, list[tuple[str, int]]] = {}
        if book_path and Path(book_path).exists():
            self._load_polyglot(book_path)

    def _load_polyglot(self, path: str) -> None:
        with open(path, "rb") as f:
            while True:
                data = f.read(16)
                if len(data) < 16:
                    break
                key, move_raw, weight, _ = struct.unpack(">QHHI", data)
                move = self._decode_polyglot_move(move_raw)
                self.entries.setdefault(key, []).append((move, weight))

    @staticmethod
    def _decode_polyglot_move(move_raw: int) -> str:
        to_sq = move_raw & 0x3F
        from_sq = (move_raw >> 6) & 0x3F
        promo = (move_raw >> 12) & 0x7
        promo_map = {0: None, 1: "n", 2: "b", 3: "r", 4: "q"}
        uci = chess.square_name(from_sq) + chess.square_name(to_sq)
        if promo_map.get(promo):
            uci += promo_map[promo]
        return uci

    def get_move(self, board: chess.Board, temperature: float = 0.0) -> str | None:
        import chess.polyglot

        key = chess.polyglot.zobrist_hash(board)
        moves = self.entries.get(key, [])
        if not moves:
            return None

        legal_uci = {m.uci() for m in board.legal_moves}
        moves = [(uci, w) for uci, w in moves if uci in legal_uci]
        if not moves:
            return None

        if temperature < 1e-3:
            return max(moves, key=lambda x: x[1])[0]

        weights = [w for _, w in moves]
        total = sum(weights)
        if total <= 0:
            return moves[0][0]
        probs = [w / total for w in weights]
        return random.choices([m for m, _ in moves], weights=probs)[0]
