"""Tests for opening book edge cases."""

import chess
import chess.polyglot

from src.search.opening_book import OpeningBook


def test_zero_weights_fallback():
    book = OpeningBook(None)
    board = chess.Board()
    key = chess.polyglot.zobrist_hash(board)
    book.entries[key] = [("e2e4", 0), ("d2d4", 0)]
    move = book.get_move(board, temperature=0.5)
    assert move in ("e2e4", "d2d4")


def test_filters_illegal_book_moves():
    book = OpeningBook(None)
    board = chess.Board()
    key = chess.polyglot.zobrist_hash(board)
    book.entries[key] = [("a1a1", 100), ("e2e4", 50)]
    move = book.get_move(board, temperature=0.0)
    assert move == "e2e4"
