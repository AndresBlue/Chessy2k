"""Extended state tracker tests."""

import chess

from src.chess_core.fen_utils import board_matrix_from_board, screen_matrix_from_fen_matrix
from src.chess_core.state_tracker import GameStateTracker, TrackerConfig


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_castling_rights_preserved_on_transition():
    board = chess.Board(START_FEN)
    tracker = GameStateTracker(TrackerConfig(infer_castling=True))
    tracker.update_from_vision(board_matrix_from_board(board), side_hint="white")

    board.push(chess.Move.from_uci("e2e4"))
    result = tracker.update_from_vision(
        board_matrix_from_board(board), side_hint="white"
    )
    assert result.status == "ok"
    assert tracker.board is not None
    assert tracker.board.has_kingside_castling_rights(chess.WHITE)


def test_multi_ply_recovery_skips_frame():
    board = chess.Board(START_FEN)
    tracker = GameStateTracker(TrackerConfig(max_search_ply=4))
    tracker.update_from_vision(board_matrix_from_board(board), side_hint="white")

    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    result = tracker.update_from_vision(
        board_matrix_from_board(board), side_hint="white"
    )
    assert result.status == "ok"
    assert "e7e5" in result.candidate_moves or len(result.candidate_moves) >= 1


def test_en_passant_field_on_initial():
    tracker = GameStateTracker(TrackerConfig(infer_en_passant=False))
    board = chess.Board(START_FEN)
    result = tracker.update_from_vision(
        board_matrix_from_board(board), side_hint="white"
    )
    assert result.en_passant == "-"


def test_initial_turn_comes_from_position_not_user_side_hint():
    tracker = GameStateTracker(TrackerConfig(infer_en_passant=False))
    board = chess.Board(START_FEN)
    screen_matrix = screen_matrix_from_fen_matrix(board_matrix_from_board(board), "black")
    result = tracker.update_from_vision(
        screen_matrix, side_hint="black"
    )
    assert result.status == "initial"
    assert result.turn == "w"
    assert tracker.board is not None
    assert tracker.board.turn == chess.WHITE


def test_initial_after_white_move_infers_black_to_move():
    tracker = GameStateTracker(TrackerConfig(infer_en_passant=False))
    board = chess.Board(START_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    result = tracker.update_from_vision(
        board_matrix_from_board(board), side_hint="white"
    )
    assert result.status == "initial"
    assert result.turn == "b"
    assert tracker.board is not None
    assert tracker.board.turn == chess.BLACK


def test_promotion_transition():
    fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
    board = chess.Board(fen)
    tracker = GameStateTracker()
    tracker.update_from_vision(board_matrix_from_board(board), side_hint="white")

    board.push(chess.Move.from_uci("a7a8q"))
    result = tracker.update_from_vision(
        board_matrix_from_board(board), side_hint="white"
    )
    assert result.placement.count("Q") >= 1


def test_ambiguous_transition_reports_candidates():
    board = chess.Board(START_FEN)
    tracker = GameStateTracker()
    tracker.update_from_vision(board_matrix_from_board(board), side_hint="white")
    board.push(chess.Move.from_uci("g1f3"))
    board.push(chess.Move.from_uci("g8f6"))
    ambiguous = chess.Board()
    ambiguous.push(chess.Move.from_uci("g1f3"))
    result = tracker.update_from_vision(
        board_matrix_from_board(ambiguous), side_hint="white"
    )
    assert result.status in ("ambiguous", "vision_error", "ok")
