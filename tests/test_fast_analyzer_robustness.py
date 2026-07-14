"""FastAnalyzer guards and cancellation wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.chess_core.fen_utils import board_matrix_from_fen
from src.vision.pipeline import VisionResult


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _vision_result() -> VisionResult:
  return VisionResult(
    board_matrix=board_matrix_from_fen(START_FEN),
    fen_pieces="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    orientation="white",
    confidence=np.full((8, 8), 0.99, dtype=np.float32),
    ambiguous_squares=[],
    debug_image=None,
    cropped_board=np.zeros((64, 64, 3), dtype=np.uint8),
    time_ms=1.0,
    detection_method="test",
    board_bbox=(0, 0, 64, 64),
  )


def _engine_result(**kwargs):
  base = dict(
    best_move_san="e5",
    best_move_uci="e7e5",
    evaluation="0.00",
    engine_ms=1.0,
    phase="opening",
    criticality="low",
    planned_ms=0,
    think_note="Apertura conocida",
    from_book=False,
    game_over=False,
  )
  base.update(kwargs)
  return MagicMock(**base)


@pytest.fixture
def analyzer():
  with patch("src.app.fast_analyzer.VisionPipeline") as vision_cls, patch(
    "src.app.fast_analyzer.StockfishClient"
  ) as sf_cls, patch("src.app.fast_analyzer.HumanEngine") as human_cls:
    vision_cls.from_config.return_value = MagicMock(device_label="cpu")
    sf = sf_cls.return_value
    human = human_cls.return_value
    from src.app.fast_analyzer import FastAnalyzer

    fa = FastAnalyzer("sf", "ckpt")
    fa.stockfish = sf
    fa.human_engine = human
    yield fa


def test_analyze_forces_selected_black_side(analyzer):
  analyzer.tracker.update_from_vision = MagicMock(side_effect=AssertionError)
  analyzer.human_engine.analyze.return_value = _engine_result()
  image = np.zeros((64, 64, 3), dtype=np.uint8)
  result = analyzer.analyze_precomputed(image, _vision_result(), side="black")
  assert not result.error
  assert result.tracker_status == "forced_side"
  assert " b " in result.fen
  analyzer.human_engine.analyze.assert_called_once()
  engine_fen = analyzer.human_engine.analyze.call_args.args[0]
  assert " b " in engine_fen
  analyzer.tracker.update_from_vision.assert_not_called()


def test_analyze_forces_selected_white_side(analyzer):
  analyzer.tracker.update_from_vision = MagicMock(side_effect=AssertionError)
  analyzer.human_engine.analyze.return_value = _engine_result(
    best_move_san="e4",
    best_move_uci="e2e4",
  )
  image = np.zeros((64, 64, 3), dtype=np.uint8)
  result = analyzer.analyze_precomputed(image, _vision_result(), side="white")
  assert not result.error
  assert " w " in result.fen
  engine_fen = analyzer.human_engine.analyze.call_args.args[0]
  assert " w " in engine_fen
  analyzer.tracker.update_from_vision.assert_not_called()


def test_interrupt_delegates_to_stockfish(analyzer):
  analyzer.interrupt()
  analyzer.stockfish.interrupt_search.assert_called_once()
