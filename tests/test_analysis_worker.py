"""Analysis worker queue, cancel, and stale callback suppression."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np

from src.app.analysis_worker import AnalysisJob, AnalysisWorker
from src.app.fast_analyzer import AnalysisOutput


def _output(**kwargs) -> AnalysisOutput:
  base = dict(
    fen="",
    best_move_san="",
    best_move_uci="",
    evaluation="",
    vision_ms=0.0,
    engine_ms=0.0,
    total_ms=0.0,
  )
  base.update(kwargs)
  return AnalysisOutput(**base)


def test_cancel_interrupts_analyzer():
  analyzer = MagicMock()
  analyzer.analyze.return_value = _output(best_move_uci="e2e4")
  worker = AnalysisWorker(analyzer)
  worker.cancel()
  analyzer.interrupt.assert_called_once()
  worker.shutdown()


def test_stale_analysis_callback_suppressed():
  analyzer = MagicMock()
  done: list[int] = []

  def slow_analyze(*_args, **kwargs):
    time.sleep(0.3)
    return _output(best_move_uci="e2e4")

  analyzer.analyze.side_effect = slow_analyze
  worker = AnalysisWorker(analyzer)

  job1 = AnalysisJob(
    analysis_id=1,
    image=np.zeros((8, 8, 3), dtype=np.uint8),
    side="white",
    show_predictions=False,
    human_mode=False,
    target_elo=2000,
    on_thinking=None,
    on_complete=lambda _r: done.append(1),
  )
  job2 = AnalysisJob(
    analysis_id=2,
    image=np.zeros((8, 8, 3), dtype=np.uint8),
    side="white",
    show_predictions=False,
    human_mode=False,
    target_elo=2000,
    on_thinking=None,
    on_complete=lambda _r: done.append(2),
  )
  worker.submit(job1)
  time.sleep(0.05)
  worker.submit(job2)
  time.sleep(0.6)
  worker.shutdown()
  assert done == [2]

