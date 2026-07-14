"""Single-threaded analysis worker (vision + engine serialized)."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

from src.app.fast_analyzer import AnalysisOutput, FastAnalyzer
from src.app.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class AnalysisJob:
    analysis_id: int
    image: object
    side: str
    show_predictions: bool
    human_mode: bool
    target_elo: int
    on_thinking: Callable[[int], None] | None
    on_complete: Callable[[AnalysisOutput], None]
    precomputed_vision: object | None = None
    engine_mode: str = "stockfish"


@dataclass
class _WorkItem:
    analysis: AnalysisJob


class AnalysisWorker:
    """Process analysis jobs one at a time."""

    def __init__(self, analyzer: FastAnalyzer) -> None:
        self._analyzer = analyzer
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue(maxsize=1)
        self._latest_analysis_id = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="chessy-analyzer", daemon=True)
        self._thread.start()

    @property
    def latest_id(self) -> int:
        return self._latest_analysis_id

    def submit(self, job: AnalysisJob) -> int:
        self._latest_analysis_id = job.analysis_id
        self._enqueue(_WorkItem(analysis=job))
        return job.analysis_id

    def _enqueue(self, item: _WorkItem) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def cancel(self) -> int:
        self._latest_analysis_id += 1
        self._analyzer.interrupt()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        return self._latest_analysis_id

    def _loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            self._run_analysis(item.analysis)

    def _run_analysis(self, job: AnalysisJob) -> None:
        if job.analysis_id != self._latest_analysis_id:
            return
        try:
            log.info("worker analysis[%s] start", job.analysis_id)
            if job.precomputed_vision is not None:
                log.info("worker analysis[%s] using precomputed vision", job.analysis_id)
                result = self._analyzer.analyze_precomputed(
                    job.image,
                    job.precomputed_vision,
                    side=job.side,
                    show_predictions=job.show_predictions,
                    human_mode=job.human_mode,
                    target_elo=job.target_elo,
                    on_thinking=job.on_thinking,
                    analysis_id=job.analysis_id,
                    is_cancelled=lambda: job.analysis_id != self._latest_analysis_id,
                    engine_mode=job.engine_mode,
                )
            else:
                result = self._analyzer.analyze(
                    job.image,
                    side=job.side,
                    show_predictions=job.show_predictions,
                    human_mode=job.human_mode,
                    target_elo=job.target_elo,
                    on_thinking=job.on_thinking,
                    analysis_id=job.analysis_id,
                    is_cancelled=lambda: job.analysis_id != self._latest_analysis_id,
                    engine_mode=job.engine_mode,
                )
        except Exception as exc:
            log.exception("Analysis worker error")
            result = AnalysisOutput(
                fen="",
                best_move_san="",
                best_move_uci="",
                evaluation="",
                vision_ms=0.0,
                engine_ms=0.0,
                total_ms=0.0,
                error=str(exc),
            )
        if job.analysis_id == self._latest_analysis_id:
            log.info("worker analysis[%s] complete", job.analysis_id)
            job.on_complete(result)

    def shutdown(self) -> None:
        self._running = False
        self._analyzer.interrupt()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
