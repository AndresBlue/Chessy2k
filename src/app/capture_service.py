"""Dedicated screen-capture worker thread (mss stays on one thread)."""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from src.app.errors import CaptureError
from src.app.logging_config import get_logger
from src.app.screen_capture import ScreenRegion, capture_region, capture_virtual_desktop

log = get_logger(__name__)

CallbackResult = tuple[Callable[..., None], object]


class _JobKind(Enum):
    REGION = "region"
    DESKTOP = "desktop"
    STOP = "stop"


@dataclass
class _Job:
    kind: _JobKind
    region: ScreenRegion | None = None
    request_id: str = ""


@dataclass
class CaptureResult:
    image: np.ndarray | None
    origin_x: int = 0
    origin_y: int = 0
    error: str | None = None


class CaptureService:
    """Serializes all mss captures on a single background thread."""

    def __init__(self, result_queue: queue.Queue[CallbackResult] | None = None) -> None:
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._pending: dict[str, queue.Queue[CaptureResult]] = {}
        self._lock = threading.Lock()
        self._running = True
        self._result_queue = result_queue
        self._thread = threading.Thread(target=self._worker, name="chessy-capture", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while self._running:
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if job.kind == _JobKind.STOP:
                break
            result = self._execute(job)
            with self._lock:
                waiter = self._pending.pop(job.request_id, None)
            if waiter is not None:
                waiter.put(result)

    def _deliver(self, callback: Callable[..., None], result: CaptureResult) -> None:
        """Push callback+result to the UI queue; never touch Tk from worker threads."""
        if self._result_queue is not None:
            self._result_queue.put((callback, result))
            return
        callback(result)

    def _execute(self, job: _Job) -> CaptureResult:
        try:
            if job.kind == _JobKind.REGION:
                assert job.region is not None
                image = capture_region(job.region)
                return CaptureResult(image=image)
            if job.kind == _JobKind.DESKTOP:
                image, ox, oy = capture_virtual_desktop()
                return CaptureResult(image=image, origin_x=ox, origin_y=oy)
        except CaptureError as exc:
            log.warning("Capture failed: %s", exc)
            return CaptureResult(image=None, error=str(exc))
        except Exception as exc:
            log.exception("Unexpected capture error")
            return CaptureResult(image=None, error=str(exc))
        return CaptureResult(image=None, error="Unknown capture job")

    def _submit(self, job: _Job, timeout: float = 10.0) -> CaptureResult:
        waiter: queue.Queue[CaptureResult] = queue.Queue(maxsize=1)
        job.request_id = str(uuid.uuid4())
        with self._lock:
            self._pending[job.request_id] = waiter
        self._queue.put(job)
        try:
            return waiter.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                self._pending.pop(job.request_id, None)
            return CaptureResult(image=None, error="Capture timed out")

    def capture_region_async(
        self,
        region: ScreenRegion,
        callback: Callable[[CaptureResult], None],
        *,
        timeout: float = 10.0,
    ) -> None:
        """Run capture on worker thread; deliver result via the UI result queue."""

        def _run() -> None:
            result = self._submit(_Job(_JobKind.REGION, region=region), timeout=timeout)
            self._deliver(callback, result)

        threading.Thread(target=_run, name="chessy-capture-dispatch", daemon=True).start()

    def capture_desktop_async(
        self,
        callback: Callable[[CaptureResult], None],
        *,
        timeout: float = 15.0,
    ) -> None:
        def _run() -> None:
            result = self._submit(_Job(_JobKind.DESKTOP), timeout=timeout)
            self._deliver(callback, result)

        threading.Thread(target=_run, name="chessy-capture-dispatch", daemon=True).start()

    def capture_region_sync(self, region: ScreenRegion, timeout: float = 10.0) -> CaptureResult:
        """Blocking capture routed through the worker (safe from any thread)."""
        return self._submit(_Job(_JobKind.REGION, region=region), timeout=timeout)

    def shutdown(self) -> None:
        self._running = False
        self._queue.put(_Job(_JobKind.STOP))
        self._thread.join(timeout=2.0)
