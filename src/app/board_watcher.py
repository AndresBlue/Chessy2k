"""Detect board changes and wait for animations to settle before analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

PollKind = Literal["none", "motion", "ready"]


@dataclass
class PollResult:
    """Result of polling a new capture against the stable board reference."""

    kind: PollKind
    image: np.ndarray | None = None


class BoardStableDetector:
    """
    Trigger analysis only after the board image stops changing.

    Chess sites animate piece moves. This detector compares against the last
    *settled* frame, requires evidence of a real move (typically two squares),
    then waits for consecutive stable frames before emitting ``ready``.
    """

    def __init__(
        self,
        mean_diff: float = 8.0,
        std_diff: float = 5.0,
        min_changed_cells: int = 2,
        min_peak_diff: float = 18.0,
        single_cell_peak: float = 38.0,
        settle_frames: int = 2,
        settle_max_diff: float = 4.5,
        settle_timeout_s: float = 1.2,
        fingerprint_size: int = 128,
        cell_inner_ratio: float = 0.6,
    ):
        self.mean_diff = mean_diff
        self.std_diff = std_diff
        self.min_changed_cells = min_changed_cells
        self.min_peak_diff = min_peak_diff
        self.single_cell_peak = single_cell_peak
        self.settle_frames = settle_frames
        self.settle_max_diff = settle_max_diff
        self.settle_timeout_s = settle_timeout_s
        self.fingerprint_size = fingerprint_size
        self.cell_inner_ratio = cell_inner_ratio

        self._stable_ref: np.ndarray | None = None
        self._last_fp: np.ndarray | None = None
        self._state: Literal["idle", "motion"] = "idle"
        self._motion_started = 0.0
        self._settle_streak = 0
        self._candidate_image: np.ndarray | None = None

    @property
    def poll_interval_ms(self) -> int:
        return 60 if self._state == "motion" else 140

    @property
    def is_settling(self) -> bool:
        return self._state == "motion"

    @property
    def has_baseline(self) -> bool:
        return self._stable_ref is not None

    def reset(self) -> None:
        self._stable_ref = None
        self._last_fp = None
        self._state = "idle"
        self._motion_started = 0.0
        self._settle_streak = 0
        self._candidate_image = None

    def seed(self, image_bgr: np.ndarray) -> None:
        """Store baseline without triggering analysis (call when enabling auto)."""
        fp = self._fingerprint(image_bgr)
        self._stable_ref = fp
        self._last_fp = fp
        self._state = "idle"
        self._motion_started = 0.0
        self._settle_streak = 0
        self._candidate_image = None

    def mark_analyzed(self, image_bgr: np.ndarray) -> None:
        """Sync stable reference after a successful board analysis."""
        self.seed(image_bgr)

    def discard_pending(self) -> None:
        """Allow re-detection after a failed analysis without moving the reference."""
        self._state = "idle"
        self._settle_streak = 0
        self._candidate_image = None

    def poll(self, image_bgr: np.ndarray, now: float | None = None) -> PollResult:
        now = now or time.perf_counter()
        fp = self._fingerprint(image_bgr)

        if self._stable_ref is None:
            self.seed(image_bgr)
            return PollResult(kind="none")

        if self._state == "idle":
            if not self._is_move_change(self._stable_ref, fp):
                self._last_fp = fp
                return PollResult(kind="none")
            self._state = "motion"
            self._motion_started = now
            self._settle_streak = 0
            self._candidate_image = image_bgr.copy()
            self._last_fp = fp
            return PollResult(kind="motion")

        self._candidate_image = image_bgr.copy()
        if self._is_settled(self._last_fp, fp):
            self._settle_streak += 1
        else:
            self._settle_streak = 0
        self._last_fp = fp

        timed_out = (now - self._motion_started) >= self.settle_timeout_s
        if self._settle_streak >= self.settle_frames:
            return self._finish_motion(self._candidate_image)

        if timed_out:
            if self._is_move_change(
                self._stable_ref, self._fingerprint(self._candidate_image)
            ):
                return self._finish_motion(self._candidate_image)
            self._cancel_motion()
            return PollResult(kind="none")

        return PollResult(kind="motion")

    def _finish_motion(self, image_bgr: np.ndarray) -> PollResult:
        self._state = "idle"
        self._settle_streak = 0
        self._motion_started = 0.0
        self._candidate_image = None
        ready_fp = self._fingerprint(image_bgr)
        self._stable_ref = ready_fp
        self._last_fp = ready_fp
        return PollResult(kind="ready", image=image_bgr.copy())

    def _cancel_motion(self) -> None:
        self._state = "idle"
        self._settle_streak = 0
        self._motion_started = 0.0
        self._candidate_image = None

    def _fingerprint(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        size = self.fingerprint_size
        small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        grid = size // 8
        margin = max(1, int(grid * (1.0 - self.cell_inner_ratio) / 2))
        features: list[float] = []
        for rank in range(8):
            for file in range(8):
                patch = small[
                    rank * grid + margin : (rank + 1) * grid - margin,
                    file * grid + margin : (file + 1) * grid - margin,
                ]
                if patch.size:
                    features.append(float(patch.mean()))
                    features.append(float(patch.std()))
                else:
                    features.extend((0.0, 0.0))
        return np.array(features, dtype=np.float32)

    def _cell_diffs(self, ref: np.ndarray, cur: np.ndarray) -> tuple[int, float]:
        if ref.shape != cur.shape:
            return 0, 0.0
        n_cells = len(ref) // 2
        changed = 0
        peak = 0.0
        for i in range(n_cells):
            mean_delta = abs(ref[i * 2] - cur[i * 2])
            std_delta = abs(ref[i * 2 + 1] - cur[i * 2 + 1])
            cell_peak = max(mean_delta, std_delta * 2.5)
            peak = max(peak, cell_peak)
            if mean_delta > self.mean_diff or std_delta > self.std_diff:
                changed += 1
        return changed, peak

    def _is_move_change(self, ref: np.ndarray, cur: np.ndarray) -> bool:
        changed, peak = self._cell_diffs(ref, cur)
        if changed >= self.min_changed_cells and peak >= self.min_peak_diff:
            return True
        return changed >= 1 and peak >= self.single_cell_peak

    def _is_settled(self, ref: np.ndarray, cur: np.ndarray) -> bool:
        if ref is None:
            return False
        return float(np.abs(ref - cur).max()) < self.settle_max_diff


BoardChangeDetector = BoardStableDetector
