"""End-to-end vision pipeline: screenshot to FEN."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.app.logging_config import get_logger

from src.vision.board_cropper import crop_board
from src.vision.fen_builder import build_fen
from src.vision.piece_classifier import PieceClassifier
from src.vision.screenshot_detector import BoardDetection, detect_board
from src.vision.square_preprocess import crop_square_center, mask_chesscom_coordinates
from src.vision.square_segmenter import segment_squares
from src.vision.visualizer import draw_classification_overlay

log = get_logger(__name__)


@dataclass
class VisionResult:
    board_matrix: list[list[str | None]]
    fen_pieces: str
    orientation: str
    confidence: np.ndarray
    ambiguous_squares: list[tuple[int, int]]
    debug_image: np.ndarray | None
    cropped_board: np.ndarray
    time_ms: float
    detection_method: str
    board_bbox: tuple[int, int, int, int]
    repaired: bool = False
    repair_notes: list[str] | None = None
    device: str = "cpu"
    device_label: str = "CPU"
    inference_mode: str = "full"
    squares_updated: int = 64


class VisionPipeline:
    """Screenshot → FEN vision pipeline."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        architecture: str = "resnet18",
        device: str | None = None,
        *,
        repair_fen: bool = False,
        incremental: bool = False,
        preprocess_squares: bool = False,
    ):
        self.classifier = PieceClassifier(
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            device=device,
        )
        self.repair_fen = repair_fen
        self.incremental = incremental
        self.preprocess_squares = preprocess_squares
        self._last_squares: np.ndarray | None = None
        self._last_pred: dict | None = None

    @classmethod
    def from_config(cls, config: dict | None = None, checkpoint_path: str | None = None) -> "VisionPipeline":
        """Build pipeline using ``config['vision']`` inference flags."""
        vision_cfg = (config or {}).get("vision", {})
        ckpt = checkpoint_path or vision_cfg.get("checkpoint")
        return cls(
            checkpoint_path=ckpt,
            architecture=vision_cfg.get("architecture", "resnet18"),
            repair_fen=bool(vision_cfg.get("repair_fen", False)),
            incremental=bool(vision_cfg.get("incremental", False)),
            preprocess_squares=bool(vision_cfg.get("preprocess_squares", False)),
        )

    @property
    def device_label(self) -> str:
        return self.classifier.device_label

    def reset_cache(self) -> None:
        """Clear incremental inference cache (e.g. after region change)."""
        self._last_squares = None
        self._last_pred = None

    def warmup(self) -> None:
        self.classifier.warmup()

    def process(
        self,
        image_path: str,
        side: str = "white",
        *,
        fast_mode: bool = False,
        incremental: bool | None = None,
    ) -> VisionResult:
        """Process screenshot and return vision result."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self._process_bgr(
            image,
            side=side,
            fast_mode=fast_mode,
            incremental=self.incremental if incremental is None else incremental,
        )

    def process_image(
        self,
        image: np.ndarray,
        side: str = "white",
        *,
        fast_mode: bool = False,
        incremental: bool | None = None,
    ) -> VisionResult:
        """Process in-memory BGR image without writing temp files."""
        if image is None or image.size == 0:
            raise ValueError("Empty image")
        return self._process_bgr(
            image,
            side=side,
            fast_mode=fast_mode,
            incremental=self.incremental if incremental is None else incremental,
        )

    @staticmethod
    def _detect_board_region(image: np.ndarray) -> BoardDetection:
        h, w = image.shape[:2]
        aspect = w / h if h > 0 else 0
        if min(h, w) >= 96 and 0.75 <= aspect <= 1.33:
            size = min(h, w)
            mx = (w - size) // 2
            my = (h - size) // 2
            log.info(
                "vision detect: selected region treated as direct board (%sx%s -> %s)",
                w,
                h,
                size,
            )
            return BoardDetection(
                bbox=(mx, my, size, size),
                corners=None,
                confidence=1.0,
                method="region_direct",
            )
        log.info("vision detect: running board detector (%sx%s aspect=%.2f)", w, h, aspect)
        return detect_board(image)

    def _process_bgr(
        self,
        image: np.ndarray,
        side: str = "white",
        *,
        fast_mode: bool = False,
        incremental: bool = False,
    ) -> VisionResult:
        t0 = time.perf_counter()

        log.info("vision pipeline: detect:start")
        detection = self._detect_board_region(image)
        log.info("vision pipeline: detect:done method=%s bbox=%s", detection.method, detection.bbox)
        log.info("vision pipeline: crop+segment:start")
        cropped = crop_board(image, detection)
        squares = segment_squares(cropped)
        log.info("vision pipeline: crop+segment:done")
        if self.preprocess_squares:
            squares = mask_chesscom_coordinates(squares)
            squares = crop_square_center(squares)

        log.info("vision pipeline: classify:start incremental=%s", incremental)
        if incremental and self._last_squares is not None and self._last_pred is not None:
            pred = self.classifier.predict_squares(
                squares,
                previous_squares=self._last_squares,
                previous_result=self._last_pred,
            )
        else:
            pred = self.classifier.predict_squares(squares)
        log.info(
            "vision pipeline: classify:done mode=%s updated=%s",
            pred.get("inference_mode", "full"),
            pred.get("squares_updated", 64),
        )

        self._last_squares = squares.copy()
        self._last_pred = {
            "board_matrix": pred["board_matrix"],
            "class_matrix": pred["class_matrix"],
            "confidence": pred["confidence"],
            "ambiguous_squares": pred["ambiguous_squares"],
            "probs": pred["probs"].copy(),
            "inference_mode": pred.get("inference_mode", "full"),
            "squares_updated": pred.get("squares_updated", 64),
        }

        board_matrix = pred["board_matrix"]
        repaired = False
        repair_notes: list[str] | None = None
        if self.repair_fen:
            from src.vision.fen_repair import repair_board_fen

            repair = repair_board_fen(pred["board_matrix"], pred["probs"], side_hint=side)
            board_matrix = repair.board_matrix
            repaired = repair.repaired
            repair_notes = repair.notes

        log.info("vision pipeline: fen:start")
        fen_result = build_fen(
            board_matrix=board_matrix,
            confidence=pred["confidence"],
            ambiguous_squares=pred["ambiguous_squares"],
        )
        log.info("vision pipeline: fen:done %s", fen_result.placement)

        debug = None
        if not fast_mode:
            debug = draw_classification_overlay(
                cropped,
                pred["class_matrix"],
                pred["confidence"],
                pred["ambiguous_squares"],
            )

        elapsed = (time.perf_counter() - t0) * 1000

        return VisionResult(
            board_matrix=board_matrix,
            fen_pieces=fen_result.placement,
            orientation=fen_result.orientation,
            confidence=fen_result.confidence,
            ambiguous_squares=fen_result.ambiguous_squares,
            debug_image=debug,
            cropped_board=cropped,
            time_ms=elapsed,
            detection_method=detection.method,
            board_bbox=detection.bbox,
            repaired=repaired,
            repair_notes=repair_notes,
            device=self.classifier.device,
            device_label=self.classifier.device_label,
            inference_mode=pred.get("inference_mode", "full"),
            squares_updated=int(pred.get("squares_updated", 64)),
        )
