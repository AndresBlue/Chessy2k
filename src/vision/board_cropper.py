"""Crop and normalize detected board to fixed resolution."""

from __future__ import annotations

import cv2
import numpy as np

from src.vision.screenshot_detector import BoardDetection

BOARD_SIZE = 512


def crop_board(image: np.ndarray, detection: BoardDetection) -> np.ndarray:
    """Crop board region and normalize to BOARD_SIZE x BOARD_SIZE."""
    x, y, w, h = detection.bbox

    if detection.corners is not None and _is_valid_quad(detection.corners):
        return _perspective_crop(image, detection.corners)

    cropped = image[y : y + h, x : x + w]
    if cropped.size == 0:
        raise ValueError("Empty crop region")

    return cv2.resize(cropped, (BOARD_SIZE, BOARD_SIZE), interpolation=cv2.INTER_AREA)


def _is_valid_quad(corners: np.ndarray) -> bool:
    if corners.shape != (4, 2):
        return False
    area = cv2.contourArea(corners.astype(np.float32))
    return area > 100


def _perspective_crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply minimal perspective correction for slightly skewed digital captures."""
    corners = _order_corners(corners)
    dst = np.array(
        [[0, 0], [BOARD_SIZE - 1, 0], [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, matrix, (BOARD_SIZE, BOARD_SIZE))


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order corners: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
