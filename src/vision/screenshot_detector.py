"""OpenCV-based board region detection for digital screenshots."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoardDetection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    corners: np.ndarray | None  # 4x2 float32
    confidence: float
    method: str


def detect_board(image: np.ndarray) -> BoardDetection:
    """
    Detect chess board region in a digital screenshot.

    Strategy:
    1. Find largest near-square contour
    2. Fallback: detect 8x8 alternating color grid
    3. Fallback: use central square region
    """
    h, w = image.shape[:2]

    result = _detect_by_contour(image)
    if result is not None:
        return result

    result = _detect_by_grid(image)
    if result is not None:
        return result

    # Fallback: assume board is central 80% square
    size = min(h, w)
    margin_x = (w - size) // 2
    margin_y = (h - size) // 2
    return BoardDetection(
        bbox=(margin_x, margin_y, size, size),
        corners=None,
        confidence=0.3,
        method="center_fallback",
    )


def _detect_by_contour(image: np.ndarray) -> BoardDetection | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = image.shape[0] * image.shape[1]
    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        aspect = bw / bh if bh > 0 else 0
        if 0.85 <= aspect <= 1.15:
            squareness = 1.0 - abs(1.0 - aspect)
            score = (area / img_area) * squareness
            if score > best_score:
                best_score = score
                corners = approx.reshape(4, 2).astype(np.float32)
                best = BoardDetection(
                    bbox=(x, y, bw, bh),
                    corners=corners,
                    confidence=min(0.95, 0.5 + score),
                    method="contour",
                )

    return best


def _detect_by_grid(image: np.ndarray) -> BoardDetection | None:
    """Detect board by finding region with strong 8x8 checkerboard pattern."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Scan for square regions with high local variance (checkerboard)
    best_score = 0.0
    best_bbox = None

    for frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        size = int(min(h, w) * frac)
        if size < 100:
            continue
        for y0 in range(0, h - size + 1, max(1, (h - size) // 4 or 1)):
            for x0 in range(0, w - size + 1, max(1, (w - size) // 4 or 1)):
                patch = gray[y0 : y0 + size, x0 : x0 + size]
                score = _checkerboard_score(patch)
                if score > best_score:
                    best_score = score
                    best_bbox = (x0, y0, size, size)

    if best_bbox and best_score > 0.15:
        x, y, bw, bh = best_bbox
        return BoardDetection(
            bbox=(x, y, bw, bh),
            corners=None,
            confidence=min(0.9, 0.4 + best_score),
            method="grid",
        )
    return None


def _checkerboard_score(gray: np.ndarray) -> float:
    """Score how checkerboard-like a patch is."""
    h, w = gray.shape
    cell_h = h // 8
    cell_w = w // 8
    if cell_h < 4 or cell_w < 4:
        return 0.0

    means = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            cell = gray[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            means[r, c] = cell.mean()

    alternating_diff = 0.0
    count = 0
    for r in range(8):
        for c in range(8):
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if nr < 8 and nc < 8:
                    alternating_diff += abs(means[r, c] - means[nr, nc])
                    count += 1

    return alternating_diff / max(count, 1) / 255.0
