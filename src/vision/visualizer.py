"""Draw debug overlays and move arrows on board images."""

from __future__ import annotations

import cv2
import numpy as np

from src.chess_core.fen_utils import CLASS_TO_PIECE, PIECE_CLASSES, uci_square_to_screen


def draw_vision_predictions_on_capture(
    board_image: np.ndarray,
    board_matrix: list[list[str | None]],
    confidence: np.ndarray,
    ambiguous_squares: list[tuple[int, int]],
    board_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Draw per-square model predictions on a capture or cropped board image."""
    img = board_image.copy()
    rx, ry, rw, rh = board_bbox if board_bbox is not None else (0, 0, img.shape[1], img.shape[0])
    cell = max(8, min(rw, rh) // 8)
    offset_x = rx + (rw - cell * 8) // 2
    offset_y = ry + (rh - cell * 8) // 2
    font = cv2.FONT_HERSHEY_DUPLEX
    ambiguous_set = set(ambiguous_squares)

    for rank in range(8):
        for file in range(8):
            x0 = offset_x + file * cell
            y0 = offset_y + rank * cell
            x1, y1 = x0 + cell, y0 + cell

            conf = float(confidence[rank, file])
            is_ambiguous = (rank, file) in ambiguous_set
            border = (0, 80, 255) if is_ambiguous else (0, 200, 120)
            if conf < 0.65:
                border = (0, 80, 255)

            overlay = img.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), border, -1)
            cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
            cv2.rectangle(img, (x0, y0), (x1, y1), border, 2)

            piece = board_matrix[rank][file]
            piece_label = piece if piece else "."
            font_scale = max(0.55, cell / 72)
            conf_scale = max(0.35, cell / 110)

            piece_y = y0 + int(cell * 0.58)
            conf_y = y0 + int(cell * 0.88)
            _draw_text_outlined(
                img,
                piece_label,
                (x0 + cell // 2 - int(cell * 0.12), piece_y),
                font_scale,
                (255, 255, 255),
            )
            _draw_text_outlined(
                img,
                f"{conf:.2f}",
                (x0 + 4, conf_y),
                conf_scale,
                (220, 220, 220),
            )

    return img


def draw_classification_overlay(
    board_image: np.ndarray,
    class_matrix: np.ndarray,
    confidence: np.ndarray,
    ambiguous_squares: list[tuple[int, int]],
) -> np.ndarray:
    """Draw grid, piece labels, and confidence on board image."""
    img = board_image.copy()
    h, w = img.shape[:2]
    cell_h, cell_w = h // 8, w // 8

    for rank in range(8):
        for file in range(8):
            x0, y0 = file * cell_w, rank * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h

            is_ambiguous = (rank, file) in ambiguous_squares
            color = (0, 0, 255) if is_ambiguous else (0, 255, 0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

            cls = int(class_matrix[rank, file])
            label = PIECE_CLASSES[cls] if cls < len(PIECE_CLASSES) else "?"
            conf = confidence[rank, file]
            text = f"{label}:{conf:.2f}"
            cv2.putText(img, text, (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return img


def _sq_to_center(
    sq: str,
    cell: int,
    orientation: str,
    offset_x: int = 0,
    offset_y: int = 0,
) -> tuple[int, int]:
    screen_rank, screen_file = uci_square_to_screen(sq, orientation)  # type: ignore[arg-type]
    cx = offset_x + screen_file * cell + cell // 2
    cy = offset_y + screen_rank * cell + cell // 2
    return cx, cy


def _draw_text_outlined(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = max(2, int(font_scale * 2))
    x, y = pos
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)):
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_best_move_arrow(
    board_image: np.ndarray,
    uci_move: str,
    orientation: str = "white",
    color: tuple[int, int, int] = (80, 255, 80),
    region: tuple[int, int, int, int] | None = None,
    opacity: float = 1.0,
    highlight_destination: bool = True,
) -> np.ndarray:
    """Draw a high-contrast move arrow on a board image or region within it."""
    img = board_image.copy()
    if len(uci_move) < 4:
        return img

    if region is not None:
        rx, ry, rw, rh = region
        cell = min(rw, rh) // 8
        offset_x, offset_y = rx + (rw - cell * 8) // 2, ry + (rh - cell * 8) // 2
    else:
        h, w = img.shape[:2]
        cell = min(h, w) // 8
        offset_x = (w - cell * 8) // 2
        offset_y = (h - cell * 8) // 2

    from_sq = uci_move[:2]
    to_sq = uci_move[2:4]
    pt1 = _sq_to_center(from_sq, cell, orientation, offset_x, offset_y)
    pt2 = _sq_to_center(to_sq, cell, orientation, offset_x, offset_y)

    thickness = max(4, cell // 14)
    radius = max(6, cell // 5)
    alpha = max(0.0, min(1.0, float(opacity)))
    draw_on = img.copy() if alpha < 1.0 else img

    if highlight_destination:
        to_screen_rank, to_screen_file = uci_square_to_screen(to_sq, orientation)  # type: ignore[arg-type]
        tx0 = offset_x + to_screen_file * cell
        ty0 = offset_y + to_screen_rank * cell
        dest_overlay = draw_on.copy()
        cv2.rectangle(dest_overlay, (tx0, ty0), (tx0 + cell, ty0 + cell), (0, 220, 255), -1)
        cv2.addWeighted(dest_overlay, 0.35, draw_on, 0.65, 0, draw_on)

    cv2.circle(draw_on, pt1, radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(draw_on, pt1, radius, (40, 40, 40), max(2, radius // 4), cv2.LINE_AA)
    cv2.circle(draw_on, pt2, radius + 2, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(draw_on, pt2, radius + 2, (20, 20, 20), max(2, radius // 3), cv2.LINE_AA)

    cv2.arrowedLine(
        draw_on, pt1, pt2, (20, 20, 20), thickness=thickness + 4, tipLength=0.28, line_type=cv2.LINE_AA
    )
    cv2.arrowedLine(
        draw_on, pt1, pt2, color, thickness=thickness, tipLength=0.28, line_type=cv2.LINE_AA
    )

    if alpha < 1.0:
        cv2.addWeighted(draw_on, alpha, img, 1.0 - alpha, 0, img)
    return img


def draw_opponent_move_arrow(
    board_image: np.ndarray,
    uci_move: str,
    orientation: str = "white",
    region: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Draw a semi-transparent red arrow for the predicted opponent reply."""
    return draw_best_move_arrow(
        board_image,
        uci_move,
        orientation=orientation,
        color=(60, 60, 255),
        region=region,
        opacity=0.3,
        highlight_destination=False,
    )
