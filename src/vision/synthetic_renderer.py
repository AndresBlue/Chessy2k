"""Synthetic board rendering helpers used by vision tests."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import chess
import chess.svg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.chess_core.fen_utils import board_matrix_from_board, PIECE_TO_CLASS
from src.vision.chesscom_pieces import load_piece
from src.vision.chesscom_theme import (
    CHESSCOM_SVG_COLORS,
    CHESSCOM_SVG_STYLE,
    THEME_SVG_COLORS,
)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass
class BoardTheme:
    name: str
    light_square: tuple[int, int, int]
    dark_square: tuple[int, int, int]
    margin_color: tuple[int, int, int] = (40, 40, 40)
    coord_color: tuple[int, int, int] = (200, 200, 200)
    piece_style: str = "svg"  # svg | unicode (unicode kept for debugging only)
    inline_coords: bool = False


THEMES: dict[str, BoardTheme] = {
    "chesscom": BoardTheme(
        "chesscom",
        _hex_to_rgb("#eeeed2"),
        _hex_to_rgb("#769656"),
        margin_color=_hex_to_rgb("#eeeed2"),
        coord_color=_hex_to_rgb("#769656"),
        piece_style="chesscom",
        inline_coords=True,
    ),
    "lichess": BoardTheme(
        "lichess",
        _hex_to_rgb("#f0d9b5"),
        _hex_to_rgb("#b58863"),
        margin_color=_hex_to_rgb("#312e2b"),
        coord_color=_hex_to_rgb("#b58863"),
        piece_style="svg",
    ),
    "green": BoardTheme(
        "green",
        (240, 217, 181),
        (181, 136, 99),
        piece_style="svg",
    ),
    "wood": BoardTheme(
        "wood",
        (222, 184, 135),
        (139, 90, 43),
        piece_style="svg",
    ),
    "blue": BoardTheme(
        "blue",
        (222, 227, 230),
        (140, 162, 173),
        piece_style="svg",
    ),
    "gray": BoardTheme(
        "gray",
        (200, 200, 200),
        (120, 120, 120),
        piece_style="svg",
    ),
}

UNICODE_PIECES = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}


CHESSCOM_LEGAL_DOT_RGBA = (246, 246, 105, 145)
CHESSCOM_LEGAL_RING_RGBA = (186, 202, 68, 210)


def last_move_squares(board: chess.Board) -> tuple[str, str] | None:
    """Origin and destination of the last move (requires move stack)."""
    if not board.move_stack:
        return None
    move = board.peek()
    return chess.square_name(move.from_square), chess.square_name(move.to_square)


def random_training_board(
    rng: random.Random,
    *,
    min_moves: int = 1,
    max_moves: int = 80,
) -> chess.Board:
    """Random reachable position; keeps move stack for last-move highlights."""
    board = chess.Board()
    num_moves = rng.randint(min_moves, max_moves)
    for _ in range(num_moves):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def sample_opponent_last_move_highlight(
    board: chess.Board,
    rng: random.Random,
    *,
    p_show: float = 0.92,
) -> tuple[str, str] | None:
    """
    Yellow last-move tint on opponent origin + destination (Chess.com).

    Matches overlay use: opponent already moved, user is thinking (no legal-move hints).
    Works for either side to move (user white or black).
    """
    last_move = last_move_squares(board)
    if last_move is None:
        return None
    if rng.random() < p_show:
        return last_move
    return None


class SyntheticBoardRenderer:
    """Render digital chess boards from FEN strings."""

    def __init__(
        self,
        board_size: int = 512,
        theme: BoardTheme | str = "chesscom",
        show_coords: bool = True,
        margin: int = 0,
        inline_coords: bool | None = None,
    ):
        self.board_size = board_size
        self.margin = margin
        if isinstance(theme, str):
            theme = THEMES.get(theme, THEMES["chesscom"])
        self.theme = theme
        self.show_coords = show_coords
        self.inline_coords = theme.inline_coords if inline_coords is None else inline_coords
        self.cell_size = board_size // 8

    def render(self, fen: str, flip: bool = False) -> np.ndarray:
        """Render FEN to BGR numpy image."""
        return self._to_bgr(self._render_pil(fen, flip=flip))

    def render_svg(self, fen: str, flip: bool = False) -> np.ndarray:
        """Alias for render — SVG is the default renderer."""
        return self.render(fen, flip=flip)

    def render_with_highlights(
        self,
        fen: str,
        last_move: tuple[str, str] | None = None,
        marked_squares: list[str] | None = None,
        legal_move_squares: list[str] | None = None,
        flip: bool = False,
    ) -> np.ndarray:
        """Render board with last-move tint and legal-move destination markers."""
        return self._to_bgr(
            self._render_pil(
                fen,
                flip=flip,
                last_move=last_move,
                marked_squares=marked_squares,
                legal_move_squares=legal_move_squares,
            )
        )

    def _render_pil(
        self,
        fen: str,
        flip: bool = False,
        last_move: tuple[str, str] | None = None,
        marked_squares: list[str] | None = None,
        legal_move_squares: list[str] | None = None,
    ) -> Image.Image:
        if self.theme.piece_style == "chesscom":
            img = self._render_chesscom_sprites(
                fen,
                flip=flip,
                last_move=last_move,
                marked_squares=marked_squares,
                legal_move_squares=legal_move_squares,
            )
        elif self.theme.piece_style == "unicode":
            img = self._render_unicode(fen, flip=flip)
        else:
            img = self._render_svg_image(
                fen,
                flip=flip,
                last_move=last_move,
                marked_squares=marked_squares,
                legal_move_squares=legal_move_squares,
            )

        if self.show_coords and self.inline_coords:
            self._draw_inline_coords(img, flip)

        if self.margin > 0:
            padded = Image.new("RGB", (self.board_size + 2 * self.margin, self.board_size + 2 * self.margin), self.theme.margin_color)
            padded.paste(img, (self.margin, self.margin))
            img = padded

        return img

    def _render_svg_image(
        self,
        fen: str,
        flip: bool = False,
        last_move: tuple[str, str] | None = None,
        marked_squares: list[str] | None = None,
        legal_move_squares: list[str] | None = None,
    ) -> Image.Image:
        board = chess.Board(fen)
        colors = dict(THEME_SVG_COLORS.get(self.theme.name, {}))
        style = CHESSCOM_SVG_STYLE if self.theme.name == "chesscom" else ""

        move_obj = None
        if last_move:
            try:
                move_obj = chess.Move.from_uci(last_move[0] + last_move[1])
            except ValueError:
                move_obj = None

        last_squares = set(last_move or [])
        fill: dict[chess.Square, str] = {}
        if marked_squares:
            for sq_name in marked_squares:
                fill[chess.parse_square(sq_name)] = "#00c8ff80"
        for sq_name in legal_move_squares or []:
            if sq_name in last_squares:
                continue
            fill[chess.parse_square(sq_name)] = "#f6f66999"

        board_kwargs: dict = {
            "size": self.board_size,
            "flipped": flip,
            "coordinates": not self.inline_coords and self.show_coords,
            "lastmove": move_obj,
        }
        if colors:
            board_kwargs["colors"] = colors
        if style:
            board_kwargs["style"] = style
        if fill:
            board_kwargs["fill"] = fill

        svg_bytes = chess.svg.board(board, **board_kwargs)

        try:
            import cairosvg

            png_bytes = cairosvg.svg2png(bytestring=svg_bytes.encode(), output_width=self.board_size)
            return Image.open(io.BytesIO(png_bytes)).convert("RGB")
        except (OSError, ImportError):
            return self._render_unicode(fen, flip=flip)

    def _render_chesscom_sprites(
        self,
        fen: str,
        flip: bool = False,
        last_move: tuple[str, str] | None = None,
        marked_squares: list[str] | None = None,
        legal_move_squares: list[str] | None = None,
    ) -> Image.Image:
        board = chess.Board(fen)
        img = Image.new("RGB", (self.board_size, self.board_size), self.theme.light_square)
        highlight_light = _hex_to_rgb("#f6f669")
        highlight_dark = _hex_to_rgb("#baca44")
        last_squares = set(last_move) if last_move else set()
        marked = set(marked_squares or [])
        legal_dests = set(legal_move_squares or [])
        piece_px = int(self.cell_size * 0.92)

        for screen_rank in range(8):
            for screen_file in range(8):
                board_rank = 7 - screen_rank if not flip else screen_rank
                board_file = screen_file if not flip else 7 - screen_file
                is_light = (board_rank + board_file) % 2 == 0
                base = self.theme.light_square if is_light else self.theme.dark_square

                x0 = screen_file * self.cell_size
                y0 = screen_rank * self.cell_size
                square_name = chess.square_name(chess.square(board_file, board_rank))

                if square_name in last_squares:
                    color = highlight_light if is_light else highlight_dark
                elif square_name in marked:
                    color = tuple(min(255, c + 40) for c in base)
                else:
                    color = base

                tile = Image.new("RGB", (self.cell_size, self.cell_size), color)
                img.paste(tile, (x0, y0))

                piece = board.piece_at(chess.square(board_file, board_rank))
                if piece:
                    sprite = load_piece(piece.symbol(), piece_px)
                    px = x0 + (self.cell_size - sprite.width) // 2
                    py = y0 + (self.cell_size - sprite.height) // 2
                    img.paste(sprite, (px, py), sprite)

                if square_name in legal_dests and square_name not in last_squares:
                    self._draw_legal_move_marker(
                        img, x0, y0, self.cell_size, occupied=piece is not None
                    )

        return img

    @staticmethod
    def _draw_legal_move_marker(
        img: Image.Image,
        x0: int,
        y0: int,
        cell: int,
        *,
        occupied: bool,
    ) -> None:
        """Chess.com-style yellow dot / capture ring on legal destinations."""
        overlay = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if occupied:
            margin = cell // 8
            draw.ellipse(
                [margin, margin, cell - margin, cell - margin],
                outline=CHESSCOM_LEGAL_RING_RGBA,
                width=max(3, cell // 16),
            )
        else:
            radius = max(4, cell // 7)
            cx, cy = cell // 2, cell // 2
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=CHESSCOM_LEGAL_DOT_RGBA,
            )
        base = img.crop((x0, y0, x0 + cell, y0 + cell)).convert("RGBA")
        composed = Image.alpha_composite(base, overlay)
        img.paste(composed.convert("RGB"), (x0, y0))

    def _render_unicode(self, fen: str, flip: bool = False) -> Image.Image:
        board = chess.Board(fen)
        img = Image.new("RGB", (self.board_size, self.board_size), self.theme.margin_color)
        draw = ImageDraw.Draw(img)

        for rank in range(8):
            for file in range(8):
                display_rank = 7 - rank if not flip else rank
                display_file = file if not flip else 7 - file
                is_light = (display_rank + display_file) % 2 == 0
                color = self.theme.light_square if is_light else self.theme.dark_square
                x0 = file * self.cell_size
                y0 = rank * self.cell_size
                draw.rectangle([x0, y0, x0 + self.cell_size, y0 + self.cell_size], fill=color)

                square = chess.square(display_file, display_rank)
                piece = board.piece_at(square)
                if piece:
                    self._draw_piece(draw, piece.symbol(), x0, y0)

        if self.show_coords and not self.inline_coords:
            self._draw_margin_coords(draw, 0, flip)

        return img

    def _draw_piece(self, draw: ImageDraw.ImageDraw, symbol: str, x: int, y: int) -> None:
        char = UNICODE_PIECES.get(symbol, symbol)
        color = (255, 255, 255) if symbol.isupper() else (30, 30, 30)
        outline = (0, 0, 0) if symbol.isupper() else (255, 255, 255)
        font_size = int(self.cell_size * 0.75)
        try:
            font = ImageFont.truetype("segoeui.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x + (self.cell_size - tw) // 2
        ty = y + (self.cell_size - th) // 2
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + dx, ty + dy), char, fill=outline, font=font)
        draw.text((tx, ty), char, fill=color, font=font)

    def _draw_margin_coords(self, draw: ImageDraw.ImageDraw, offset: int, flip: bool) -> None:
        files = "abcdefgh"
        try:
            font = ImageFont.truetype("arial.ttf", max(10, self.cell_size // 6))
        except OSError:
            font = ImageFont.load_default()
        for i in range(8):
            file_char = files[i]
            rank_char = str(8 - i if not flip else i + 1)
            draw.text((offset + i * self.cell_size + 2, offset + 2), file_char, fill=self.theme.coord_color, font=font)
            draw.text(
                (offset + 2, offset + i * self.cell_size + self.cell_size - 14),
                rank_char,
                fill=self.theme.coord_color,
                font=font,
            )

    def _draw_inline_coords(self, img: Image.Image, flip: bool) -> None:
        """Chess.com-style coordinates inside edge squares."""
        draw = ImageDraw.Draw(img)
        files = "abcdefgh"
        font_size = max(9, self.cell_size // 7)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        for i in range(8):
            file_idx = i if not flip else 7 - i
            rank_idx = 7 - i if not flip else i
            file_char = files[file_idx]
            rank_char = str(i + 1 if flip else 8 - i)

            # Files on bottom rank (screen row 7)
            bottom_y = 7 * self.cell_size
            bx0 = i * self.cell_size
            is_light_file = (7 + file_idx) % 2 == 0
            file_color = self.theme.dark_square if is_light_file else self.theme.light_square
            fbbox = draw.textbbox((0, 0), file_char, font=font)
            fw = fbbox[2] - fbbox[0]
            fh = fbbox[3] - fbbox[1]
            draw.text(
                (bx0 + self.cell_size - fw - 3, bottom_y + self.cell_size - fh - 2),
                file_char,
                fill=file_color,
                font=font,
            )

            # Ranks on a-file (screen column 0)
            left_x = 0
            ry0 = i * self.cell_size
            is_light_rank = (rank_idx + 0) % 2 == 0
            rank_color = self.theme.dark_square if is_light_rank else self.theme.light_square
            draw.text((left_x + 3, ry0 + 2), rank_char, fill=rank_color, font=font)

    @staticmethod
    def _to_bgr(img: Image.Image) -> np.ndarray:
        return np.array(img)[:, :, ::-1]


def random_fen(rng: random.Random | None = None) -> str:
    """Generate a random legal-ish FEN for synthetic boards in tests."""
    rng = rng or random.Random()
    return random_training_board(rng, min_moves=0).fen()
