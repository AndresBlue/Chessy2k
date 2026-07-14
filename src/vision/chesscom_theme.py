"""Chess.com Neo board theme — colors and SVG styling."""

from __future__ import annotations

# Official Chess.com Neo / default green board (from chess.com play page)
CHESSCOM_LIGHT = "#eeeed2"
CHESSCOM_DARK = "#769656"
CHESSCOM_HIGHLIGHT = "#f6f66980"  # last-move tint (semi-transparent yellow-green)
CHESSCOM_CHECK = "#ff000080"

CHESSCOM_SVG_COLORS: dict[str, str] = {
    "square light": CHESSCOM_LIGHT,
    "square dark": CHESSCOM_DARK,
    "square light lastmove": "#f6f669",
    "square dark lastmove": "#baca44",
    "margin": CHESSCOM_LIGHT,
    "coord": "#769656",
}

# Neo-style piece strokes (closer to Chess.com than default black-only stroke)
CHESSCOM_SVG_STYLE = """
.white { fill: #ffffff; stroke: #6b6b6b; stroke-width: 1.2; }
.black { fill: #3d3d3d; stroke: #1a1a1a; stroke-width: 1.2; }
.white.pawn path { fill: #ffffff; stroke: #6b6b6b; }
.black.pawn path { fill: #3d3d3d; stroke: #1a1a1a; }
"""

LICHESS_SVG_COLORS: dict[str, str] = {
    "square light": "#f0d9b5",
    "square dark": "#b58863",
    "square light lastmove": "#cdd16a",
    "square dark lastmove": "#aaa23b",
    "coord": "#b58863",
}

THEME_SVG_COLORS: dict[str, dict[str, str]] = {
    "chesscom": CHESSCOM_SVG_COLORS,
    "lichess": LICHESS_SVG_COLORS,
    "green": {
        "square light": "#f0d9b5",
        "square dark": "#b58863",
        "coord": "#b58863",
    },
    "wood": {
        "square light": "#deb887",
        "square dark": "#8b5a2b",
        "coord": "#8b5a2b",
    },
    "blue": {
        "square light": "#dee3e6",
        "square dark": "#8ca2ad",
        "coord": "#8ca2ad",
    },
    "gray": {
        "square light": "#c8c8c8",
        "square dark": "#787878",
        "coord": "#787878",
    },
}
