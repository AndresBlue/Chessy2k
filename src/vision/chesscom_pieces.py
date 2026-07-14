"""Chess.com Neo piece sprites — download once, cache locally."""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

from PIL import Image

PIECE_CODES = {
    "K": "wk",
    "Q": "wq",
    "R": "wr",
    "B": "wb",
    "N": "wn",
    "P": "wp",
    "k": "bk",
    "q": "bq",
    "r": "br",
    "b": "bb",
    "n": "bn",
    "p": "bp",
}

CDN_BASE = "https://images.chesscomfiles.com/chess-themes/pieces/neo/150"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "pieces" / "neo"


def ensure_pieces_cached() -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for code in PIECE_CODES.values():
        path = ASSETS_DIR / f"{code}.png"
        if not path.exists():
            url = f"{CDN_BASE}/{code}.png"
            urllib.request.urlretrieve(url, path)
    return ASSETS_DIR


@lru_cache(maxsize=32)
def load_piece(symbol: str, target_px: int) -> Image.Image:
    """Load and scale a Neo piece sprite."""
    ensure_pieces_cached()
    code = PIECE_CODES[symbol]
    path = ASSETS_DIR / f"{code}.png"
    img = Image.open(path).convert("RGBA")
    if target_px != img.width:
        img = img.resize((target_px, target_px), Image.Resampling.LANCZOS)
    return img
