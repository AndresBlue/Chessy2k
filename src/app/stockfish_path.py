"""Resolve Stockfish binary path from config or project folder."""

from __future__ import annotations

from pathlib import Path


def find_stockfish_path(root: Path | None = None) -> Path | None:
    """Find Stockfish executable under engines/stockfish/ (legacy stockfish/ also checked)."""
    root = root or Path(__file__).resolve().parents[2]
    candidates = [
        root / "engines" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        root / "engines" / "stockfish" / "stockfish-windows-x86-64.exe",
        root / "engines" / "stockfish" / "stockfish.exe",
        root / "engines" / "stockfish" / "stockfish",
        # Legacy layout before engines/ reorganization
        root / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        root / "stockfish" / "stockfish.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    for sf_dir in (root / "engines" / "stockfish", root / "stockfish"):
        if sf_dir.is_dir():
            for exe in sf_dir.glob("stockfish*.exe"):
                return exe
    return None
