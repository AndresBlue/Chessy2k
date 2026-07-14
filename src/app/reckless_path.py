"""Resolve Reckless UCI binary path from env, config, or project folder."""

from __future__ import annotations

from pathlib import Path


def find_reckless_path(root: Path | None = None) -> Path | None:
    """Find Reckless executable under engines/reckless/."""
    root = root or Path(__file__).resolve().parents[2]
    candidates = [
        root / "engines" / "reckless" / "reckless-windows-avx2.exe",
        root / "engines" / "reckless" / "reckless-windows-avx512.exe",
        root / "engines" / "reckless" / "reckless-windows-generic.exe",
        root / "engines" / "reckless" / "reckless.exe",
        root / "engines" / "reckless" / "reckless",
    ]
    for path in candidates:
        if path.exists():
            return path
    reckless_dir = root / "engines" / "reckless"
    if reckless_dir.is_dir():
        for exe in reckless_dir.glob("reckless*.exe"):
            return exe
        for name in ("reckless", "reckless-linux-avx2", "reckless-macos"):
            candidate = reckless_dir / name
            if candidate.exists():
                return candidate
    return None
