"""Resolve runtime paths for CLI and overlay (all paths relative to repo root)."""

from __future__ import annotations

import os
from pathlib import Path

from src.app.config import load_config
from src.app.reckless_path import find_reckless_path
from src.app.stockfish_path import find_stockfish_path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_configured_path(root: Path, configured: str | None) -> Path | None:
    if not configured:
        return None
    candidate = Path(configured)
    path = candidate if candidate.is_absolute() else root / candidate
    return path if path.exists() else None


def resolve_runtime_paths(root: Path | None = None) -> dict[str, Path | None]:
    """Resolve Stockfish, Reckless, and vision checkpoint paths."""
    root = root or project_root()
    cfg = load_config(root / "config" / "default.yaml")
    engines = cfg.get("engines", {})

    stockfish_env = os.environ.get("CHESSY_STOCKFISH_PATH")
    if stockfish_env:
        stockfish = Path(stockfish_env)
    else:
        stockfish = find_stockfish_path(root)
        if stockfish is None:
            stockfish = _resolve_configured_path(
                root, engines.get("stockfish", {}).get("path")
            )

    reckless_env = os.environ.get("CHESSY_RECKLESS_PATH")
    if reckless_env:
        reckless = Path(reckless_env)
    else:
        reckless = find_reckless_path(root)
        if reckless is None:
            reckless = _resolve_configured_path(
                root, engines.get("reckless", {}).get("path")
            )

    classifier_env = os.environ.get("CHESSY_CLASSIFIER_CHECKPOINT")
    if classifier_env:
        classifier = Path(classifier_env)
    else:
        rel = cfg.get("vision", {}).get("checkpoint", "data/checkpoints/vision/best.pt")
        classifier = root / rel

    return {
        "root": root,
        "stockfish": stockfish if stockfish and stockfish.exists() else None,
        "reckless": reckless if reckless and reckless.exists() else None,
        "classifier": classifier if classifier and classifier.exists() else None,
    }
