"""Smoke tests for Reckless UCI client and path resolution."""

from pathlib import Path

import chess
import pytest

from src.app.reckless_path import find_reckless_path
from src.app.runtime_paths import project_root, resolve_runtime_paths
from src.engines.human_engine import HumanEngine
from src.engines.humanization_config import HumanizationConfig
from src.engines.reckless_client import RecklessClient


def test_find_reckless_path_in_repo():
    path = find_reckless_path(project_root())
    assert path is not None
    assert path.exists()
    assert "reckless" in path.name.lower()


def test_resolve_runtime_paths_includes_reckless():
    paths = resolve_runtime_paths()
    assert paths["reckless"] is not None
    assert paths["reckless"].exists()


@pytest.mark.integration
def test_reckless_uci_handshake_and_legal_move():
    path = find_reckless_path(project_root())
    if path is None:
        pytest.skip("Reckless binary not present")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with RecklessClient(path=str(path), threads=2, hash_mb=64) as engine:
        assert engine.supports_uci_elo is False
        result = engine.analyze(fen, movetime_ms=200, multipv=3)
    board = chess.Board(fen)
    assert result.best_move.uci
    assert chess.Move.from_uci(result.best_move.uci) in board.legal_moves


@pytest.mark.integration
def test_human_engine_over_reckless_returns_legal_move():
    path = find_reckless_path(project_root())
    if path is None:
        pytest.skip("Reckless binary not present")
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    client = RecklessClient(path=str(path), threads=2, hash_mb=64)
    client.start()
    try:
        human = HumanEngine(
            client,
            config=HumanizationConfig(
                enabled=True,
                target_elo=1800,
                use_book=False,
            ),
            project_root=project_root(),
        )
        # Patch opening config to skip book completely
        human.config.opening.max_fullmove = 0
        result = human.analyze(fen, target_elo=1800)
    finally:
        client.stop()
    board = chess.Board(fen)
    assert result.best_move_uci
    assert chess.Move.from_uci(result.best_move_uci) in board.legal_moves
