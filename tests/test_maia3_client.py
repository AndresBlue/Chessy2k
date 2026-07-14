"""Tests for Maia-3 UCI client helpers."""

from pathlib import Path

from src.app.runtime_paths import project_root
from src.engines.maia3_client import build_maia3_command, format_maia3_wdl, is_maia3_installed
from src.engines.stockfish_client import EngineMove


def test_build_maia3_command():
    root = project_root()
    cmd = build_maia3_command(
        root,
        {
            "model": "maia3-23m",
            "use_uci_history": True,
            "cache_dir": "data/maia3/cache",
            "temperature": 0,
            "default_elo": 1800,
            "device": "cuda",
            "use_amp": True,
            "multipv": 5,
        },
    )
    assert "-m" in cmd and "maia3.uci" in cmd
    assert "--model" in cmd and "maia3-23m" in cmd
    assert "--use-uci-history" in cmd
    assert "--temperature" in cmd and "0" in cmd
    assert "--elo" in cmd and "1800" in cmd
    assert "--device" in cmd and "cuda" in cmd
    assert "--multipv" in cmd and "5" in cmd
    assert "--no-use-amp" not in cmd


def test_format_maia3_wdl():
    move = EngineMove(uci="e2e4", san="e4", wdl_permille=(450, 300, 250))
    assert format_maia3_wdl(move) == "W45% D30% L25%"
    assert format_maia3_wdl(EngineMove(uci="e2e4", san="e4")) is None


def test_maia3_installed():
    # Should not raise; True or False depending on environment.
    assert isinstance(is_maia3_installed(), bool)
