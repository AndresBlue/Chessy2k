"""Tests for overlay Elo persistence."""

import tempfile
from pathlib import Path

from src.app.elo_power import MAX_EFFORT_ELO
from src.app.overlay_theme import load_target_elo, save_target_elo


def test_save_load_arbitrary_elo():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_target_elo(root, 1750)
        assert load_target_elo(root) == 1750


def test_save_load_max_effort():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_target_elo(root, MAX_EFFORT_ELO)
        assert load_target_elo(root) == MAX_EFFORT_ELO
