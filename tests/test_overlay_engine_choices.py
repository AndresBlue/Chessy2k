"""Tests for overlay engine selection (Stockfish / Reckless / Maia-3)."""

import tempfile
from pathlib import Path

from src.app.overlay_theme import (
    MAIA3_MODEL_5M,
    MAIA3_MODEL_23M,
    MAIA3_MODEL_79M,
    MAIA3_MODEL_LABELS,
    OVERLAY_ENGINE_LABELS,
    OVERLAY_ENGINE_MAIA3,
    OVERLAY_ENGINE_RECKLESS,
    OVERLAY_ENGINE_STOCKFISH,
    load_maia3_model,
    load_overlay_engine,
    normalize_maia3_model,
    save_maia3_model,
    save_overlay_engine,
)


def test_overlay_engine_labels():
    assert OVERLAY_ENGINE_LABELS[OVERLAY_ENGINE_STOCKFISH] == "Stockfish"
    assert OVERLAY_ENGINE_LABELS[OVERLAY_ENGINE_RECKLESS] == "Reckless"
    assert OVERLAY_ENGINE_LABELS[OVERLAY_ENGINE_MAIA3] == "Maia-3"


def test_save_load_maia3_engine():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_overlay_engine(root, OVERLAY_ENGINE_MAIA3)
        assert load_overlay_engine(root) == OVERLAY_ENGINE_MAIA3


def test_save_load_reckless_engine():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_overlay_engine(root, OVERLAY_ENGINE_RECKLESS)
        assert load_overlay_engine(root) == OVERLAY_ENGINE_RECKLESS


def test_unknown_engine_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_overlay_engine(root, "cheesy2k")
        assert load_overlay_engine(root) == OVERLAY_ENGINE_STOCKFISH


def test_normalize_maia3_model():
    assert normalize_maia3_model("5m") == MAIA3_MODEL_5M
    assert normalize_maia3_model("79M") == MAIA3_MODEL_79M
    assert normalize_maia3_model("maia3_23m") == MAIA3_MODEL_23M
    assert normalize_maia3_model("nope") == MAIA3_MODEL_23M


def test_save_load_maia3_model():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_maia3_model(root, "79m")
        assert load_maia3_model(root) == MAIA3_MODEL_79M
        assert "79M" in MAIA3_MODEL_LABELS[MAIA3_MODEL_79M]
