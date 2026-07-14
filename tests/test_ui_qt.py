"""Tests for the PySide6/Qt UI layer (pure helpers + offscreen smoke)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Force a headless Qt backend before importing any Qt module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.app.ui_qt import image_utils, theme  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# -- pure helpers --------------------------------------------------------
def test_clamp01():
    assert image_utils.clamp01(-1.0) == 0.0
    assert image_utils.clamp01(2.0) == 1.0
    assert image_utils.clamp01(0.3) == 0.3


def test_think_fraction():
    assert image_utils.think_fraction(5.0, 10.0) == 0.5
    assert image_utils.think_fraction(0.0, 10.0) == 0.0
    assert image_utils.think_fraction(20.0, 10.0) == 1.0
    assert image_utils.think_fraction(5.0, 0.0) == 0.0


def test_eval_to_fraction_balanced_and_signs():
    assert image_utils.eval_to_fraction("+0.00") == pytest.approx(0.5, abs=1e-6)
    assert image_utils.eval_to_fraction("") == 0.5
    assert image_utils.eval_to_fraction("garbage") == 0.5
    assert image_utils.eval_to_fraction("+5.00") > 0.8
    assert image_utils.eval_to_fraction("-5.00") < 0.2


def test_eval_to_fraction_mate():
    assert image_utils.eval_to_fraction("#+3") == pytest.approx(0.99)
    assert image_utils.eval_to_fraction("#-2") == pytest.approx(0.01)


def test_bgr_to_qimage_dimensions(qapp):
    img = (np.random.rand(40, 60, 3) * 255).astype("uint8")
    qimg = image_utils.bgr_to_qimage(img)
    assert qimg.width() == 60
    assert qimg.height() == 40
    assert not qimg.isNull()


def test_bgr_to_qimage_empty_is_null(qapp):
    qimg = image_utils.bgr_to_qimage(np.zeros((0, 0, 3), dtype="uint8"))
    assert qimg.isNull()


# -- theme ---------------------------------------------------------------
def test_build_qss_contains_palette_accent(qapp):
    fonts = theme.load_fonts(PROJECT_ROOT)
    palette = theme.THEMES["dark"]
    qss = theme.build_qss(palette, fonts)
    assert palette.accent in qss
    assert fonts.base in qss
    assert len(qss) > 1000


def test_load_fonts_returns_family(qapp):
    fonts = theme.load_fonts(PROJECT_ROOT)
    assert isinstance(fonts.base, str) and fonts.base
    assert isinstance(fonts.mono, str) and fonts.mono


def test_both_themes_have_distinct_backgrounds():
    assert theme.THEMES["dark"].bg != theme.THEMES["light"].bg
    assert theme.THEMES["dark"].accent  # non-empty


# -- offscreen window smoke ---------------------------------------------
def test_app_window_smoke(qapp):
    fake_paths = {"root": PROJECT_ROOT, "stockfish": None, "classifier": None}
    with patch("src.app.ui_qt.app_window.resolve_runtime_paths", return_value=fake_paths):
        from src.app.ui_qt.app_window import AppWindow

        window = AppWindow()
        try:
            window.show()
            for _ in range(10):
                qapp.processEvents()

            # Theme toggling must not raise and must flip the palette.
            initial = window._theme_name
            window._toggle_theme()
            qapp.processEvents()
            assert window._theme_name != initial

            # Feed a synthetic successful result through the render path.
            from src.app.fast_analyzer import AnalysisOutput

            out = AnalysisOutput(
                fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                best_move_san="e4",
                best_move_uci="e2e4",
                evaluation="+0.30",
                vision_ms=10,
                engine_ms=200,
                total_ms=230,
                planned_ms=4000,
                think_note="Jugada natural",
                humanized=True,
                phase="opening",
                criticality="low",
                target_elo=1800,
                inference_mode="full",
            )
            window._analysis_id = 7
            window._last_capture = (np.random.rand(60, 60, 3) * 255).astype("uint8")
            window._show_result(out, 7)
            for _ in range(5):
                qapp.processEvents()

            assert "e2e4" in window.move_value.text()
            assert window.think_value.text() == "4s"
            assert window.eval_value.text() == "+0.30"
            assert window.preview.is_thinking
        finally:
            window._closing = True
            window.close()
            qapp.processEvents()


def test_app_window_error_result_resets_eval(qapp):
    fake_paths = {"root": PROJECT_ROOT, "stockfish": None, "classifier": None}
    with patch("src.app.ui_qt.app_window.resolve_runtime_paths", return_value=fake_paths):
        from src.app.ui_qt.app_window import AppWindow
        from src.app.fast_analyzer import AnalysisOutput

        window = AppWindow()
        try:
            window._analysis_id = 3
            err = AnalysisOutput(
                fen="",
                best_move_san="",
                best_move_uci="",
                evaluation="",
                vision_ms=5,
                engine_ms=0,
                total_ms=12,
                error="Vision poco confiable",
                tracker_status="vision_error",
            )
            window._show_result(err, 3)
            qapp.processEvents()
            assert "Error" in window.status_label.text()
            assert window.move_value.text() == "\u2014"
            assert not window.preview.is_thinking
        finally:
            window._closing = True
            window.close()
            qapp.processEvents()
