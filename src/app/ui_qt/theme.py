"""Chess.com-style theme: palette, QSS, fonts, shadows and settings IO.

The settings helpers read/write the same ``data/overlay_settings.json`` used by
the legacy tkinter client so user preferences carry over, but this module has no
tkinter dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget

from src.app.elo_power import ELO_MAX_HUMAN, ELO_MIN, MAX_EFFORT_ELO, is_max_effort

SETTINGS_FILE = "data/overlay_settings.json"


@dataclass(frozen=True)
class ChessPalette:
    name: str
    bg: str
    bg_elevated: str
    surface: str
    surface_alt: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str
    success: str
    danger: str
    warning: str
    preview_bg: str
    input_bg: str
    shadow_rgba: tuple[int, int, int, int]
    eval_track: str


THEMES: dict[str, ChessPalette] = {
    # Sober Chess.com dark board theme: charcoal surfaces + signature green.
    "dark": ChessPalette(
        name="dark",
        bg="#262421",
        bg_elevated="#2d2a27",
        surface="#312e2b",
        surface_alt="#3a3633",
        border="#403b36",
        border_strong="#4d473f",
        text="#ededec",
        text_muted="#9d9893",
        accent="#81b64c",
        accent_hover="#95c95d",
        accent_pressed="#6fa23f",
        accent_text="#ffffff",
        success="#81b64c",
        danger="#e0524b",
        warning="#e6912c",
        preview_bg="#1c1a18",
        input_bg="#2a2724",
        shadow_rgba=(0, 0, 0, 150),
        eval_track="#1f1d1b",
    ),
    "light": ChessPalette(
        name="light",
        bg="#eceae6",
        bg_elevated="#f3f1ed",
        surface="#ffffff",
        surface_alt="#f4f3ee",
        border="#dcdad3",
        border_strong="#c8c5bc",
        text="#2b2926",
        text_muted="#6f6b64",
        accent="#5b8a3c",
        accent_hover="#69a046",
        accent_pressed="#4c7531",
        accent_text="#ffffff",
        success="#5b8a3c",
        danger="#c23b34",
        warning="#c9781f",
        preview_bg="#dedcd5",
        input_bg="#ffffff",
        shadow_rgba=(40, 38, 34, 60),
        eval_track="#cdcabf",
    ),
}


@dataclass(frozen=True)
class FontConfig:
    base: str
    mono: str


def _settings_path(root: Path) -> Path:
    return root / SETTINGS_FILE


def load_settings(root: Path) -> dict:
    path = _settings_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(root: Path, data: dict) -> None:
    path = _settings_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _update_setting(root: Path, key: str, value) -> None:
    data = load_settings(root)
    data[key] = value
    save_settings(root, data)


def load_theme_preference(root: Path) -> str:
    theme = load_settings(root).get("theme", "dark")
    return theme if theme in THEMES else "dark"


def save_theme_preference(root: Path, theme: str) -> None:
    _update_setting(root, "theme", theme)


def load_show_predictions(root: Path) -> bool:
    return bool(load_settings(root).get("show_predictions", True))


def save_show_predictions(root: Path, show: bool) -> None:
    _update_setting(root, "show_predictions", bool(show))


def load_target_elo(root: Path) -> int:
    raw = load_settings(root).get("target_elo", 2000)
    try:
        elo = int(raw)
    except (TypeError, ValueError):
        return 2000
    if is_max_effort(elo):
        return MAX_EFFORT_ELO
    return max(ELO_MIN, min(ELO_MAX_HUMAN, elo))


def save_target_elo(root: Path, elo: int) -> None:
    if is_max_effort(elo):
        _update_setting(root, "target_elo", MAX_EFFORT_ELO)
    else:
        _update_setting(root, "target_elo", max(ELO_MIN, min(ELO_MAX_HUMAN, elo)))


def load_window_geometry(root: Path) -> str | None:
    geo = load_settings(root).get("qt_window_geometry")
    return str(geo) if geo else None


def save_window_geometry(root: Path, geometry_hex: str) -> None:
    _update_setting(root, "qt_window_geometry", geometry_hex)


def load_fonts(root: Path) -> FontConfig:
    """Register any bundled fonts and pick the best sober family available."""
    fonts_dir = root / "assets" / "fonts"
    if fonts_dir.exists():
        for path in sorted(fonts_dir.glob("*")):
            if path.suffix.lower() in (".ttf", ".otf"):
                QFontDatabase.addApplicationFont(str(path))

    available = set(QFontDatabase.families())

    def _first(candidates: list[str], default: str) -> str:
        for name in candidates:
            if name in available:
                return name
        return default

    base = _first(
        ["Inter", "Segoe UI Variable Display", "Segoe UI", "Helvetica Neue", "Arial"],
        "Segoe UI",
    )
    mono = _first(
        ["Cascadia Code", "JetBrains Mono", "Consolas", "Courier New"],
        "monospace",
    )
    return FontConfig(base=base, mono=mono)


def make_app_font(fonts: FontConfig, point_size: int = 10) -> QFont:
    font = QFont(fonts.base, point_size)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def card_shadow(palette: ChessPalette, *, blur: int = 28, dy: int = 6) -> QGraphicsDropShadowEffect:
    """Build a soft elevation shadow effect for a card-like widget."""
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(blur)
    effect.setColor(QColor(*palette.shadow_rgba))
    effect.setOffset(0, dy)
    return effect


def apply_shadow(widget: QWidget, palette: ChessPalette, *, blur: int = 28, dy: int = 6) -> None:
    widget.setGraphicsEffect(card_shadow(palette, blur=blur, dy=dy))


def build_qss(palette: ChessPalette, fonts: FontConfig) -> str:
    """Generate the application stylesheet for the given palette."""
    p = palette
    return f"""
* {{
    font-family: "{fonts.base}";
    color: {p.text};
    outline: 0;
}}

QWidget#Root {{
    background: {p.bg};
}}

QToolTip {{
    background: {p.surface_alt};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: 6px;
    padding: 6px 8px;
}}

QLabel {{
    background: transparent;
    color: {p.text};
}}

QLabel#AppTitle {{
    font-size: 22px;
    font-weight: 700;
    color: {p.text};
}}

QLabel#AppSubtitle {{
    font-size: 12px;
    color: {p.text_muted};
}}

QLabel#RegionChip {{
    color: {p.text_muted};
    font-size: 12px;
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 999px;
    padding: 5px 12px;
}}

QFrame#Card {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 16px;
}}

QFrame#CardInner {{
    background: {p.surface_alt};
    border: 1px solid {p.border};
    border-radius: 12px;
}}

QLabel.cardTitle, QLabel#CardTitle {{
    color: {p.text_muted};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}

QLabel#MoveValue {{
    font-size: 30px;
    font-weight: 700;
    color: {p.text};
}}

QLabel#ThinkValue {{
    font-size: 30px;
    font-weight: 700;
    color: {p.accent};
}}

QLabel#ThinkNote {{
    color: {p.text};
    font-size: 13px;
}}

QLabel#EvalValue {{
    font-size: 18px;
    font-weight: 700;
    color: {p.success};
}}

QLabel#MetaValue {{
    color: {p.text};
    font-size: 13px;
}}

QLabel#FenValue {{
    color: {p.text_muted};
    font-family: "{fonts.mono}";
    font-size: 12px;
}}

QLabel#TimesValue {{
    color: {p.text};
    font-family: "{fonts.mono}";
    font-size: 12px;
}}

QLabel#StatusLabel {{
    color: {p.text_muted};
    font-size: 12px;
}}

QPushButton {{
    background: {p.surface_alt};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: 12px;
    padding: 11px 18px;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton:hover {{
    background: {p.border};
    border-color: {p.border_strong};
}}

QPushButton:pressed {{
    background: {p.border_strong};
}}

QPushButton:disabled {{
    color: {p.text_muted};
    background: {p.surface};
    border-color: {p.border};
}}

QPushButton#PrimaryButton {{
    background: {p.accent};
    color: {p.accent_text};
    border: 1px solid {p.accent};
    border-radius: 12px;
    padding: 13px 18px;
    font-size: 14px;
    font-weight: 700;
}}

QPushButton#PrimaryButton:hover {{
    background: {p.accent_hover};
    border-color: {p.accent_hover};
}}

QPushButton#PrimaryButton:pressed {{
    background: {p.accent_pressed};
    border-color: {p.accent_pressed};
}}

QPushButton#PrimaryButton:disabled {{
    background: {p.surface_alt};
    color: {p.text_muted};
    border-color: {p.border};
}}

QPushButton#GhostButton {{
    background: transparent;
    color: {p.text_muted};
    border: 1px solid {p.border};
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
}}

QPushButton#GhostButton:hover {{
    background: {p.surface_alt};
    color: {p.text};
}}

QCheckBox, QRadioButton {{
    color: {p.text};
    font-size: 13px;
    spacing: 9px;
    background: transparent;
    padding: 3px 0;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {p.border_strong};
    background: {p.input_bg};
}}

QCheckBox::indicator {{
    border-radius: 5px;
}}

QRadioButton::indicator {{
    border-radius: 9px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {p.accent};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {p.accent};
    border-color: {p.accent};
}}

QSlider::groove:horizontal {{
    height: 6px;
    border-radius: 3px;
    background: {p.surface_alt};
}}

QSlider::sub-page:horizontal {{
    height: 6px;
    border-radius: 3px;
    background: {p.accent};
}}

QSlider::handle:horizontal {{
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
    background: {p.text};
    border: 2px solid {p.accent};
}}

QSlider::handle:horizontal:hover {{
    background: {p.accent_hover};
}}

QComboBox {{
    background: {p.input_bg};
    color: {p.text};
    border: 1px solid {p.border_strong};
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 13px;
}}

QComboBox:hover {{
    border-color: {p.accent};
}}

QComboBox QAbstractItemView {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.border_strong};
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
    outline: 0;
}}

QScrollArea {{
    background: transparent;
    border: 0;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: {p.border_strong};
    border-radius: 5px;
    min-height: 28px;
}}

QScrollBar::handle:vertical:hover {{
    background: {p.text_muted};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""
