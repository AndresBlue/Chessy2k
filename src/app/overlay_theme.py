"""Overlay settings persistence and shared theme constants.

Theme painting for the legacy Tk UI lived here historically; the Qt client uses
`src.app.ui_qt.theme` for stylesheets. This module still owns engine choice
constants and `data/overlay_settings.json` read/write helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

from src.app.elo_power import ELO_MAX_HUMAN, ELO_MIN, MAX_EFFORT_ELO, is_max_effort

@dataclass(frozen=True)
class ThemePalette:
    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_text: str
    success: str
    danger: str
    preview_bg: str
    input_bg: str


THEMES: dict[str, ThemePalette] = {
    "dark": ThemePalette(
        name="dark",
        bg="#0f1117",
        surface="#181c27",
        surface_alt="#222836",
        border="#2f3648",
        text="#eef0f6",
        text_muted="#9aa3b8",
        accent="#5b7cfa",
        accent_hover="#4a6ae8",
        accent_text="#ffffff",
        success="#3dd68c",
        danger="#f07178",
        preview_bg="#0a0c10",
        input_bg="#1a1f2e",
    ),
    "light": ThemePalette(
        name="light",
        bg="#eef1f8",
        surface="#ffffff",
        surface_alt="#f6f8fc",
        border="#d5dbe8",
        text="#151821",
        text_muted="#5f6778",
        accent="#3b5bdb",
        accent_hover="#2f4ec4",
        accent_text="#ffffff",
        success="#1f9d63",
        danger="#d64545",
        preview_bg="#e3e8f2",
        input_bg="#ffffff",
    ),
}

SETTINGS_FILE = "data/overlay_settings.json"

OVERLAY_ENGINE_STOCKFISH = "stockfish"
OVERLAY_ENGINE_RECKLESS = "reckless"
OVERLAY_ENGINE_MAIA3 = "maia3"
OVERLAY_ENGINE_CHOICES = (
    OVERLAY_ENGINE_STOCKFISH,
    OVERLAY_ENGINE_RECKLESS,
    OVERLAY_ENGINE_MAIA3,
)

OVERLAY_ENGINE_LABELS: dict[str, str] = {
    OVERLAY_ENGINE_STOCKFISH: "Stockfish",
    OVERLAY_ENGINE_RECKLESS: "Reckless",
    OVERLAY_ENGINE_MAIA3: "Maia-3",
}

MAIA3_MODEL_5M = "maia3-5m"
MAIA3_MODEL_23M = "maia3-23m"
MAIA3_MODEL_79M = "maia3-79m"
MAIA3_MODEL_CHOICES = (MAIA3_MODEL_5M, MAIA3_MODEL_23M, MAIA3_MODEL_79M)
MAIA3_MODEL_LABELS: dict[str, str] = {
    MAIA3_MODEL_5M: "5M (fast)",
    MAIA3_MODEL_23M: "23M (balanced)",
    MAIA3_MODEL_79M: "79M (max accuracy)",
}


def normalize_maia3_model(model: str | None, default: str = MAIA3_MODEL_23M) -> str:
    raw = str(model or default).strip().lower().replace("_", "-")
    aliases = {
        "5m": MAIA3_MODEL_5M,
        "maia3-5m": MAIA3_MODEL_5M,
        "23m": MAIA3_MODEL_23M,
        "maia3-23m": MAIA3_MODEL_23M,
        "79m": MAIA3_MODEL_79M,
        "maia3-79m": MAIA3_MODEL_79M,
    }
    return aliases.get(raw, default if default in MAIA3_MODEL_CHOICES else MAIA3_MODEL_23M)


def _load_settings(root: Path) -> dict:
    path = root / SETTINGS_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_settings(root: Path, data: dict) -> None:
    path = root / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_theme_preference(root: Path) -> str:
    theme = _load_settings(root).get("theme", "dark")
    return theme if theme in THEMES else "dark"


def save_theme_preference(root: Path, theme: str) -> None:
    data = _load_settings(root)
    data["theme"] = theme
    _save_settings(root, data)


def load_show_predictions(root: Path) -> bool:
    return bool(_load_settings(root).get("show_predictions", True))


def save_show_predictions(root: Path, show: bool) -> None:
    data = _load_settings(root)
    data["show_predictions"] = show
    _save_settings(root, data)


def load_human_mode(root: Path) -> bool:
    return bool(_load_settings(root).get("human_mode", True))


def save_human_mode(root: Path, enabled: bool) -> None:
    data = _load_settings(root)
    data["human_mode"] = enabled
    _save_settings(root, data)


def load_target_elo(root: Path) -> int:
    raw = _load_settings(root).get("target_elo", 2000)
    try:
        elo = int(raw)
    except (TypeError, ValueError):
        return 2000
    if is_max_effort(elo):
        return MAX_EFFORT_ELO
    return max(ELO_MIN, min(ELO_MAX_HUMAN, elo))


def save_target_elo(root: Path, elo: int) -> None:
    data = _load_settings(root)
    if is_max_effort(elo):
        data["target_elo"] = MAX_EFFORT_ELO
    else:
        data["target_elo"] = max(ELO_MIN, min(ELO_MAX_HUMAN, elo))
    _save_settings(root, data)


def load_overlay_engine(root: Path, default: str = OVERLAY_ENGINE_STOCKFISH) -> str:
    mode = _load_settings(root).get("overlay_engine", default)
    return mode if mode in OVERLAY_ENGINE_CHOICES else default


def save_overlay_engine(root: Path, mode: str) -> None:
    data = _load_settings(root)
    data["overlay_engine"] = (
        mode if mode in OVERLAY_ENGINE_CHOICES else OVERLAY_ENGINE_STOCKFISH
    )
    _save_settings(root, data)


def load_maia3_model(root: Path, default: str = MAIA3_MODEL_23M) -> str:
    saved = _load_settings(root).get("maia3_model")
    if saved:
        return normalize_maia3_model(str(saved), default=default)
    return normalize_maia3_model(default)


def save_maia3_model(root: Path, model: str) -> None:
    data = _load_settings(root)
    data["maia3_model"] = normalize_maia3_model(model)
    _save_settings(root, data)


def load_window_geometry(root: Path) -> str | None:
    geo = _load_settings(root).get("window_geometry")
    return str(geo) if geo else None


def save_window_geometry(root: Path, geometry: str) -> None:
    data = _load_settings(root)
    data["window_geometry"] = geometry
    _save_settings(root, data)


def apply_theme(style: ttk.Style, palette: ThemePalette) -> None:
    style.theme_use("clam")

    style.configure(".", background=palette.bg, foreground=palette.text, font=("Segoe UI", 11))
    style.configure("TFrame", background=palette.bg)
    style.configure("Card.TFrame", background=palette.surface)
    style.configure("CardInner.TFrame", background=palette.surface_alt)
    style.configure("ThinkBadge.TFrame", background=palette.surface_alt, relief="flat")

    style.configure(
        "TLabelframe",
        background=palette.surface,
        foreground=palette.text_muted,
        bordercolor=palette.border,
        relief="flat",
    )
    style.configure("TLabelframe.Label", background=palette.surface, foreground=palette.text_muted)

    style.configure("TLabel", background=palette.bg, foreground=palette.text)
    style.configure("Muted.TLabel", background=palette.bg, foreground=palette.text_muted)
    style.configure("Card.TLabel", background=palette.surface, foreground=palette.text, font=("Segoe UI", 11))
    style.configure("CardMuted.TLabel", background=palette.surface, foreground=palette.text_muted, font=("Segoe UI Semibold", 10))
    style.configure("Title.TLabel", background=palette.bg, foreground=palette.text, font=("Segoe UI Semibold", 18))
    style.configure("Subtitle.TLabel", background=palette.bg, foreground=palette.text_muted, font=("Segoe UI", 11))
    style.configure("Move.TLabel", background=palette.surface, foreground=palette.text, font=("Segoe UI Semibold", 26))
    style.configure("Eval.TLabel", background=palette.surface, foreground=palette.success, font=("Segoe UI Semibold", 16))
    style.configure("ThinkTitle.TLabel", background=palette.surface_alt, foreground=palette.text_muted, font=("Segoe UI Semibold", 10))
    style.configure("ThinkValue.TLabel", background=palette.surface_alt, foreground=palette.accent, font=("Segoe UI Semibold", 30))
    style.configure("ThinkNote.TLabel", background=palette.surface_alt, foreground=palette.text, font=("Segoe UI", 10))
    style.configure("ThinkPanel.TLabel", background=palette.surface, foreground=palette.accent, font=("Segoe UI Semibold", 20))
    style.configure("TimeList.TLabel", background=palette.surface, foreground=palette.text, font=("Consolas", 11))

    style.configure(
        "TButton",
        background=palette.surface_alt,
        foreground=palette.text,
        bordercolor=palette.border,
        focusthickness=0,
        padding=(18, 11),
        font=("Segoe UI Semibold", 11),
    )
    style.map(
        "TButton",
        background=[("active", palette.border), ("pressed", palette.border)],
        foreground=[("disabled", palette.text_muted)],
    )

    style.configure(
        "Primary.TButton",
        background=palette.accent,
        foreground=palette.accent_text,
        bordercolor=palette.accent,
        padding=(20, 13),
        font=("Segoe UI Semibold", 11),
    )
    style.map(
        "Primary.TButton",
        background=[("active", palette.accent_hover), ("pressed", palette.accent_hover)],
        foreground=[("disabled", palette.text_muted)],
    )

    style.configure(
        "Ghost.TButton",
        background=palette.surface,
        foreground=palette.text_muted,
        bordercolor=palette.border,
        padding=(10, 6),
    )
    style.map("Ghost.TButton", background=[("active", palette.surface_alt)])

    style.configure(
        "TCheckbutton",
        background=palette.bg,
        foreground=palette.text,
        focuscolor=palette.bg,
        font=("Segoe UI", 11),
        padding=(4, 6),
    )
    style.map("TCheckbutton", background=[("active", palette.bg)])

    style.configure(
        "TRadiobutton",
        background=palette.surface_alt,
        foreground=palette.text,
        focuscolor=palette.surface_alt,
        padding=(10, 7),
        font=("Segoe UI", 11),
    )
    style.map("TRadiobutton", background=[("active", palette.surface_alt)])

    style.configure(
        "TCombobox",
        fieldbackground=palette.input_bg,
        background=palette.surface_alt,
        foreground=palette.text,
        arrowcolor=palette.text_muted,
        bordercolor=palette.border,
    )
