"""Humanization settings loaded from config/default.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.app.config import load_config
from src.engines.human_move_selector import SelectionConfig
from src.engines.position_profiler import OpeningConfig, TimingConfig


@dataclass
class HumanizationConfig:
    enabled: bool = True
    target_elo: int = 2000
    opening: OpeningConfig = None  # type: ignore[assignment]
    timing: TimingConfig = None  # type: ignore[assignment]
    selection: SelectionConfig = None  # type: ignore[assignment]
    use_book: bool = True
    book_path: str = "data/openings/performance.bin"
    book_temperature: float = 0.3

    def __post_init__(self) -> None:
        if self.opening is None:
            self.opening = OpeningConfig()
        if self.timing is None:
            self.timing = TimingConfig()
        if self.selection is None:
            self.selection = SelectionConfig()


def humanization_from_config(config: dict[str, Any] | None = None) -> HumanizationConfig:
    cfg = config or load_config()
    raw = cfg.get("humanization", {}) or {}
    opening_raw = raw.get("opening", {}) or {}
    timing_raw = raw.get("timing", {}) or {}
    selection_raw = raw.get("selection", {}) or {}

    return HumanizationConfig(
        enabled=bool(raw.get("enabled", True)),
        target_elo=int(raw.get("target_elo", 2000)),
        opening=OpeningConfig(
            max_fullmove=int(opening_raw.get("max_fullmove", 12)),
            movetime_ms=int(opening_raw.get("movetime_ms", 120)),
            min_pieces=int(opening_raw.get("min_pieces", 24)),
        ),
        timing=TimingConfig(
            probe_movetime_ms=int(timing_raw.get("probe_movetime_ms", 200)),
            normal_movetime_ms=int(timing_raw.get("normal_movetime_ms", 3500)),
            critical_movetime_ms=int(timing_raw.get("critical_movetime_ms", 9500)),
            critical_cp_gap=int(timing_raw.get("critical_cp_gap", 25)),
        ),
        selection=SelectionConfig(
            multipv=int(selection_raw.get("multipv", 5)),
            practical_move_chance=float(selection_raw.get("practical_move_chance", 0.15)),
            practical_cp_slack_1900=int(selection_raw.get("practical_cp_slack_1900", 45)),
            practical_cp_slack_2100=int(selection_raw.get("practical_cp_slack_2100", 25)),
            inaccuracy_chance_floor=float(selection_raw.get("inaccuracy_chance_floor", 0.35)),
            inaccuracy_chance_ceiling=float(selection_raw.get("inaccuracy_chance_ceiling", 0.06)),
            blunder_chance_floor=float(selection_raw.get("blunder_chance_floor", 0.08)),
            blunder_cp_min=int(selection_raw.get("blunder_cp_min", 120)),
        ),
        use_book=bool(opening_raw.get("use_book", True)),
        book_path=str(opening_raw.get("book_path", "data/openings/performance.bin")),
        book_temperature=float(opening_raw.get("book_temperature", 0.3)),
    )


def resolve_book_path(book_path: str, project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    path = Path(book_path)
    if path.is_absolute():
        return path
    return root / path
