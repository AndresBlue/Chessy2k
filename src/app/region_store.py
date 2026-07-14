"""Persist the captured board region to disk (shared by UI clients)."""

from __future__ import annotations

import json
from pathlib import Path

from src.app.screen_capture import ScreenRegion

REGION_CONFIG = "data/overlay_region.json"


def load_region(root: Path) -> ScreenRegion | None:
    path = root / REGION_CONFIG
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        region = ScreenRegion.from_dict(data)
        return region if region.is_valid() else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None


def save_region(root: Path, region: ScreenRegion) -> None:
    path = root / REGION_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(region.as_dict(), indent=2), encoding="utf-8")
