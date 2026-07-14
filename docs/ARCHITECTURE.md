# Architecture

Chessy is a local pipeline from screen pixels to a UCI move suggestion.

```
┌───────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌───────────────────────┐
│ Screen capture│ ──▶│   VisionPipeline  │──▶ │ FEN + checks │ ──▶│     Engine backend    │
│ (mss / region)│     │  ResNet squares  │     │ tracker/legal│     │ SF / Reckless / Maia  │
└───────────────┘     └──────────────────┘     └──────────────┘     └───────────────────────┘
                                                                             │
                                                                             ▼
                                                                      Qt overlay /
                                                                       CLI / API
```

## Overlay path

1. User selects a screen region covering the digital board.
2. `CaptureService` grabs frames; `BoardStableDetector` waits for a settled board.
3. `FastAnalyzer` runs vision inference (full or incremental squares).
4. Placement is converted to a FEN with the user-selected side to move.
5. Quality gates reject low-confidence or illegal placements.
6. The selected engine produces a move, evaluation, and optional think-time hint.
7. The UI draws an arrow overlay and shows timing / meta text.

Entrypoint: `python -m src.app.ui_qt` → `src.app.ui_qt.app_window.AppWindow`.

## Vision

- Input: BGR screenshot of the board region.
- Board crop + 8×8 segmentation (`src/vision/`).
- Per-square ResNet18 classifier (`data/checkpoints/vision/best.pt`).
- Incremental mode reuses cached square predictions when the board is mostly unchanged.

This repository ships **inference only**. Training datasets and trainers were removed; retrain offline if you need a new checkpoint, then replace `best.pt`.

## Engines

| Mode | Process | Humanization |
|------|---------|--------------|
| Stockfish | `engines/stockfish/*.exe` via UCI | UCI_Elo + MultiPV Python selection + book + timing |
| Reckless | `engines/reckless/*.exe` via UCI | MultiPV Python selection + book + timing (no UCI Elo) |
| Maia-3 | `python -m maia3.uci` | Native Elo options + Maia timing advisor |

Shared orchestration lives in `src/app/fast_analyzer.py`. UCI I/O is in `src/engines/stockfish_client.py` (subclassed by Reckless and reused by Maia-3).

## Path resolution

All runtime assets resolve relative to the repository root (`src/app/runtime_paths.py`):

- Stockfish / Reckless binaries under `engines/`
- Vision checkpoint under `data/checkpoints/vision/`
- Maia-3 cache under `data/maia3/cache/`
- Opening book under `data/openings/`

Environment overrides: `CHESSY_STOCKFISH_PATH`, `CHESSY_RECKLESS_PATH`, `CHESSY_CLASSIFIER_CHECKPOINT`.
