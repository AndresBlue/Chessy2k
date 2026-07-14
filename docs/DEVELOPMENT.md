# Development

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts/setup_maia3.py --model maia3-23m   # optional
```

Confirm engines resolve:

```bash
python -c "from src.app.runtime_paths import resolve_runtime_paths; print(resolve_runtime_paths())"
```

## Run

```bash
python -m src.app.ui_qt
python app.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

## Tests

```bash
pytest tests/ -v
pytest tests/test_reckless_client.py -v -m integration
```

Integration tests require the binaries under `engines/`. Unit tests should not need GPU.

## Layout conventions

- Application code lives under `src/` only.
- Never hardcode absolute machine paths; use `project_root()` / config-relative paths.
- Engine binaries live under `engines/<name>/`, not scattered at the repo root.
- Large datasets and HF caches are gitignored; do not commit them.
- Prefer short module docstrings that say *why*; avoid tutorial comments and emoji.

## Code style

- Python 3.10+ type hints.
- `pathlib.Path` for filesystem work.
- UCI engines are subprocesses — do not vendor engine source trees unless required for licensing redistrib of sources separately.

## Packaging notes

- Vision checkpoint is force-included via `.gitignore` exception: `!data/checkpoints/vision/best.pt`.
- Stockfish (GPL) and Reckless (AGPL) binaries remain separate processes; document licenses when redistributing.
