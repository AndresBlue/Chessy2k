[English](/README.md) / [Español](/docs/i18n/README_es.md)

# Chessy

Local desktop chess assistant: capture a digital board screenshot, recover FEN with a vision model, and get a move suggestion from **Stockfish**, **Reckless**, or **Maia-3**.

```
Screenshot → board crop → 64-square classifier → FEN → UCI engine → suggested move + think time
```

No cloud dependency for the core loop. Engines and the vision checkpoint ship inside the repository (Maia-3 weights are cached locally on first setup).

## Features

- Qt desktop overlay (`PySide6`) with board region capture and auto-analyze
- Vision inference from a shipped ResNet square classifier
- Three engines selectable in the UI:
  - **Stockfish** — full strength or humanized (`UCI_Elo` + MultiPV selection)
  - **Reckless** — competitive UCI engine with the same Python humanization layer
  - **Maia-3** — human-like neural policies at configurable Elo
- Adaptive think-time hints and optional Polyglot opening book
- CLI and optional FastAPI surface for FEN / screenshot workflows

## Requirements

- Python 3.10+ (3.10 recommended for Torch CUDA builds)
- Windows x86-64 with AVX2 for the bundled engine binaries (Linux/macOS: place matching binaries under `engines/`)
- Optional: NVIDIA GPU for faster vision / Maia-3 inference

## Quick start

```bash
# From the repository root
python -m pip install -e ".[dev]"

# Optional: install Maia-3 and cache weights into data/maia3/cache
python scripts/setup_maia3.py --model maia3-23m

# Desktop overlay
python -m src.app.ui_qt
# or on Windows: start_chessy_overlay.bat
```

### CLI

```bash
# FEN → Stockfish
python app.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# FEN → Reckless
python app.py --fen "..." --engine reckless

# Screenshot → move (uses data/checkpoints/vision/best.pt by default)
python app.py --image board.png --side white
```

Binary paths resolve automatically from `engines/stockfish/` and `engines/reckless/`. Override with `--stockfish-path`, `--reckless-path`, or env vars `CHESSY_STOCKFISH_PATH` / `CHESSY_RECKLESS_PATH` / `CHESSY_CLASSIFIER_CHECKPOINT`.

## Project layout

| Path | Role |
|------|------|
| `src/app/` | Overlay UI, CLI, analyzer, path resolution |
| `src/vision/` | Screenshot → FEN inference |
| `src/chess_core/` | FEN helpers, legality, game tracker |
| `src/engines/` | Stockfish, Reckless, Maia-3, humanization |
| `src/search/` | Polyglot opening book |
| `engines/` | Vendored UCI binaries |
| `third_party/maia3/` | Vendored Maia-3 package (or cloned by setup script) |
| `config/default.yaml` | Default configuration (repo-relative paths) |
| `data/checkpoints/vision/best.pt` | Shipped vision weights |
| `docs/` | Architecture, engines, configuration, development |

## Configuration

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md). Engine-specific notes and humanization behavior: [docs/ENGINES.md](docs/ENGINES.md).

## Tests

```bash
pytest tests/ -v
# Engine binary smoke tests
pytest tests/ -v -m integration
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Engines](docs/ENGINES.md)
- [Configuration](docs/CONFIGURATION.md)
- [Development](docs/DEVELOPMENT.md)
- [Contributing](CONTRIBUTING.md)

## License

Application code is MIT — see [LICENSE](LICENSE).

Bundled and third-party engines retain their own licenses:

| Component | License | Notes |
|-----------|---------|-------|
| Stockfish | GPL-3.0 | Binary under `engines/stockfish/` |
| Reckless | AGPL-3.0 | Binary under `engines/reckless/` |
| Maia-3 | Upstream (CSSLab) | See `third_party/maia3/` |
| Vision checkpoint | Project (MIT distribution) | `data/checkpoints/vision/best.pt` |

Distributing modified versions of AGPL/GPL binaries may impose source obligations for those components. Chessy invokes them as separate UCI processes and does not statically link them.
