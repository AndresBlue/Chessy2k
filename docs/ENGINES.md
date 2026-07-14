# Engines

Chessy supports three move engines. Selection is persisted in `data/overlay_settings.json`.

## Stockfish

- Binary: `engines/stockfish/stockfish-windows-x86-64-avx2.exe`
- Config: `config/default.yaml` → `engines.stockfish`
- Client: `src/engines/stockfish_client.py`
- Env override: `CHESSY_STOCKFISH_PATH`

Human mode sets `UCI_LimitStrength` / `UCI_Elo`, then samples from MultiPV lines with practical-move heuristics (`HumanEngine`). Max effort disables limit strength.

## Reckless

Competitive UCI engine ([codedeliveryservice/Reckless](https://github.com/codedeliveryservice/Reckless), AGPL-3.0).

- Binary: `engines/reckless/reckless-windows-avx2.exe`
- Config: `engines.reckless` (`Threads`, `Hash`, `MultiPV`, `MoveOverhead`)
- Client: `src/engines/reckless_client.py` (`supports_uci_elo = False`)
- Env override: `CHESSY_RECKLESS_PATH`

Reckless does **not** implement Stockfish strength options. Humanization is entirely in Python: opening book → MultiPV search → `select_human_move` / think-time profiler. The same `HumanEngine` wrapper is used as for Stockfish.

To update the binary, download the matching OS/CPU build from Reckless releases into `engines/reckless/` and keep the filename pattern `reckless*.exe` (or set `engines.reckless.path`).

## Maia-3

Human-play modeling via the CSSLab Maia-3 package.

```bash
python scripts/setup_maia3.py --model maia3-23m
```

- Package: `third_party/maia3` (editable install)
- Weights cache: `data/maia3/cache` (gitignored; filled by setup)
- Models: `maia3-5m`, `maia3-23m`, `maia3-79m`
- Client: `src/engines/maia3_client.py`

Requires the same Python interpreter used to run Chessy (Torch must import). CUDA is preferred when available; the client falls back to CPU.

## Opening book

Optional Polyglot book at `data/openings/performance.bin`. See `data/openings/README.md`. Missing books are logged and ignored.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CHESSY_STOCKFISH_PATH` | Absolute path to Stockfish binary |
| `CHESSY_RECKLESS_PATH` | Absolute path to Reckless binary |
| `CHESSY_CLASSIFIER_CHECKPOINT` | Absolute path to vision `.pt` |
