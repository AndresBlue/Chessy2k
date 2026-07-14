# Configuration

Primary file: [`config/default.yaml`](../config/default.yaml). Paths are relative to the repository root unless absolute.

## `vision`

| Key | Meaning |
|-----|---------|
| `checkpoint` | Relative path to square classifier weights |
| `ambiguity_threshold` | Softmax confidence gate per square |
| `incremental` | Reuse unchanged squares between frames |
| `min_mean_confidence` | Reject captures below this mean confidence |
| `max_ambiguous_squares` | Reject captures with too many uncertain squares |

## `engines.stockfish`

| Key | Meaning |
|-----|---------|
| `path` | Stockfish binary (under `engines/stockfish/`) |
| `threads` / `hash_mb` | UCI Threads / Hash |
| `multipv` / `depth` / `skill_level` | Defaults for analysis |

## `engines.reckless`

| Key | Meaning |
|-----|---------|
| `path` | Reckless binary |
| `threads` / `hash_mb` / `multipv` | UCI options |
| `move_overhead_ms` | UCI `MoveOverhead` |

## `engines.maia3`

| Key | Meaning |
|-----|---------|
| `enabled` | Attempt to load Maia-3 in the overlay |
| `model` | `maia3-5m` / `maia3-23m` / `maia3-79m` |
| `cache_dir` | Hugging Face cache root |
| `device` / `use_amp` | Inference device preferences |
| `timing.*` | Think-time advisor for Maia suggestions |

## `tracker`

FEN transition / castling inference settings used when recovering game state across frames.

## `humanization`

| Key | Meaning |
|-----|---------|
| `enabled` | Default humanized play (UI Elo slider can force max effort) |
| `target_elo` | Default Elo target |
| `opening.*` | Book usage and calm-opening budgets |
| `timing.*` | Probe / normal / critical movetimes |
| `selection.*` | MultiPV practical-move probabilities |

Runtime UI preferences (theme, selected engine, Elo, region) are stored separately in `data/overlay_settings.json` and are gitignored.
