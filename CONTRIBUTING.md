# Contributing

Thanks for contributing to Chessy.

## Workflow

1. Fork / branch from `master` (or the default branch).
2. Keep changes focused — one concern per PR when possible.
3. Run `pytest tests/ -v` locally before opening a PR.
4. Update docs if you change engines, paths, or configuration keys.

## Guidelines

- Do not reintroduce training pipelines, custom neural engines, or absolute Windows paths into the main app.
- Keep the repo self-contained: new assets go under `engines/`, `data/`, or `third_party/` with relative resolution.
- Prefer extending `StockfishClient` / `HumanEngine` over duplicating UCI process management.
- Match existing naming and module boundaries (`src/app`, `src/engines`, `src/vision`).

## Reporting issues

Include OS, Python version, selected engine, and a short log snippet from `data/logs/` when relevant.
