#!/usr/bin/env python3
"""Install Maia-3 into this repository and optionally cache Hugging Face weights."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo(root: Path) -> Path:
    repo = root / "third_party" / "maia3"
    if (repo / "pyproject.toml").exists():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning Maia-3 into {repo} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/CSSLab/maia3.git", str(repo)],
        check=True,
    )
    return repo


def pip_install(repo: Path) -> None:
    print(f"Installing Maia-3 editable from {repo} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(repo)],
        check=True,
    )


def cache_model(model: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pre-downloading {model} into {cache_dir} ...")
    print(f"Using interpreter: {sys.executable}")
    try:
        import maia3  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Maia-3 is not installed for this Python interpreter.\n"
            f"  Current interpreter: {sys.executable}\n"
            "Install with the same Python you will use to run Chessy:\n"
            f"  {sys.executable} scripts/setup_maia3.py\n"
            "Or, if the repo is already cloned:\n"
            f"  {sys.executable} scripts/setup_maia3.py --skip-clone --model maia3-23m"
        ) from exc
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maia3.cache",
            "--model",
            model,
            "--cache-dir",
            str(cache_dir),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Maia-3 for Chessy")
    parser.add_argument("--model", default="maia3-23m", help="Model alias to cache")
    parser.add_argument(
        "--cache-dir",
        default="data/maia3/cache",
        help="Hugging Face cache directory (relative to repo root)",
    )
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()

    root = project_root()
    repo = root / "third_party" / "maia3"
    if not args.skip_clone:
        repo = ensure_repo(root)
    elif not (repo / "pyproject.toml").exists():
        raise SystemExit(f"Maia-3 repo missing at {repo}. Run without --skip-clone.")

    if not args.skip_install:
        pip_install(repo)

    if not args.skip_cache:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = root / cache_dir
        cache_model(args.model, cache_dir)

    print("Maia-3 setup complete.")
    print("In the overlay, select engine 'Maia-3' (Elo slider maps to SelfElo/OppoElo).")


if __name__ == "__main__":
    main()
