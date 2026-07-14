"""Maia-3 UCI client for human-like move prediction."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.app.logging_config import get_logger
from src.engines.stockfish_client import StockfishClient

log = get_logger(__name__)

DEFAULT_MAIA3_TIMEOUT_S = 180.0
MAIA3_ELO_MIN = 0
MAIA3_ELO_MAX = 5000


def is_maia3_installed() -> bool:
    try:
        import maia3  # noqa: F401
    except ImportError:
        return False
    return True


def maia3_repo_path(project_root: Path) -> Path:
    return project_root / "third_party" / "maia3"


def build_maia3_command(project_root: Path, cfg: dict[str, Any]) -> list[str]:
    """Build argv to launch `python -m maia3.uci` with project settings."""
    cmd = [sys.executable, "-m", "maia3.uci"]
    model = str(cfg.get("model", "maia3-5m"))
    cmd += ["--model", model]

    if cfg.get("use_uci_history", True):
        cmd.append("--use-uci-history")

    cache_dir = cfg.get("cache_dir")
    if cache_dir:
        path = Path(cache_dir)
        if not path.is_absolute():
            path = project_root / path
        path.mkdir(parents=True, exist_ok=True)
        cmd += ["--cache-dir", str(path)]

    device = str(cfg.get("device", "auto") or "auto").lower()
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    cmd += ["--device", device]
    if device.startswith("cuda"):
        log.info("Maia-3 will use GPU device=%s amp=%s", device, cfg.get("use_amp", True))

    temperature = cfg.get("temperature")
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]

    top_p = cfg.get("top_p")
    if top_p is not None:
        cmd += ["--top-p", str(top_p)]

    multipv = cfg.get("multipv")
    if multipv is not None:
        cmd += ["--multipv", str(multipv)]

    if cfg.get("use_amp") is False:
        cmd.append("--no-use-amp")

    default_elo = int(cfg.get("default_elo", 1500))
    cmd += ["--elo", str(default_elo)]

    if cfg.get("local_files_only"):
        cmd.append("--local-files-only")

    return cmd


class Maia3Client(StockfishClient):
    """UCI wrapper around the Maia-3 inference process."""

    def __init__(
        self,
        command: list[str],
        *,
        default_elo: int = 1500,
        multipv: int = 3,
        model_label: str = "Maia-3",
        uci_timeout_s: float = DEFAULT_MAIA3_TIMEOUT_S,
    ):
        self._command = list(command)
        self.model_label = model_label
        self.multipv_default = max(1, int(multipv))
        super().__init__(
            path=self._command[0],
            threads=1,
            hash_mb=16,
            uci_timeout_s=uci_timeout_s,
        )
        self.uci_elo = max(MAIA3_ELO_MIN, min(MAIA3_ELO_MAX, int(default_elo)))
        self._self_elo = self.uci_elo
        self._oppo_elo = self.uci_elo

    def _start_unlocked(self) -> None:
        if self._process_alive_unlocked():
            return
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            self._stop_reader_unlocked()
            self._drain_lines()
            try:
                proc.kill()
            except OSError:
                pass
        self._stop_reader_unlocked()
        self._drain_lines()
        log.info("Starting Maia-3 UCI: %s", " ".join(self._command))
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_reader_unlocked()
        self._send_unlocked("uci")
        self._wait_for_unlocked("uciok", timeout_s=30.0)
        self._apply_strength_options_unlocked()
        self._send_unlocked(f"setoption name MultiPV value {self.multipv_default}")
        self._send_unlocked("isready")
        self._wait_for_unlocked("readyok", timeout_s=self.uci_timeout_s)

    def _apply_strength_options_unlocked(self) -> None:
        self_elo = max(MAIA3_ELO_MIN, min(MAIA3_ELO_MAX, int(self._self_elo)))
        oppo = max(MAIA3_ELO_MIN, min(MAIA3_ELO_MAX, int(self._oppo_elo)))
        self._send_unlocked(f"setoption name Elo value {self_elo}")
        self._send_unlocked(f"setoption name SelfElo value {self_elo}")
        self._send_unlocked(f"setoption name OppoElo value {oppo}")

    def configure_elos(self, self_elo: int, oppo_elo: int | None = None) -> None:
        """Set Maia-3 SelfElo (side to move) and OppoElo (opponent rating)."""
        self._self_elo = max(MAIA3_ELO_MIN, min(MAIA3_ELO_MAX, int(self_elo)))
        self._oppo_elo = max(
            MAIA3_ELO_MIN,
            min(MAIA3_ELO_MAX, int(oppo_elo if oppo_elo is not None else self_elo)),
        )
        self.uci_elo = self._self_elo
        with self._lock:
            if not self._process_alive_unlocked():
                return
            try:
                self._apply_strength_options_unlocked()
                self._send_unlocked("isready")
                self._wait_for_unlocked("readyok", timeout_s=self.uci_timeout_s)
            except Exception as exc:
                log.warning("Maia-3 elo configure failed (%s), restarting", exc)
                self._restart_unlocked()

    def configure_strength(
        self,
        elo: int,
        *,
        limit_strength: bool = True,
    ) -> None:
        del limit_strength
        self.configure_elos(elo, elo)

    def _analyze_locked(
        self,
        board,
        fen: str,
        depth: int | None,
        movetime_ms: int | None,
        multipv: int,
        analyze_timeout: float,
        *,
        is_cancelled=None,
    ):
        del depth, movetime_ms
        import time

        t0 = time.perf_counter()
        deadline = time.monotonic() + analyze_timeout

        self._send_unlocked(f"setoption name MultiPV value {multipv}")
        self._send_unlocked("ucinewgame")
        self._send_unlocked(f"position fen {fen}")
        self._send_unlocked("go nodes 1")

        infos: dict[int, dict[str, Any]] = {}
        bestmove_uci = ""
        ponder = None

        while time.monotonic() < deadline:
            if is_cancelled and is_cancelled():
                self._send_unlocked("stop")
                from src.app.errors import EngineError

                raise EngineError("Analysis cancelled")
            line = self._read_line_unlocked(deadline, is_cancelled=is_cancelled)
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[1]
                    if token in ("(none)", "0000"):
                        from src.engines.stockfish_client import (
                            AnalysisResult,
                            _game_over_move,
                        )

                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        move = _game_over_move(board)
                        return AnalysisResult(
                            best_move=move,
                            top_moves=[move],
                            fen=fen,
                            game_over=True,
                            game_over_reason="no legal moves",
                            time_ms=elapsed_ms,
                        )
                    bestmove_uci = token
                if len(parts) > 3 and parts[2] == "ponder":
                    ponder = parts[3]
                break
            if line.startswith("info"):
                parsed = self._parse_info(line)
                if parsed and "multipv" in parsed:
                    infos[parsed["multipv"]] = parsed
        else:
            self._send_unlocked("stop")
            from src.app.errors import EngineTimeout

            raise EngineTimeout("Timed out waiting for Maia-3 bestmove")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        top_moves = self._build_top_moves(board, bestmove_uci, infos, multipv)
        from src.engines.stockfish_client import AnalysisResult

        return AnalysisResult(
            best_move=top_moves[0],
            top_moves=top_moves,
            fen=fen,
            ponder=ponder,
            time_ms=elapsed_ms,
        )


def format_maia3_wdl(move) -> str | None:
    wdl = getattr(move, "wdl_permille", None)
    if not wdl:
        return None
    win, draw, loss = wdl
    return f"W{win / 10:.0f}% D{draw / 10:.0f}% L{loss / 10:.0f}%"


def find_maia3_launcher(model: str) -> list[str] | None:
    """Return preset launcher command if installed on PATH."""
    alias = {
        "5m": "maia3-5m",
        "maia3-5m": "maia3-5m",
        "23m": "maia3-23m",
        "maia3-23m": "maia3-23m",
        "79m": "maia3-79m",
        "maia3-79m": "maia3-79m",
    }.get(model.lower().replace("_", "-"))
    if alias and shutil.which(alias):
        return [alias]
    return None
