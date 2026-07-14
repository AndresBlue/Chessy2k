"""Stockfish UCI client with MultiPV, threads, hash configuration, and timeouts."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import chess

from src.app.elo_power import STOCKFISH_UCI_ELO_MAX, STOCKFISH_UCI_ELO_MIN, stockfish_uci_elo
from src.app.errors import EngineError, EngineTimeout
from src.app.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_UCI_TIMEOUT_S = 30.0
ANALYZE_EXTRA_TIMEOUT_S = 60.0
READ_POLL_S = 0.05


@dataclass
class EngineMove:
    uci: str
    san: str
    score_cp: int | None = None
    score_mate: int | None = None
    depth: int = 0
    pv: list[str] = field(default_factory=list)
    nodes: int = 0
    nps: int = 0
    wdl_permille: tuple[int, int, int] | None = None

    @property
    def score_str(self) -> str:
        if self.score_mate is not None:
            sign = "+" if self.score_mate > 0 else ""
            return f"#{sign}{self.score_mate}"
        if self.score_cp is not None:
            return f"{self.score_cp / 100:+.2f}"
        return "0.00"


@dataclass
class AnalysisResult:
    best_move: EngineMove
    top_moves: list[EngineMove]
    fen: str
    ponder: str | None = None
    time_ms: float = 0.0
    game_over: bool = False
    game_over_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "best_move": {
                "uci": self.best_move.uci,
                "san": self.best_move.san,
                "score": self.best_move.score_str,
                "pv": self.best_move.pv,
            },
            "top_moves": [
                {
                    "uci": m.uci,
                    "san": m.san,
                    "score": m.score_str,
                    "pv": m.pv,
                }
                for m in self.top_moves
            ],
            "time_ms": self.time_ms,
            "game_over": self.game_over,
        }


def _game_over_move(board: chess.Board) -> EngineMove:
    return EngineMove(uci="", san="(fin)", score_cp=0)


class StockfishClient:
    """UCI wrapper for Stockfish (base also reused by RecklessClient)."""

    # Subclasses that lack UCI_Elo / LimitStrength should set this False.
    supports_uci_elo: bool = True

    def __init__(
        self,
        path: str,
        threads: int = 8,
        hash_mb: int = 1024,
        skill_level: int = 20,
        ponder: bool = False,
        limit_strength: bool = False,
        uci_elo: int = 2000,
        uci_timeout_s: float = DEFAULT_UCI_TIMEOUT_S,
    ):
        self.path = path
        self.threads = threads
        self.hash_mb = hash_mb
        self.skill_level = skill_level
        self.ponder = ponder
        self.limit_strength = limit_strength
        self.uci_elo = stockfish_uci_elo(uci_elo)
        self.uci_timeout_s = uci_timeout_s
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._line_queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()

    def start(self) -> None:
        """Start the UCI process and configure engine options."""
        with self._lock:
            self._start_unlocked()

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
        self._proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._start_reader_unlocked()
        self._send_unlocked("uci")
        self._wait_for_unlocked("uciok")
        self._configure_engine_options_unlocked()
        self._send_unlocked("isready")
        self._wait_for_unlocked("readyok")

    def _configure_engine_options_unlocked(self) -> None:
        """Apply Stockfish-specific UCI options after `uciok`."""
        self._send_unlocked(f"setoption name Threads value {self.threads}")
        self._send_unlocked(f"setoption name Hash value {self.hash_mb}")
        self._send_unlocked(f"setoption name Skill Level value {self.skill_level}")
        self._send_unlocked(
            f"setoption name Ponder value {'true' if self.ponder else 'false'}"
        )
        self._apply_strength_options_unlocked()

    def _apply_strength_options_unlocked(self) -> None:
        if not self.supports_uci_elo:
            return
        self._send_unlocked(
            f"setoption name UCI_LimitStrength value "
            f"{'true' if self.limit_strength else 'false'}"
        )
        self._send_unlocked(f"setoption name UCI_Elo value {self.uci_elo}")

    def _process_alive_unlocked(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
            and self._proc.stdin is not None
        )

    def configure_strength(
        self,
        elo: int,
        *,
        limit_strength: bool = True,
    ) -> None:
        """Set human-like strength via UCI_Elo when the engine supports it."""
        self.uci_elo = stockfish_uci_elo(elo)
        self.limit_strength = limit_strength if self.supports_uci_elo else False
        if not self.supports_uci_elo:
            return
        with self._lock:
            if not self._process_alive_unlocked():
                self._restart_unlocked()
            try:
                self._apply_strength_options_unlocked()
                self._send_unlocked("isready")
                self._wait_for_unlocked("readyok")
            except (BrokenPipeError, OSError, ValueError, EngineError) as exc:
                log.warning("Stockfish strength configure failed (%s), restarting", exc)
                self._restart_unlocked()
                self._apply_strength_options_unlocked()
                self._send_unlocked("isready")
                self._wait_for_unlocked("readyok")

    def interrupt_search(self) -> None:
        """Send UCI stop to abort an in-flight search (safe from any thread)."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            log.debug("Stockfish stop failed: %s", exc)

    def _interrupt_search_unlocked(self) -> None:
        self.interrupt_search()

    def stop(self) -> None:
        """Terminate Stockfish process."""
        with self._lock:
            if self._proc is None:
                return
            proc = self._proc
            self._proc = None
            self._stop_reader_unlocked()
        try:
            if proc.stdin:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        self._drain_lines()

    def __enter__(self) -> "StockfishClient":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def _send_unlocked(self, cmd: str) -> None:
        if not self._process_alive_unlocked():
            raise EngineError("Stockfish not started")
        try:
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            proc = self._proc
            self._proc = None
            self._stop_reader_unlocked()
            self._drain_lines()
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
            raise EngineError(f"Stockfish pipe write failed: {exc}") from exc

    def _send(self, cmd: str) -> None:
        with self._lock:
            self._send_unlocked(cmd)

    def _drain_lines(self) -> None:
        while True:
            try:
                self._line_queue.get_nowait()
            except queue.Empty:
                break

    def _start_reader_unlocked(self) -> None:
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="chessy-stockfish-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_reader_unlocked(self) -> None:
        self._reader_stop.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

    def _reader_loop(self) -> None:
        while not self._reader_stop.is_set():
            proc = self._proc
            if proc is None or proc.stdout is None:
                time.sleep(READ_POLL_S)
                continue
            try:
                line = proc.stdout.readline()
            except (OSError, ValueError):
                break
            if not line:
                if proc.poll() is not None:
                    self._line_queue.put(None)
                    break
                time.sleep(READ_POLL_S)
                continue
            self._line_queue.put(line.strip())

    def _read_line_unlocked(
        self,
        deadline: float,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> str:
        while time.monotonic() < deadline:
            if is_cancelled and is_cancelled():
                self._interrupt_search_unlocked()
                raise EngineError("Analysis cancelled")
            remaining = deadline - time.monotonic()
            try:
                line = self._line_queue.get(timeout=min(READ_POLL_S, max(0.001, remaining)))
            except queue.Empty:
                continue
            if line is None:
                raise EngineError("Stockfish process ended unexpectedly")
            return line
        raise EngineTimeout(
            f"Timed out waiting for Stockfish response ({self.uci_timeout_s}s)"
        )

    def _wait_for_unlocked(
        self,
        token: str,
        timeout_s: float | None = None,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> list[str]:
        deadline = time.monotonic() + (timeout_s or self.uci_timeout_s)
        lines: list[str] = []
        while time.monotonic() < deadline:
            line = self._read_line_unlocked(deadline, is_cancelled=is_cancelled)
            lines.append(line)
            if token in line:
                return lines
        raise EngineTimeout(f"Timed out waiting for '{token}'")

    def _restart_unlocked(self) -> None:
        self._stop_reader_unlocked()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
        self._drain_lines()
        self._start_unlocked()

    def analyze(
        self,
        fen: str,
        depth: int | None = None,
        movetime_ms: int | None = None,
        multipv: int = 1,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AnalysisResult:
        """Analyze position and return best move + top-k lines."""
        self.start()
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise EngineError(f"Invalid FEN: {fen}") from exc

        if board.is_game_over():
            reason = board.result(claim_draw=True) or "game over"
            move = _game_over_move(board)
            return AnalysisResult(
                best_move=move,
                top_moves=[move],
                fen=fen,
                game_over=True,
                game_over_reason=str(reason),
            )

        analyze_timeout = self.uci_timeout_s + ANALYZE_EXTRA_TIMEOUT_S
        if movetime_ms:
            analyze_timeout += movetime_ms / 1000.0

        with self._lock:
            try:
                return self._analyze_locked(
                    board,
                    fen,
                    depth,
                    movetime_ms,
                    multipv,
                    analyze_timeout,
                    is_cancelled=is_cancelled,
                )
            except (EngineTimeout, EngineError) as exc:
                if is_cancelled and is_cancelled():
                    raise EngineError("Analysis cancelled") from exc
                log.warning("Stockfish analyze failed (%s), restarting", exc)
                try:
                    self._restart_unlocked()
                    return self._analyze_locked(
                        board,
                        fen,
                        depth,
                        movetime_ms,
                        multipv,
                        analyze_timeout,
                        is_cancelled=is_cancelled,
                    )
                except Exception as retry_exc:
                    raise EngineError(str(retry_exc)) from retry_exc

    def _analyze_locked(
        self,
        board: chess.Board,
        fen: str,
        depth: int | None,
        movetime_ms: int | None,
        multipv: int,
        analyze_timeout: float,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AnalysisResult:
        t0 = time.perf_counter()
        deadline = time.monotonic() + analyze_timeout

        self._send_unlocked(f"setoption name MultiPV value {multipv}")
        self._send_unlocked("ucinewgame")
        self._send_unlocked(f"position fen {fen}")

        go_cmd = "go"
        if depth is not None:
            go_cmd += f" depth {depth}"
        if movetime_ms is not None:
            go_cmd += f" movetime {movetime_ms}"
        if depth is None and movetime_ms is None:
            go_cmd += " depth 15"
        self._send_unlocked(go_cmd)

        infos: dict[int, dict[str, Any]] = {}
        bestmove_uci = ""
        ponder = None

        while time.monotonic() < deadline:
            if is_cancelled and is_cancelled():
                self._send_unlocked("stop")
                raise EngineError("Analysis cancelled")
            line = self._read_line_unlocked(deadline, is_cancelled=is_cancelled)
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[1]
                    if token in ("(none)", "0000"):
                        move = _game_over_move(board)
                        elapsed_ms = (time.perf_counter() - t0) * 1000
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
            raise EngineTimeout("Timed out waiting for bestmove")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        top_moves = self._build_top_moves(board, bestmove_uci, infos, multipv)
        return AnalysisResult(
            best_move=top_moves[0],
            top_moves=top_moves,
            fen=fen,
            ponder=ponder,
            time_ms=elapsed_ms,
        )

    @staticmethod
    def _build_top_moves(
        board: chess.Board,
        bestmove_uci: str,
        infos: dict[int, dict[str, Any]],
        multipv: int,
    ) -> list[EngineMove]:
        top_moves: list[EngineMove] = []
        for i in range(1, multipv + 1):
            info = infos.get(i, infos.get(1, {}))
            pv_uci = info.get("pv", [bestmove_uci] if bestmove_uci else [])
            move_uci = pv_uci[0] if pv_uci else bestmove_uci
            if not move_uci:
                continue
            try:
                move = chess.Move.from_uci(move_uci)
                san = board.san(move)
            except (ValueError, chess.IllegalMoveError):
                continue
            top_moves.append(
                EngineMove(
                    uci=move_uci,
                    san=san,
                    score_cp=info.get("score_cp"),
                    score_mate=info.get("score_mate"),
                    depth=info.get("depth", 0),
                    pv=pv_uci,
                    nodes=info.get("nodes", 0),
                    nps=info.get("nps", 0),
                    wdl_permille=info.get("wdl_permille"),
                )
            )

        if not top_moves and bestmove_uci:
            move = chess.Move.from_uci(bestmove_uci)
            top_moves.append(
                EngineMove(uci=bestmove_uci, san=board.san(move))
            )
        if not top_moves:
            top_moves.append(_game_over_move(board))
        return top_moves

    @staticmethod
    def _parse_info(line: str) -> dict[str, Any] | None:
        parts = line.split()
        if "pv" not in parts:
            return None
        result: dict[str, Any] = {}
        i = 1
        while i < len(parts):
            key = parts[i]
            if key == "depth" and i + 1 < len(parts):
                result["depth"] = int(parts[i + 1])
                i += 2
            elif key == "multipv" and i + 1 < len(parts):
                result["multipv"] = int(parts[i + 1])
                i += 2
            elif key == "score":
                if i + 2 < len(parts):
                    if parts[i + 1] == "cp":
                        result["score_cp"] = int(parts[i + 2])
                    elif parts[i + 1] == "mate":
                        result["score_mate"] = int(parts[i + 2])
                    i += 3
                else:
                    i += 1
            elif key == "nodes" and i + 1 < len(parts):
                result["nodes"] = int(parts[i + 1])
                i += 2
            elif key == "nps" and i + 1 < len(parts):
                result["nps"] = int(parts[i + 1])
                i += 2
            elif key == "wdl" and i + 3 < len(parts):
                result["wdl_permille"] = (
                    int(parts[i + 1]),
                    int(parts[i + 2]),
                    int(parts[i + 3]),
                )
                i += 4
            elif key == "pv":
                result["pv"] = parts[i + 1 :]
                break
            else:
                i += 1
        return result if "pv" in result else None
