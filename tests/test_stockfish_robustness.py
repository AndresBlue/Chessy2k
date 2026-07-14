"""Stockfish client robustness: reader thread, cancel, restart."""

from __future__ import annotations

import io
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.app.errors import EngineError, EngineTimeout
from src.engines.stockfish_client import StockfishClient

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class _ScriptedProcess:
  def __init__(self, lines: list[str]):
    self._lines = list(lines)
    self.stdin = io.StringIO()
    self.stdout = _LineReader(self._lines)
    self._returncode = None

  def poll(self):
    return self._returncode

  def kill(self):
    self._returncode = -9

  def wait(self, timeout=None):
    self._returncode = 0


class _LineReader:
  def __init__(self, lines: list[str]):
    self._lines = lines
    self._delay = 0.0

  def readable(self):
    return bool(self._lines)

  def readline(self):
    if self._delay:
      time.sleep(self._delay)
    if not self._lines:
      return ""
    return self._lines.pop(0) + "\n"


class _BrokenStdin:
  def write(self, _text):
    raise OSError(22, "Invalid argument")

  def flush(self):
    raise OSError(22, "Invalid argument")


def _boot_lines() -> list[str]:
  return [
    "id name Stockfish",
    "uciok",
    "readyok",
    "readyok",
  ]


def test_interrupt_search_writes_stop_without_lock_deadlock():
  client = StockfishClient("stockfish")
  proc = _ScriptedProcess(_boot_lines())
  client._proc = proc
  client._start_reader_unlocked()
  client.interrupt_search()
  assert "stop" in proc.stdin.getvalue()


def test_analyze_restart_after_timeout_does_not_deadlock():
  info = "info depth 1 multipv 1 score cp 20 pv e2e4"
  first_run = _boot_lines() + [info]
  second_run = _boot_lines() + [info, "bestmove e2e4"]

  processes = [_ScriptedProcess(first_run), _ScriptedProcess(second_run)]

  def fake_popen(*_args, **_kwargs):
    return processes.pop(0)

  client = StockfishClient("stockfish", uci_timeout_s=0.2)
  with patch("src.engines.stockfish_client.subprocess.Popen", side_effect=fake_popen), patch(
    "src.engines.stockfish_client.ANALYZE_EXTRA_TIMEOUT_S",
    0.2,
  ):
    result = client.analyze(START_FEN, depth=1, movetime_ms=30)

  assert result.best_move.uci == "e2e4"
  client.stop()


def test_analyze_honours_cancel():
  boot = _boot_lines()
  proc = _ScriptedProcess(boot + ["info depth 1 multipv 1 score cp 10 pv e2e4"])
  proc.stdout._delay = 0.05
  cancelled = threading.Event()

  client = StockfishClient("stockfish", uci_timeout_s=2.0)
  client._proc = proc
  client._start_reader_unlocked()
  client._send_unlocked("ucinewgame")
  client._send_unlocked(f"position fen {START_FEN}")
  client._send_unlocked("go depth 20")

  def _cancel_soon():
    time.sleep(0.08)
    cancelled.set()

  threading.Thread(target=_cancel_soon, daemon=True).start()

  with client._lock:
    with pytest.raises(EngineError, match="cancelled"):
      client._analyze_locked(
        __import__("chess").Board(START_FEN),
        START_FEN,
        20,
        None,
        1,
        5.0,
        is_cancelled=cancelled.is_set,
      )
  client._stop_reader_unlocked()


def test_configure_strength_restarts_dead_process():
  dead = _ScriptedProcess([])
  dead._returncode = 1
  live = _ScriptedProcess(_boot_lines())
  client = StockfishClient("stockfish", uci_timeout_s=0.5)
  client._proc = dead

  with patch("src.engines.stockfish_client.subprocess.Popen", return_value=live):
    client.configure_strength(2023, limit_strength=True)

  assert client._proc is live
  assert "UCI_LimitStrength" in live.stdin.getvalue()
  assert "UCI_Elo value 2023" in live.stdin.getvalue()
  client.stop()


def test_configure_strength_recovers_from_broken_stdin():
  broken = _ScriptedProcess([])
  broken.stdin = _BrokenStdin()
  live = _ScriptedProcess(_boot_lines())
  client = StockfishClient("stockfish", uci_timeout_s=0.5)
  client._proc = broken

  with patch("src.engines.stockfish_client.subprocess.Popen", return_value=live):
    client.configure_strength(2023, limit_strength=True)

  assert client._proc is live
  assert "UCI_Elo value 2023" in live.stdin.getvalue()
  client.stop()
