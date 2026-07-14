"""Reckless UCI client.

Reckless speaks standard UCI but does not implement Stockfish strength options
(UCI_Elo / Skill Level). Humanization for Reckless is applied in HumanEngine
via MultiPV move selection, timing, and the opening book.
"""

from __future__ import annotations

from src.engines.stockfish_client import StockfishClient


class RecklessClient(StockfishClient):
    """UCI wrapper for Reckless (no LimitStrength / UCI_Elo)."""

    supports_uci_elo = False

    def __init__(
        self,
        path: str,
        threads: int = 8,
        hash_mb: int = 1024,
        multipv_default: int = 5,
        move_overhead_ms: int = 100,
        uci_timeout_s: float = 30.0,
    ):
        super().__init__(
            path=path,
            threads=threads,
            hash_mb=hash_mb,
            skill_level=20,
            ponder=False,
            limit_strength=False,
            uci_elo=2000,
            uci_timeout_s=uci_timeout_s,
        )
        self.multipv_default = multipv_default
        self.move_overhead_ms = move_overhead_ms

    def _configure_engine_options_unlocked(self) -> None:
        self._send_unlocked(f"setoption name Threads value {self.threads}")
        self._send_unlocked(f"setoption name Hash value {self.hash_mb}")
        self._send_unlocked(f"setoption name MultiPV value {self.multipv_default}")
        self._send_unlocked(
            f"setoption name MoveOverhead value {int(self.move_overhead_ms)}"
        )
