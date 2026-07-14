"""Fast vision + Stockfish/Reckless/Maia-3 analysis for overlay client."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.app.config import load_config
from src.app.elo_power import is_max_effort
from src.app.logging_config import get_logger
from src.app.overlay_theme import (
    OVERLAY_ENGINE_MAIA3,
    OVERLAY_ENGINE_RECKLESS,
    OVERLAY_ENGINE_STOCKFISH,
)
from src.app.reckless_path import find_reckless_path
from src.chess_core.fen_utils import (
    default_castling_for_placement,
    fen_from_matrix,
    matrix_from_screen_view,
    matrix_to_placement,
)
from src.chess_core.legal_validator import (
    validate_fen_for_analysis,
    validate_placement_sanity,
)
from src.chess_core.state_tracker import GameStateTracker, TrackerConfig
from src.engines.human_engine import HumanEngine
from src.engines.humanization_config import HumanizationConfig, humanization_from_config
from src.engines.stockfish_client import StockfishClient
from src.vision.pipeline import VisionPipeline, VisionResult
from src.vision.visualizer import (
    draw_best_move_arrow,
    draw_opponent_move_arrow,
    draw_vision_predictions_on_capture,
)

log = get_logger(__name__)


def _vision_confidence_ok(
    vision: VisionResult,
    *,
    min_mean_confidence: float,
    max_ambiguous_squares: int,
) -> tuple[bool, str]:
    """Return (ok, reason) for vision quality gate."""
    if vision.confidence is None or vision.confidence.size == 0:
        return False, "Sin datos de confianza de vision"
    mean_conf = float(vision.confidence.mean())
    ambiguous = len(vision.ambiguous_squares)
    if mean_conf < min_mean_confidence:
        return (
            False,
            f"Confianza media baja ({mean_conf:.2f} < {min_mean_confidence:.2f})",
        )
    if ambiguous > max_ambiguous_squares:
        return (
            False,
            f"Demasiadas casillas ambiguas ({ambiguous} > {max_ambiguous_squares})",
        )
    return True, ""


def _forced_side_fen(
    board_matrix: list[list[str | None]],
    side: str,
) -> tuple[str, list[list[str | None]], str, list[str]]:
    """Build a fresh FEN with side-to-move forced to the selected player."""
    view = side if side in ("white", "black") else "white"
    standard_matrix = matrix_from_screen_view(board_matrix, view)
    placement = matrix_to_placement(standard_matrix)
    warnings = validate_placement_sanity(placement)
    turn = "w" if view == "white" else "b"
    castling = default_castling_for_placement(placement)
    fen = fen_from_matrix(
        standard_matrix,
        turn=turn,
        castling=castling,
        en_passant="-",
    )
    return fen, standard_matrix, view, warnings


@dataclass
class FastEngineConfig:
    movetime_ms: int = 250
    multipv: int = 1
    threads: int = 4
    hash_mb: int = 256
    humanization: HumanizationConfig | None = None


PHASE_LABELS = {
    "opening": "Apertura",
    "middlegame": "Medio juego",
    "endgame": "Final",
}

CRITICALITY_LABELS = {
    "low": "baja",
    "medium": "media",
    "high": "alta",
}


@dataclass
class AnalysisOutput:
    fen: str
    best_move_san: str
    best_move_uci: str
    evaluation: str
    vision_ms: float
    engine_ms: float
    total_ms: float
    fen_ms: float = 0.0
    validation_ms: float = 0.0
    vision_device: str = "cpu"
    inference_mode: str = "full"
    squares_updated: int = 64
    orientation: str = "white"
    annotated_image: np.ndarray | None = None
    error: str | None = None
    board_matrix: list[list[str | None]] | None = None
    confidence: np.ndarray | None = None
    ambiguous_squares: list[tuple[int, int]] = field(default_factory=list)
    board_bbox: tuple[int, int, int, int] | None = None
    humanized: bool = False
    phase: str = ""
    criticality: str = ""
    planned_ms: int = 0
    think_note: str = ""
    from_book: bool = False
    target_elo: int = 0
    tracker_status: str = ""
    tracker_warnings: list[str] = field(default_factory=list)
    engine_mode: str = ""
    human_explanation: str = ""
    eval_bar_fraction: float | None = None
    eval_bar_post_fraction: float | None = None
    opponent_move_uci: str | None = None
    opponent_move_san: str | None = None


def compose_preview(
    image_bgr: np.ndarray,
    result: AnalysisOutput,
    *,
    show_predictions: bool,
) -> np.ndarray:
    """Build preview image: optional vision labels + move arrow."""
    img = image_bgr.copy()
    if (
        show_predictions
        and result.board_matrix is not None
        and result.confidence is not None
    ):
        img = draw_vision_predictions_on_capture(
            img,
            result.board_matrix,
            result.confidence,
            result.ambiguous_squares,
            result.board_bbox,
        )
    if result.best_move_uci and not result.error:
        img = draw_best_move_arrow(
            img,
            result.best_move_uci,
            orientation=result.orientation,
            region=result.board_bbox,
        )
    if result.opponent_move_uci and not result.error:
        img = draw_opponent_move_arrow(
            img,
            result.opponent_move_uci,
            orientation=result.orientation,
            region=result.board_bbox,
        )
    return img


def _resolve_reckless_binary(
    project_root: Path,
    reckless_path: str | None,
    cfg: dict,
) -> str | None:
    """Resolve Reckless executable from explicit path, discovery, or config."""
    if reckless_path:
        candidate = Path(reckless_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.exists():
            return str(candidate)
        log.warning("Reckless path not found: %s", candidate)
        return None

    found = find_reckless_path(project_root)
    if found is not None:
        return str(found)

    reckless_cfg = dict(cfg.get("engines", {}).get("reckless", {}) or {})
    configured = reckless_cfg.get("path")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.exists():
            return str(candidate)
        log.warning("Configured Reckless path not found: %s", candidate)
    return None


class FastAnalyzer:
    """Reuse vision model and UCI engine processes across captures."""

    def __init__(
        self,
        stockfish_path: str,
        classifier_path: str,
        config: FastEngineConfig | None = None,
        project_root: Path | None = None,
        maia_model: str | None = None,
        reckless_path: str | None = None,
    ):
        self.config = config or FastEngineConfig()
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        cfg = load_config()
        self.humanization = self.config.humanization or humanization_from_config(cfg)

        self.reckless = None
        self.human_engine_reckless: HumanEngine | None = None
        resolved_reckless = _resolve_reckless_binary(
            self.project_root, reckless_path, cfg
        )
        if resolved_reckless:
            from src.engines.reckless_client import RecklessClient

            try:
                reckless_cfg = dict(cfg.get("engines", {}).get("reckless", {}) or {})
                self.reckless = RecklessClient(
                    path=resolved_reckless,
                    threads=int(reckless_cfg.get("threads", self.config.threads)),
                    hash_mb=int(reckless_cfg.get("hash_mb", self.config.hash_mb)),
                    multipv_default=int(
                        reckless_cfg.get("multipv", max(self.config.multipv, 5))
                    ),
                    move_overhead_ms=int(reckless_cfg.get("move_overhead_ms", 100)),
                    uci_timeout_s=float(reckless_cfg.get("uci_timeout_s", 30.0)),
                )
                self.human_engine_reckless = HumanEngine(
                    self.reckless,
                    config=self.humanization,
                    project_root=self.project_root,
                )
                self.reckless.start()
                log.info("Reckless overlay engine loaded: %s", resolved_reckless)
            except Exception:
                log.exception("Failed to load Reckless overlay engine from %s", resolved_reckless)
                self.reckless = None
                self.human_engine_reckless = None
        else:
            log.warning(
                "Reckless binary not found (pass reckless_path or place under engines/reckless/)"
            )

        self.maia3_client = None
        self.maia3_model_label = None
        self.maia3_timing_config = None
        self._maia3_cfg: dict = {}
        maia_cfg = dict(cfg.get("engines", {}).get("maia3", {}) or {})
        if maia_model:
            maia_cfg["model"] = maia_model
        self._maia3_cfg = maia_cfg
        if maia_cfg.get("enabled", False):
            from src.engines.maia3_client import is_maia3_installed
            from src.engines.maia3_timing import maia_timing_config_from_dict

            self.maia3_timing_config = maia_timing_config_from_dict(maia_cfg.get("timing"))

            if is_maia3_installed():
                try:
                    self._create_maia3_client(maia_cfg)
                    log.info("Maia-3 overlay engine configured: %s", self.maia3_model_label)
                except Exception:
                    log.exception("Failed to configure Maia-3 overlay engine")
            else:
                log.warning(
                    "Maia-3 enabled in config but package not installed. "
                    "Run: python scripts/setup_maia3.py"
                )

        tracker_cfg = cfg.get("tracker", {})
        vision_cfg = cfg.get("vision", {})
        self.min_mean_confidence = float(vision_cfg.get("min_mean_confidence", 0.5))
        self.max_ambiguous_squares = int(vision_cfg.get("max_ambiguous_squares", 20))
        self.tracker = GameStateTracker(
            TrackerConfig(
                infer_castling=bool(tracker_cfg.get("infer_castling", True)),
                infer_en_passant=bool(tracker_cfg.get("infer_en_passant", False)),
                max_search_ply=int(tracker_cfg.get("max_search_ply", 4)),
                transition_max_nodes=int(tracker_cfg.get("transition_max_nodes", 5000)),
                transition_max_seconds=float(
                    tracker_cfg.get("transition_max_seconds", 0.4)
                ),
            )
        )
        self.vision = VisionPipeline.from_config(cfg, checkpoint_path=classifier_path)
        self.stockfish = StockfishClient(
            path=stockfish_path,
            threads=self.config.threads,
            hash_mb=self.config.hash_mb,
        )
        self.human_engine = HumanEngine(
            self.stockfish,
            config=self.humanization,
            project_root=self.project_root,
        )
        self.stockfish.start()
        self.vision.warmup()
        self._lock = threading.Lock()

    @property
    def vision_device(self) -> str:
        return self.vision.device_label

    @property
    def has_maia3_engine(self) -> bool:
        return self.maia3_client is not None

    @property
    def maia3_model_info(self) -> str | None:
        return getattr(self, "maia3_model_label", None)

    @property
    def has_reckless_engine(self) -> bool:
        return self.reckless is not None

    def _create_maia3_client(self, maia_cfg: dict) -> None:
        from src.app.overlay_theme import normalize_maia3_model
        from src.engines.maia3_client import Maia3Client, build_maia3_command

        cfg = dict(maia_cfg)
        model = normalize_maia3_model(cfg.get("model"))
        cfg["model"] = model
        maia_cmd = build_maia3_command(self.project_root, cfg)
        self.maia3_model_label = model
        self.maia3_client = Maia3Client(
            maia_cmd,
            default_elo=int(cfg.get("default_elo", 1500)),
            multipv=int(cfg.get("multipv", 5)),
            model_label=model,
            uci_timeout_s=float(cfg.get("uci_timeout_s", 180)),
        )
        self._maia3_cfg = cfg

    def set_maia3_model(self, model: str) -> str:
        """Recreate the Maia-3 UCI client for a different checkpoint size."""
        from src.app.overlay_theme import normalize_maia3_model
        from src.engines.maia3_client import is_maia3_installed

        target = normalize_maia3_model(model)
        with self._lock:
            if (
                self.maia3_client is not None
                and self.maia3_model_label == target
            ):
                return target
            if not is_maia3_installed():
                raise RuntimeError("Maia-3 package not installed")
            if self.maia3_client is not None:
                try:
                    self.maia3_client.interrupt_search()
                except Exception:
                    pass
                try:
                    self.maia3_client.stop()
                except Exception:
                    pass
                self.maia3_client = None
            cfg = dict(self._maia3_cfg or {})
            cfg.setdefault("enabled", True)
            cfg["model"] = target
            self._create_maia3_client(cfg)
            log.info("Maia-3 model switched to %s", target)
            return target

    def reset_vision_cache(self) -> None:
        acquired = self._lock.acquire(timeout=0.5)
        if not acquired:
            log.warning("reset_vision_cache skipped: analyzer busy")
            return
        try:
            self.vision.reset_cache()
            self.tracker.reset()
        finally:
            self._lock.release()

    def invalidate_vision_cache(self) -> None:
        """Clear incremental vision cache without resetting tracker history."""
        acquired = self._lock.acquire(timeout=0.1)
        if not acquired:
            return
        try:
            self.vision.reset_cache()
        finally:
            self._lock.release()

    def interrupt(self) -> None:
        """Abort in-flight engine search (safe from any thread)."""
        self.stockfish.interrupt_search()
        if self.reckless is not None:
            self.reckless.interrupt_search()
        if self.maia3_client is not None:
            self.maia3_client.interrupt_search()

    def close(self) -> None:
        self.stockfish.stop()
        if self.reckless is not None:
            self.reckless.stop()
        if self.maia3_client is not None:
            self.maia3_client.stop()

    def analyze(
        self,
        image_bgr: np.ndarray,
        side: str = "white",
        *,
        show_predictions: bool = True,
        human_mode: bool | None = None,
        target_elo: int | None = None,
        on_thinking: Callable[[int], None] | None = None,
        analysis_id: int = 0,
        is_cancelled: Callable[[], bool] | None = None,
        engine_mode: str = OVERLAY_ENGINE_STOCKFISH,
    ) -> AnalysisOutput:
        t0 = time.perf_counter()
        try:
            log.info("analysis[%s] vision:start", analysis_id)
            with self._lock:
                vision = self.vision.process_image(image_bgr, side=side, fast_mode=True)
            log.info(
                "analysis[%s] vision:done %.0f ms mode=%s updated=%s",
                analysis_id,
                vision.time_ms,
                vision.inference_mode,
                vision.squares_updated,
            )
            return self.analyze_precomputed(
                image_bgr,
                vision,
                side=side,
                show_predictions=show_predictions,
                human_mode=human_mode,
                target_elo=target_elo,
                on_thinking=on_thinking,
                analysis_id=analysis_id,
                is_cancelled=is_cancelled,
                started_at=t0,
                engine_mode=engine_mode,
            )
        except Exception as exc:
            log.exception("Analysis failed before engine (id=%s)", analysis_id)
            return AnalysisOutput(
                fen="",
                best_move_san="",
                best_move_uci="",
                evaluation="",
                vision_ms=0.0,
                engine_ms=0.0,
                total_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )

    def analyze_precomputed(
        self,
        image_bgr: np.ndarray,
        vision: VisionResult,
        side: str = "white",
        *,
        show_predictions: bool = True,
        human_mode: bool | None = None,
        target_elo: int | None = None,
        on_thinking: Callable[[int], None] | None = None,
        analysis_id: int = 0,
        is_cancelled: Callable[[], bool] | None = None,
        started_at: float | None = None,
        engine_mode: str = OVERLAY_ENGINE_STOCKFISH,
    ) -> AnalysisOutput:
        t0 = started_at or time.perf_counter()
        max_effort = target_elo is not None and is_max_effort(target_elo)
        use_human = (
            False
            if max_effort
            else (self.humanization.enabled if human_mode is None else human_mode)
        )
        elo = (
            self.humanization.target_elo
            if target_elo is None or max_effort
            else target_elo
        )

        def _cancelled() -> bool:
            return is_cancelled is not None and is_cancelled()

        try:
            if _cancelled():
                raise RuntimeError("Analysis cancelled")

            ok, conf_reason = _vision_confidence_ok(
                vision,
                min_mean_confidence=self.min_mean_confidence,
                max_ambiguous_squares=self.max_ambiguous_squares,
            )
            if not ok:
                log.warning("analysis[%s] vision gate: %s", analysis_id, conf_reason)
                base = AnalysisOutput(
                    fen="",
                    best_move_san="",
                    best_move_uci="",
                    evaluation="",
                    vision_ms=vision.time_ms,
                    engine_ms=0.0,
                    total_ms=(time.perf_counter() - t0) * 1000,
                    vision_device=vision.device_label,
                    inference_mode=vision.inference_mode,
                    squares_updated=vision.squares_updated,
                    orientation=side,
                    board_matrix=vision.board_matrix,
                    confidence=vision.confidence,
                    ambiguous_squares=list(vision.ambiguous_squares),
                    board_bbox=vision.board_bbox,
                    humanized=use_human,
                    target_elo=elo if use_human else 0,
                    tracker_status="vision_error",
                    tracker_warnings=[conf_reason],
                    error=f"Vision poco confiable: {conf_reason}",
                )
                base.annotated_image = compose_preview(
                    image_bgr, base, show_predictions=show_predictions
                )
                return base

            log.info("analysis[%s] fen:start", analysis_id)
            t_fen = time.perf_counter()
            fen, _standard_matrix, view_orientation, placement_warnings = _forced_side_fen(
                vision.board_matrix,
                side,
            )
            fen_ms = (time.perf_counter() - t_fen) * 1000
            log.info(
                "analysis[%s] fen:done %.0f ms status=%s %s",
                analysis_id,
                fen_ms,
                "vision_error" if placement_warnings else "forced_side",
                fen,
            )
            log.info("analysis[%s] validation:start", analysis_id)
            t_validation = time.perf_counter()
            validation = validate_fen_for_analysis(fen)
            validation_ms = (time.perf_counter() - t_validation) * 1000
            log.info(
                "analysis[%s] validation:%s %.0f ms",
                analysis_id,
                "ok" if validation.is_valid else "invalid",
                validation_ms,
            )
            base = AnalysisOutput(
                fen=fen,
                best_move_san="",
                best_move_uci="",
                evaluation="",
                vision_ms=vision.time_ms,
                engine_ms=0.0,
                total_ms=(time.perf_counter() - t0) * 1000,
                fen_ms=fen_ms,
                validation_ms=validation_ms,
                vision_device=vision.device_label,
                inference_mode=vision.inference_mode,
                squares_updated=vision.squares_updated,
                orientation=view_orientation,
                board_matrix=vision.board_matrix,
                confidence=vision.confidence,
                ambiguous_squares=list(vision.ambiguous_squares),
                board_bbox=vision.board_bbox,
                humanized=use_human,
                target_elo=elo if use_human else 0,
                tracker_status="vision_error" if placement_warnings else "forced_side",
                tracker_warnings=list(placement_warnings),
            )
            if placement_warnings:
                detail = "; ".join(placement_warnings)
                base.error = f"Vision produjo una posicion imposible: {detail}"
                log.warning("analysis[%s] placement blocked: %s", analysis_id, detail)
                base.annotated_image = compose_preview(
                    image_bgr, base, show_predictions=show_predictions
                )
                return base
            if not validation.is_valid:
                detail = "; ".join(validation.errors) if validation.errors else "FEN invalido"
                base.error = detail
                log.warning("analysis[%s] fen invalid: %s", analysis_id, detail)
                base.annotated_image = compose_preview(
                    image_bgr, base, show_predictions=show_predictions
                )
                return base

            if use_human:
                use_maia3 = (
                    engine_mode == OVERLAY_ENGINE_MAIA3
                    and self.maia3_client is not None
                )
                use_reckless = (
                    engine_mode == OVERLAY_ENGINE_RECKLESS
                    and self.human_engine_reckless is not None
                )
                if use_maia3:
                    import chess

                    from src.engines.maia3_timing import advise_maia_think_time
                    from src.engines.maia_eval import analyze_maia_line

                    player_elo = elo if elo else 1500
                    log.info("analysis[%s] engine:maia3:start elo=%s", analysis_id, player_elo)
                    line = analyze_maia_line(
                        self.maia3_client,
                        fen,
                        player_elo=player_elo,
                        multipv=self.maia3_client.multipv_default,
                        is_cancelled=_cancelled,
                    )
                    log.info(
                        "analysis[%s] engine:maia3:done %.0f ms move=%s reply=%s",
                        analysis_id,
                        line.total_time_ms,
                        line.user_move_uci,
                        line.opponent_move_uci or "—",
                    )
                    board = chess.Board(fen)
                    timing_cfg = self.maia3_timing_config
                    if timing_cfg is None or timing_cfg.enabled:
                        from src.engines.maia3_timing import MaiaTimingConfig

                        timing_advice = advise_maia_think_time(
                            board,
                            line.top_moves,
                            target_elo=player_elo,
                            config=timing_cfg or MaiaTimingConfig(),
                        )
                    else:
                        from src.engines.maia3_timing import MaiaTimingAdvice

                        timing_advice = MaiaTimingAdvice(
                            ms=0,
                            criticality_score=0,
                            note="Timing desactivado",
                        )
                    if on_thinking is not None and timing_advice.ms > 1500:
                        on_thinking(timing_advice.ms)
                    base.best_move_san = line.user_move_san
                    base.best_move_uci = line.user_move_uci
                    base.opponent_move_uci = line.opponent_move_uci
                    base.opponent_move_san = line.opponent_move_san
                    base.eval_bar_fraction = line.current_fraction
                    base.eval_bar_post_fraction = line.after_fraction
                    base.evaluation = line.label
                    base.engine_ms = line.total_time_ms
                    base.planned_ms = timing_advice.ms
                    base.phase = timing_advice.phase
                    base.criticality = str(timing_advice.criticality_score)
                    base.engine_mode = OVERLAY_ENGINE_MAIA3
                    base.target_elo = elo
                    label = self.maia3_model_label or "Maia-3"
                    secs = timing_advice.ms // 1000 if timing_advice.ms else 0
                    reply = ""
                    if line.opponent_move_san:
                        reply = f" | resp. {line.opponent_move_san}"
                    base.think_note = (
                        f"{label} {secs}s | criticidad {timing_advice.criticality_score}/10"
                        f"{reply} | {timing_advice.note}"
                    )
                elif use_reckless:
                    log.info("analysis[%s] engine:reckless-human:start elo=%s", analysis_id, elo)
                    human_result = self.human_engine_reckless.analyze(
                        fen,
                        target_elo=elo,
                        on_thinking=on_thinking,
                        is_cancelled=_cancelled,
                    )
                    log.info(
                        "analysis[%s] engine:reckless-human:done %.0f ms move=%s",
                        analysis_id,
                        human_result.engine_ms,
                        human_result.best_move_uci,
                    )
                    base.best_move_san = human_result.best_move_san
                    base.best_move_uci = human_result.best_move_uci
                    base.evaluation = human_result.evaluation
                    base.engine_ms = human_result.engine_ms
                    base.phase = human_result.phase
                    base.criticality = human_result.criticality
                    base.planned_ms = human_result.planned_ms
                    base.think_note = human_result.think_note
                    base.from_book = human_result.from_book
                    base.engine_mode = OVERLAY_ENGINE_RECKLESS
                    if human_result.game_over:
                        base.error = human_result.evaluation
                else:
                    if engine_mode == OVERLAY_ENGINE_RECKLESS:
                        log.warning(
                            "analysis[%s] Reckless no disponible; usando Stockfish humanizado",
                            analysis_id,
                        )
                    log.info("analysis[%s] engine:human:start elo=%s", analysis_id, elo)
                    human_result = self.human_engine.analyze(
                        fen,
                        target_elo=elo,
                        on_thinking=on_thinking,
                        is_cancelled=_cancelled,
                    )
                    log.info(
                        "analysis[%s] engine:human:done %.0f ms move=%s",
                        analysis_id,
                        human_result.engine_ms,
                        human_result.best_move_uci,
                    )
                    base.best_move_san = human_result.best_move_san
                    base.best_move_uci = human_result.best_move_uci
                    base.evaluation = human_result.evaluation
                    base.engine_ms = human_result.engine_ms
                    base.phase = human_result.phase
                    base.criticality = human_result.criticality
                    base.planned_ms = human_result.planned_ms
                    base.think_note = human_result.think_note
                    base.from_book = human_result.from_book
                    base.engine_mode = OVERLAY_ENGINE_STOCKFISH
                    if human_result.game_over:
                        base.error = human_result.evaluation
            else:
                use_reckless_max = (
                    engine_mode == OVERLAY_ENGINE_RECKLESS
                    and self.reckless is not None
                )
                if use_reckless_max:
                    log.info("analysis[%s] engine:reckless-max:start", analysis_id)
                    analysis = self.reckless.analyze(
                        fen=fen,
                        movetime_ms=self.config.movetime_ms,
                        multipv=self.config.multipv,
                        is_cancelled=_cancelled,
                    )
                    log.info(
                        "analysis[%s] engine:reckless-max:done %.0f ms move=%s",
                        analysis_id,
                        analysis.time_ms,
                        analysis.best_move.uci,
                    )
                    base.engine_mode = OVERLAY_ENGINE_RECKLESS
                else:
                    if engine_mode == OVERLAY_ENGINE_RECKLESS:
                        log.warning(
                            "analysis[%s] Reckless no disponible; usando Stockfish max",
                            analysis_id,
                        )
                    log.info("analysis[%s] engine:max:start", analysis_id)
                    self.stockfish.configure_strength(
                        elo if elo else 2000,
                        limit_strength=False,
                    )
                    analysis = self.stockfish.analyze(
                        fen=fen,
                        movetime_ms=self.config.movetime_ms,
                        multipv=self.config.multipv,
                        is_cancelled=_cancelled,
                    )
                    log.info(
                        "analysis[%s] engine:max:done %.0f ms move=%s",
                        analysis_id,
                        analysis.time_ms,
                        analysis.best_move.uci,
                    )
                    base.engine_mode = OVERLAY_ENGINE_STOCKFISH
                if analysis.game_over:
                    base.error = analysis.game_over_reason or "Fin de partida"
                base.best_move_san = analysis.best_move.san
                base.best_move_uci = analysis.best_move.uci
                base.evaluation = analysis.best_move.score_str
                base.engine_ms = analysis.time_ms

            base.total_ms = (time.perf_counter() - t0) * 1000
            log.info("analysis[%s] preview:start", analysis_id)
            base.annotated_image = compose_preview(
                image_bgr, base, show_predictions=show_predictions
            )
            log.info("analysis[%s] done %.0f ms", analysis_id, base.total_ms)
            return base
        except Exception as exc:
            log.exception("Analysis failed (id=%s)", analysis_id)
            return AnalysisOutput(
                fen="",
                best_move_san="",
                best_move_uci="",
                evaluation="",
                vision_ms=0.0,
                engine_ms=0.0,
                total_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )
