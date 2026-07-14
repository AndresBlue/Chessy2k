"""Main Qt window: Chess.com-style overlay client for Chessy."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from PySide6.QtCore import QByteArray, QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.app.analysis_worker import AnalysisJob, AnalysisWorker
from src.app.board_watcher import BoardStableDetector
from src.app.capture_service import CaptureResult, CaptureService
from src.app.elo_power import (
    SLIDER_STEPS,
    elo_to_slider,
    format_elo_label,
    is_max_effort,
    slider_to_elo,
)
from src.app.fast_analyzer import (
    CRITICALITY_LABELS,
    PHASE_LABELS,
    AnalysisOutput,
    FastAnalyzer,
    FastEngineConfig,
    compose_preview,
)
from src.app.config import load_config
from src.app.logging_config import get_logger, setup_logging
from src.app.overlay_theme import (
    MAIA3_MODEL_23M,
    MAIA3_MODEL_CHOICES,
    MAIA3_MODEL_LABELS,
    OVERLAY_ENGINE_LABELS,
    OVERLAY_ENGINE_MAIA3,
    OVERLAY_ENGINE_RECKLESS,
    OVERLAY_ENGINE_STOCKFISH,
    load_maia3_model,
    load_overlay_engine,
    normalize_maia3_model,
    save_maia3_model,
    save_overlay_engine,
)
from src.app.region_store import load_region, save_region
from src.app.runtime_paths import resolve_runtime_paths
from src.app.screen_capture import ScreenRegion
from src.app.ui_qt.image_utils import eval_to_fraction
from src.app.ui_qt.region_selector import RegionSelectorOverlay
from src.app.ui_qt.theme import (
    THEMES,
    ChessPalette,
    FontConfig,
    build_qss,
    load_fonts,
    load_show_predictions,
    load_target_elo,
    load_theme_preference,
    load_window_geometry,
    make_app_font,
    save_show_predictions,
    save_target_elo,
    save_theme_preference,
    save_window_geometry,
)
from src.app.ui_qt.widgets.board_preview import BoardPreview
from src.app.ui_qt.widgets.cards import Card
from src.app.ui_qt.widgets.eval_bar import EvalBar
from src.engines.humanization_config import humanization_from_config

POLL_MS_IDLE = 140
PUMP_MS = 16
UI_PUMP_BATCH = 8
MAX_TRACKER_ERRORS = 3

log = get_logger(__name__)


def _format_think_time(ms: int) -> str:
    if ms <= 0:
        return "\u2014"
    seconds = max(1, int(round(ms / 1000)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rem = divmod(seconds, 60)
    return f"{minutes}m {rem:02d}s"


class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = resolve_runtime_paths()
        setup_logging(log_dir=self.paths["root"] / "data" / "logs")

        # -- runtime state ------------------------------------------------
        self._ui_result_queue: queue.Queue = queue.Queue()
        self._ui_task_queue: queue.Queue = queue.Queue()
        self._analyzer: FastAnalyzer | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._capture_service = CaptureService(result_queue=self._ui_result_queue)
        self._watcher = BoardStableDetector()
        self._analyzer_ready = threading.Event()
        self._analyzing = False
        self._analysis_id = 0
        self._region_selecting = False
        self._region_overlay: RegionSelectorOverlay | None = None
        self._poll_busy = False
        self._last_capture = None
        self._last_analysis: AnalysisOutput | None = None
        self._last_analyzed_placement: str | None = None
        self._consecutive_tracker_errors = 0
        self._closing = False

        # -- preferences --------------------------------------------------
        root = self.paths["root"]
        self._theme_name = load_theme_preference(root)
        self._show_predictions = load_show_predictions(root)
        self._target_elo = load_target_elo(root)
        self.side = "white"
        _cfg = load_config()
        self._overlay_engine = load_overlay_engine(root, OVERLAY_ENGINE_STOCKFISH)
        _maia_default = normalize_maia3_model(
            _cfg.get("engines", {}).get("maia3", {}).get("model", MAIA3_MODEL_23M)
        )
        self._maia3_model = load_maia3_model(root, _maia_default)
        self.region: ScreenRegion | None = load_region(root)

        self._fonts: FontConfig = load_fonts(root)
        self.setFont(make_app_font(self._fonts, 10))

        self.setWindowTitle("Chessy")
        self.setMinimumSize(900, 620)

        self._build_ui()
        self._apply_theme()
        self._update_region_label()

        saved_geo = load_window_geometry(root)
        if saved_geo:
            try:
                self.restoreGeometry(QByteArray.fromHex(saved_geo.encode("ascii")))
            except Exception:
                self.resize(1060, 720)
        else:
            self.resize(1060, 720)

        self._pump_timer = QTimer(self)
        self._pump_timer.setInterval(PUMP_MS)
        self._pump_timer.timeout.connect(self._pump)
        self._pump_timer.start()

        self._init_analyzer_async()
        QTimer.singleShot(POLL_MS_IDLE, self._poll_auto)

    # -- palette ---------------------------------------------------------
    @property
    def palette_chess(self) -> ChessPalette:
        return THEMES[self._theme_name]

    # -- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 16, 18, 12)
        outer.setSpacing(14)

        outer.addLayout(self._build_header())
        outer.addLayout(self._build_body(), 1)
        outer.addWidget(self._build_status_bar())

    def _build_header(self) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(8)

        top = QHBoxLayout()
        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        title = QLabel("Chessy")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Asistente de tablero en tiempo real")
        subtitle.setObjectName("AppSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        top.addLayout(title_block)
        top.addStretch(1)

        self.theme_btn = QPushButton("Tema claro")
        self.theme_btn.setObjectName("GhostButton")
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self._toggle_theme)
        top.addWidget(self.theme_btn, 0, Qt.AlignmentFlag.AlignTop)
        header.addLayout(top)

        chip_row = QHBoxLayout()
        self.region_chip = QLabel("Region: no definida")
        self.region_chip.setObjectName("RegionChip")
        chip_row.addWidget(self.region_chip, 0, Qt.AlignmentFlag.AlignLeft)
        chip_row.addStretch(1)
        header.addLayout(chip_row)
        return header

    def _build_body(self) -> QHBoxLayout:
        body = QHBoxLayout()
        body.setSpacing(14)

        # Board column: eval bar + preview card.
        # The board card stays flat (no shadow effect) so the 60fps preview
        # animation does not force re-rasterizing a heavy graphics effect.
        self.board_card = Card()
        self.board_card.body().setContentsMargins(12, 12, 12, 12)
        board_row = QHBoxLayout()
        board_row.setSpacing(10)
        self.eval_bar = EvalBar(self.palette_chess)
        self.preview = BoardPreview(self.palette_chess)
        board_row.addWidget(self.eval_bar)
        board_row.addWidget(self.preview, 1)
        self.board_card.add_layout(board_row)
        body.addWidget(self.board_card, 3)

        # Sidebar in a scroll area.
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(14)
        self.controls_card = self._build_controls_card()
        self.rec_card = self._build_recommendation_card()
        self.help_card = self._build_help_card()
        # Recommendation first: the best move / eval / think time is the primary
        # output and should stay visible without scrolling.
        side_layout.addWidget(self.rec_card)
        side_layout.addWidget(self.controls_card)
        side_layout.addWidget(self.help_card)
        side_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(sidebar)
        body.addWidget(scroll, 2)
        return body

    def _build_controls_card(self) -> Card:
        card = Card("Controles")
        self.btn_region = QPushButton("Definir region")
        self.btn_region.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_region.clicked.connect(self._pick_region)
        card.add(self.btn_region)

        self.btn_capture = QPushButton("Capturar y analizar")
        self.btn_capture.setObjectName("PrimaryButton")
        self.btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_capture.clicked.connect(self._capture_manual)
        card.add(self.btn_capture)

        self.auto_check = QCheckBox("Auto tras jugada (espera animacion)")
        self.auto_check.toggled.connect(self._on_auto_toggle)
        card.add(self.auto_check)

        self.predictions_check = QCheckBox("Mostrar predicciones del modelo")
        self.predictions_check.setChecked(self._show_predictions)
        self.predictions_check.toggled.connect(self._on_predictions_toggle)
        card.add(self.predictions_check)

        elo_box = Card(inner=True)
        elo_title = QLabel("POTENCIA (ELO)")
        elo_title.setObjectName("CardTitle")
        elo_box.add(elo_title)
        self.elo_value_label = QLabel(format_elo_label(self._target_elo))
        self.elo_value_label.setObjectName("MetaValue")
        elo_box.add(self.elo_value_label)
        self.elo_slider = QSlider(Qt.Orientation.Horizontal)
        self.elo_slider.setRange(0, SLIDER_STEPS)
        self.elo_slider.setValue(int(round(elo_to_slider(self._target_elo))))
        self.elo_slider.valueChanged.connect(self._on_elo_slider)
        elo_box.add(self.elo_slider)
        hints = QHBoxLayout()
        lo = QLabel("Minimo")
        lo.setObjectName("CardTitle")
        hi = QLabel("Maximo esfuerzo")
        hi.setObjectName("CardTitle")
        hints.addWidget(lo)
        hints.addStretch(1)
        hints.addWidget(hi)
        elo_box.add_layout(hints)
        card.add(elo_box)

        engine_box = Card(inner=True)
        engine_title = QLabel("MOTOR DE JUGADAS")
        engine_title.setObjectName("CardTitle")
        engine_box.add(engine_title)
        engine_row = QHBoxLayout()
        self.radio_engine_stockfish = QRadioButton(
            OVERLAY_ENGINE_LABELS[OVERLAY_ENGINE_STOCKFISH]
        )
        self.radio_engine_reckless = QRadioButton(
            OVERLAY_ENGINE_LABELS[OVERLAY_ENGINE_RECKLESS]
        )
        self.radio_engine_maia3 = QRadioButton(OVERLAY_ENGINE_LABELS[OVERLAY_ENGINE_MAIA3])
        self.radio_engine_stockfish.setChecked(
            self._overlay_engine == OVERLAY_ENGINE_STOCKFISH
        )
        self.radio_engine_reckless.setChecked(
            self._overlay_engine == OVERLAY_ENGINE_RECKLESS
        )
        self.radio_engine_maia3.setChecked(
            self._overlay_engine == OVERLAY_ENGINE_MAIA3
        )
        self._engine_group = QButtonGroup(self)
        self._engine_group.addButton(self.radio_engine_stockfish)
        self._engine_group.addButton(self.radio_engine_reckless)
        self._engine_group.addButton(self.radio_engine_maia3)
        self.radio_engine_stockfish.toggled.connect(self._on_engine_changed)
        self.radio_engine_reckless.toggled.connect(self._on_engine_changed)
        self.radio_engine_maia3.toggled.connect(self._on_engine_changed)
        engine_row.addWidget(self.radio_engine_stockfish)
        engine_row.addWidget(self.radio_engine_reckless)
        engine_row.addWidget(self.radio_engine_maia3)
        self.maia_model_combo = QComboBox()
        for model_id in MAIA3_MODEL_CHOICES:
            self.maia_model_combo.addItem(MAIA3_MODEL_LABELS[model_id], model_id)
        idx = (
            max(0, list(MAIA3_MODEL_CHOICES).index(self._maia3_model))
            if self._maia3_model in MAIA3_MODEL_CHOICES
            else 1
        )
        self.maia_model_combo.setCurrentIndex(idx)
        self.maia_model_combo.setEnabled(self._overlay_engine == OVERLAY_ENGINE_MAIA3)
        self.maia_model_combo.currentIndexChanged.connect(self._on_maia_model_changed)
        engine_row.addWidget(self.maia_model_combo)
        engine_row.addStretch(1)
        engine_box.add_layout(engine_row)
        card.add(engine_box)

        side_box = Card(inner=True)
        side_title = QLabel("YO JUEGO CON")
        side_title.setObjectName("CardTitle")
        side_box.add(side_title)
        side_row = QHBoxLayout()
        self.radio_white = QRadioButton("Blancas")
        self.radio_black = QRadioButton("Negras")
        self.radio_white.setChecked(True)
        self._side_group = QButtonGroup(self)
        self._side_group.addButton(self.radio_white)
        self._side_group.addButton(self.radio_black)
        self.radio_white.toggled.connect(self._on_side_changed)
        side_row.addWidget(self.radio_white)
        side_row.addWidget(self.radio_black)
        side_row.addStretch(1)
        side_box.add_layout(side_row)
        card.add(side_box)
        return card

    def _build_recommendation_card(self) -> Card:
        card = Card("Recomendacion")

        card.add(self._mini_title("Mejor jugada"))
        self.move_value = QLabel("\u2014")
        self.move_value.setObjectName("MoveValue")
        card.add(self.move_value)

        card.add(self._mini_title("Pensar sugerido"))
        self.think_value = QLabel("\u2014")
        self.think_value.setObjectName("ThinkValue")
        card.add(self.think_value)
        self.think_note = QLabel("Tiempo sugerido")
        self.think_note.setObjectName("ThinkNote")
        card.add(self.think_note)

        card.add(self._mini_title("Evaluacion"))
        self.eval_value = QLabel("\u2014")
        self.eval_value.setObjectName("EvalValue")
        card.add(self.eval_value)

        card.add(self._mini_title("Estilo humano"))
        self.meta_value = QLabel("\u2014")
        self.meta_value.setObjectName("MetaValue")
        self.meta_value.setWordWrap(True)
        card.add(self.meta_value)

        card.add(self._mini_title("FEN"))
        self.fen_value = QLabel("\u2014")
        self.fen_value.setObjectName("FenValue")
        self.fen_value.setWordWrap(True)
        card.add(self.fen_value)

        card.add(self._mini_title("Tiempos"))
        self.times_value = QLabel("\u2014")
        self.times_value.setObjectName("TimesValue")
        card.add(self.times_value)

        # Reveal animation target (no shadow on this label, so opacity is safe).
        self._reveal_effect = QGraphicsOpacityEffect(self.move_value)
        self._reveal_effect.setOpacity(1.0)
        self.move_value.setGraphicsEffect(self._reveal_effect)
        self._reveal_anim = QPropertyAnimation(self._reveal_effect, b"opacity", self)
        self._reveal_anim.setDuration(240)
        self._reveal_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        return card

    def _build_help_card(self) -> Card:
        card = Card("Flujo rapido")
        text = QLabel(
            "1. Open Chess.com (or any digital board)\n"
            "2. Define the board screen region\n"
            "3. Select White or Black\n"
            "4. Capture or enable Auto\n"
            "Chessy always suggests for the selected side\n"
            "Engines: Stockfish, Reckless, Maia-3\n"
            "Strength: human Elo up to max effort"
        )
        text.setObjectName("MetaValue")
        text.setWordWrap(True)
        card.add(text)
        return card

    def _build_status_bar(self) -> QWidget:
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(2, 0, 2, 0)
        self.status_label = QLabel("Inicializando...")
        self.status_label.setObjectName("StatusLabel")
        row.addWidget(self.status_label)
        row.addStretch(1)
        return wrap

    @staticmethod
    def _mini_title(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("CardTitle")
        return label

    # -- theming ---------------------------------------------------------
    def _apply_theme(self) -> None:
        palette = self.palette_chess
        self.setStyleSheet(build_qss(palette, self._fonts))
        self.preview.set_palette(palette)
        self.eval_bar.set_palette(palette)
        # Soft elevation on the static sidebar cards only. The board card is
        # left flat because its preview repaints continuously (timer overlay).
        for card in (self.controls_card, self.rec_card, self.help_card):
            card.elevate(palette)
        self.theme_btn.setText("Tema claro" if palette.name == "dark" else "Tema oscuro")

    def _toggle_theme(self) -> None:
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        save_theme_preference(self.paths["root"], self._theme_name)
        self._apply_theme()

    # -- threading helpers ----------------------------------------------
    def _schedule_ui(self, fn: Callable[[], None]) -> None:
        self._ui_task_queue.put(fn)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _trace(self, message: str, *, status: bool = True) -> None:
        log.info("ui %s", message)
        if status:
            self._set_status(message)

    def _safe(self, fn):
        def _wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                log.exception("Overlay callback failed")
                self._set_status("Error interno. Revisa data/logs/.")

        return _wrapped

    def _pump(self) -> None:
        if self._closing:
            return
        processed = 0
        try:
            while processed < UI_PUMP_BATCH:
                callback, result = self._ui_result_queue.get_nowait()
                try:
                    callback(result)
                except Exception:
                    log.exception("UI callback error")
                processed += 1
        except queue.Empty:
            pass
        processed = 0
        try:
            while processed < UI_PUMP_BATCH:
                fn = self._ui_task_queue.get_nowait()
                try:
                    fn()
                except Exception:
                    log.exception("UI task error")
                processed += 1
        except queue.Empty:
            pass

    # -- analyzer init ---------------------------------------------------
    def _init_analyzer_async(self) -> None:
        sf = self.paths.get("stockfish")
        ckpt = self.paths.get("classifier")
        if sf is None or ckpt is None:
            hints: list[str] = []
            if sf is None:
                hints.append(
                    "Stockfish: place binary under engines/stockfish/ "
                    "or set CHESSY_STOCKFISH_PATH"
                )
            if ckpt is None:
                hints.append("Vision: place data/checkpoints/vision/best.pt")
            self._set_status("Error: " + " | ".join(hints))
            return

        self._set_status("Loading model and engines...")

        def _load() -> None:
            try:
                from src.app.config import load_config

                cfg = load_config()
                human_cfg = humanization_from_config(cfg)
                reckless = self.paths.get("reckless")
                analyzer = FastAnalyzer(
                    stockfish_path=str(sf),
                    classifier_path=str(ckpt),
                    config=FastEngineConfig(
                        movetime_ms=250,
                        multipv=1,
                        threads=4,
                        hash_mb=256,
                        humanization=human_cfg,
                    ),
                    project_root=self.paths["root"],
                    maia_model=self._maia3_model,
                    reckless_path=str(reckless) if reckless else None,
                )

                def _on_ready() -> None:
                    self._analyzer = analyzer
                    self._analysis_worker = AnalysisWorker(analyzer)
                    self._analyzer_ready.set()
                    if not analyzer.has_reckless_engine:
                        if self._overlay_engine == OVERLAY_ENGINE_RECKLESS:
                            self._overlay_engine = OVERLAY_ENGINE_STOCKFISH
                            save_overlay_engine(
                                self.paths["root"], OVERLAY_ENGINE_STOCKFISH
                            )
                        self.radio_engine_reckless.setEnabled(False)
                        if self.radio_engine_reckless.isChecked():
                            self.radio_engine_stockfish.setChecked(True)
                    if not analyzer.has_maia3_engine:
                        if self._overlay_engine == OVERLAY_ENGINE_MAIA3:
                            self._overlay_engine = OVERLAY_ENGINE_STOCKFISH
                            save_overlay_engine(
                                self.paths["root"], OVERLAY_ENGINE_STOCKFISH
                            )
                        self.radio_engine_maia3.setEnabled(False)
                        self.maia_model_combo.setEnabled(False)
                        if self.radio_engine_maia3.isChecked():
                            self.radio_engine_stockfish.setChecked(True)
                    else:
                        self.maia_model_combo.setEnabled(
                            self.radio_engine_maia3.isChecked()
                        )
                        current = analyzer.maia3_model_info or self._maia3_model
                        self._sync_maia_combo(current)
                    status = f"Ready ({analyzer.vision_device})."
                    motors: list[str] = ["Stockfish"]
                    if analyzer.has_reckless_engine:
                        motors.append("Reckless")
                    if analyzer.has_maia3_engine:
                        motors.append(f"Maia-3 ({analyzer.maia3_model_info or '23M'})")
                    status += " Engines: " + ", ".join(motors) + "."
                    if not analyzer.has_maia3_engine:
                        status += " Maia-3 unavailable (python scripts/setup_maia3.py)."
                    if not analyzer.has_reckless_engine:
                        status += " Reckless binary not found under engines/reckless/."
                    status += " Define a region or capture the board."
                    self._set_status(status)

                self._schedule_ui(_on_ready)
            except Exception as exc:
                log.exception("Analyzer init failed")
                self._schedule_ui(lambda: self._set_status(f"Init error: {exc}"))

        threading.Thread(target=_load, name="chessy-init", daemon=True).start()

    # -- preferences handlers -------------------------------------------
    def _current_elo(self) -> int:
        return slider_to_elo(float(self.elo_slider.value()))

    def _human_mode_enabled(self) -> bool:
        return not is_max_effort(self._current_elo())

    def _on_elo_slider(self, _value: int) -> None:
        elo = self._current_elo()
        self._target_elo = elo
        self.elo_value_label.setText(format_elo_label(elo))
        save_target_elo(self.paths["root"], elo)

    def _current_engine_mode(self) -> str:
        if self.radio_engine_reckless.isChecked():
            return OVERLAY_ENGINE_RECKLESS
        if self.radio_engine_maia3.isChecked():
            return OVERLAY_ENGINE_MAIA3
        return OVERLAY_ENGINE_STOCKFISH

    def _sync_maia_combo(self, model: str) -> None:
        model = normalize_maia3_model(model)
        self._maia3_model = model
        for i in range(self.maia_model_combo.count()):
            if self.maia_model_combo.itemData(i) == model:
                blocked = self.maia_model_combo.blockSignals(True)
                self.maia_model_combo.setCurrentIndex(i)
                self.maia_model_combo.blockSignals(blocked)
                break

    def _on_engine_changed(self, _checked: bool = False) -> None:
        sender = self.sender()
        if isinstance(sender, QRadioButton) and not sender.isChecked():
            return
        self._overlay_engine = self._current_engine_mode()
        save_overlay_engine(self.paths["root"], self._overlay_engine)
        maia_ok = (
            self._analyzer is not None and self._analyzer.has_maia3_engine
        )
        self.maia_model_combo.setEnabled(
            self._overlay_engine == OVERLAY_ENGINE_MAIA3 and maia_ok
        )

    def _on_maia_model_changed(self, index: int) -> None:
        if index < 0:
            return
        model = normalize_maia3_model(self.maia_model_combo.itemData(index))
        if model == self._maia3_model and (
            self._analyzer is None or self._analyzer.maia3_model_info == model
        ):
            return
        self._maia3_model = model
        save_maia3_model(self.paths["root"], model)
        analyzer = self._analyzer
        if analyzer is None or not analyzer.has_maia3_engine:
            return
        label = MAIA3_MODEL_LABELS.get(model, model)
        self._set_status(f"Cargando Maia-3 {label}...")

        def _switch() -> None:
            try:
                applied = analyzer.set_maia3_model(model)

                def _done() -> None:
                    self._sync_maia_combo(applied)
                    self._set_status(
                        f"Maia-3 listo: {MAIA3_MODEL_LABELS.get(applied, applied)}"
                    )

                self._schedule_ui(_done)
            except Exception as exc:
                log.exception("Failed to switch Maia-3 model")
                self._schedule_ui(
                    lambda: self._set_status(f"Error al cargar Maia-3 ({exc})")
                )

        threading.Thread(target=_switch, daemon=True).start()

    def _on_predictions_toggle(self, checked: bool) -> None:
        self._show_predictions = bool(checked)
        save_show_predictions(self.paths["root"], self._show_predictions)
        self._refresh_preview_from_last()

    def _on_side_changed(self, _checked: bool = False) -> None:
        side = "white" if self.radio_white.isChecked() else "black"
        if side == self.side:
            return
        self.side = side
        self._cancel_analysis()
        self._last_analyzed_placement = None
        if self._analyzer is not None:
            self._analyzer.reset_vision_cache()

    def _refresh_preview_from_last(self) -> None:
        if self._last_capture is None or self._last_analysis is None:
            return
        preview = compose_preview(
            self._last_capture,
            self._last_analysis,
            show_predictions=self._show_predictions,
        )
        self.preview.set_image_bgr(preview)

    # -- region ----------------------------------------------------------
    def _update_region_label(self) -> None:
        if self.region and self.region.is_valid():
            r = self.region
            self.region_chip.setText(
                f"Region activa: ({r.left}, {r.top})   |   {r.width} x {r.height} px"
            )
        else:
            self.region_chip.setText("Region: no definida")

    def _pick_region(self) -> None:
        if self._region_selecting:
            return
        self._region_selecting = True
        self._set_status("Selecciona region: arrastra sobre el tablero (ESC cancela).")
        overlay = RegionSelectorOverlay(self.palette_chess)
        overlay.finished.connect(self._on_region_selected)
        self._region_overlay = overlay
        self.hide()
        QTimer.singleShot(120, overlay.show_selector)

    def _on_region_selected(self, region: ScreenRegion | None) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()
        try:
            if region is not None and region.is_valid():
                self.region = region
                save_region(self.paths["root"], region)
                self._watcher.reset()
                self._last_analyzed_placement = None
                self._cancel_analysis()
                if self._analyzer is not None:
                    self._analyzer.reset_vision_cache()
                self._update_region_label()
                self._set_status("Region guardada. Listo para capturar.")
            else:
                self._set_status("Seleccion de region cancelada.")
        finally:
            self._region_selecting = False
            self._region_overlay = None

    def _capture_region_async(self, region: ScreenRegion, callback) -> None:
        self._capture_service.capture_region_async(region, callback)

    # -- capture / analysis ---------------------------------------------
    def _capture_manual(self) -> None:
        if not self.region or not self.region.is_valid():
            self._set_status("Define primero la region del tablero.")
            return
        if not self._analyzer_ready.is_set():
            self._set_status("Espera a que termine la inicializacion...")
            return
        self._trace("Manual: capturando region...")

        def _on_capture(result: CaptureResult) -> None:
            if result.error or result.image is None:
                self._trace(f"Captura fallida: {result.error or 'desconocido'}")
                return
            self._trace("Manual: captura OK; iniciando analisis...")
            self._cancel_analysis()
            if self._analyzer is not None:
                self._analyzer.reset_vision_cache()
            self._consecutive_tracker_errors = 0
            self._last_analyzed_placement = None
            self._start_analysis(result.image, from_auto=False)

        self._capture_region_async(self.region, self._safe(_on_capture))

    def _on_auto_toggle(self, checked: bool) -> None:
        if not checked:
            self._set_status("Auto desactivado.")
            return
        if not self.region or not self.region.is_valid():
            self.auto_check.setChecked(False)
            self._set_status("Define primero la region del tablero.")
            return
        if not self._analyzer_ready.is_set():
            self.auto_check.setChecked(False)
            self._set_status("Espera a que termine la inicializacion.")
            return

        def _on_capture(result: CaptureResult) -> None:
            if result.error or result.image is None:
                self.auto_check.setChecked(False)
                self._set_status(f"No se pudo activar auto: {result.error}")
                return
            self._cancel_analysis()
            self._watcher.seed(result.image)
            side_label = "blancas" if self.side == "white" else "negras"
            self._set_status(f"Auto activo ({side_label}). Analizare cada tablero estable.")

        self._capture_region_async(self.region, self._safe(_on_capture))

    def _cancel_analysis(self) -> None:
        self.preview.stop_think()
        if self._analysis_worker is not None:
            self._analysis_id = self._analysis_worker.cancel()
        else:
            self._analysis_id += 1
        self._analyzing = False

    def _schedule_auto_poll(self, interval: int = POLL_MS_IDLE) -> None:
        if self._closing or not self.auto_check.isChecked():
            return
        QTimer.singleShot(interval, self._poll_auto)

    def _poll_auto(self) -> None:
        if self._closing:
            return
        if (
            self.auto_check.isChecked()
            and self.region
            and not self._region_selecting
            and not self._poll_busy
            and self._analyzer_ready.is_set()
        ):
            self._poll_busy = True
            region = self.region

            def _on_capture(result: CaptureResult) -> None:
                self._poll_busy = False
                interval = POLL_MS_IDLE
                if self._closing:
                    return
                if result.error or result.image is None:
                    self._set_status(f"Auto: {result.error}")
                else:
                    try:
                        image = result.image
                        poll_result = self._watcher.poll(image, time.perf_counter())
                        interval = self._watcher.poll_interval_ms
                        if poll_result.kind == "motion":
                            if self._analyzer is not None:
                                self._analyzer.invalidate_vision_cache()
                            if self._analyzing:
                                self._cancel_analysis()
                                self._set_status(
                                    "Movimiento detectado. Cancelando analisis obsoleto..."
                                )
                            else:
                                self._set_status(
                                    "Movimiento detectado, esperando fin de animacion..."
                                )
                        elif poll_result.kind == "ready":
                            assert poll_result.image is not None
                            if self._analyzing:
                                self._set_status(
                                    "Auto: analisis en curso; esperando resultado..."
                                )
                            else:
                                self._start_analysis(poll_result.image, from_auto=True)
                                return
                    except Exception as exc:
                        log.exception("Auto poll error")
                        self._set_status(f"Auto: {exc}")
                if not self._closing:
                    QTimer.singleShot(interval, self._poll_auto)

            self._capture_region_async(region, self._safe(_on_capture))
            return

        QTimer.singleShot(POLL_MS_IDLE, self._poll_auto)

    def _start_analysis(self, image, *, from_auto: bool = False, precomputed_vision=None) -> None:
        if self._analysis_worker is None:
            return
        self.preview.stop_think()
        if from_auto and precomputed_vision is not None:
            placement = getattr(precomputed_vision, "fen_pieces", None)
            if placement and placement == self._last_analyzed_placement:
                self._trace("Analisis omitido: posicion sin cambios.", status=False)
                if self.auto_check.isChecked():
                    self._schedule_auto_poll()
                return
        self._analysis_id += 1
        aid = self._analysis_id
        self._analyzing = True
        self._last_capture = image.copy()
        prefix = "Auto: " if from_auto else ""
        self._trace(f"{prefix}Analizando posicion...")
        self.preview.set_image_bgr(image)
        side = self.side
        human = self._human_mode_enabled()
        elo = self._current_elo()
        engine_mode = self._current_engine_mode()

        def on_thinking(planned_ms: int) -> None:
            if aid != self._analysis_id:
                return
            secs = max(1, planned_ms // 1000)
            self._schedule_ui(lambda: self._set_status(f"{prefix}Pensando... (~{secs}s)"))

        def on_complete(result: AnalysisOutput) -> None:
            self._schedule_ui(lambda: self._show_result(result, aid))

        self._analysis_worker.submit(
            AnalysisJob(
                analysis_id=aid,
                image=image,
                side=side,
                show_predictions=self._show_predictions,
                human_mode=human,
                target_elo=elo,
                on_thinking=on_thinking if human else None,
                on_complete=on_complete,
                precomputed_vision=precomputed_vision,
                engine_mode=engine_mode,
            )
        )

    def _flash_reveal(self) -> None:
        # Start partially visible so the move text is never fully invisible.
        self._reveal_anim.stop()
        self._reveal_effect.setOpacity(0.4)
        self._reveal_anim.setStartValue(0.4)
        self._reveal_anim.setEndValue(1.0)
        self._reveal_anim.start()

    def _show_result(self, result: AnalysisOutput, aid: int) -> None:
        if aid != self._analysis_id:
            return
        self._analyzing = False
        self._last_analysis = result

        if result.annotated_image is not None:
            self.preview.set_image_bgr(result.annotated_image)
        elif self._last_capture is not None:
            self.preview.set_image_bgr(self._last_capture)

        if result.error:
            self.preview.stop_think()
            if result.tracker_status == "vision_error":
                self._consecutive_tracker_errors += 1
                if self._consecutive_tracker_errors >= MAX_TRACKER_ERRORS:
                    log.warning(
                        "ui auto-reset tracker after %d consecutive vision_errors",
                        self._consecutive_tracker_errors,
                    )
                    if self._analyzer is not None:
                        self._analyzer.reset_vision_cache()
                    self._consecutive_tracker_errors = 0
                    self._last_analyzed_placement = None
            else:
                self._consecutive_tracker_errors = 0

            self._set_status(f"Error: {result.error}")
            self.move_value.setText("\u2014")
            self.eval_value.setText("\u2014")
            self.eval_bar.reset()
            self.meta_value.setText("\u2014")
            self.think_value.setText("\u2014")
            self.think_note.setText("Sin sugerencia")
            if result.fen:
                fen_short = result.fen[:80] + ("..." if len(result.fen) > 80 else "")
                self.fen_value.setText(fen_short)
            self.times_value.setText(
                f"Vis: {result.vision_ms:.0f} ms\n"
                f"FEN: {result.fen_ms:.0f} ms\n"
                f"Val: {result.validation_ms:.0f} ms\n"
                f"Total: {result.total_ms:.0f} ms"
            )
            self._watcher.discard_pending()
            if self.auto_check.isChecked():
                self._schedule_auto_poll()
            return

        self._consecutive_tracker_errors = 0
        if result.fen:
            self._last_analyzed_placement = result.fen.split()[0]
        if self._last_capture is not None:
            self._watcher.mark_analyzed(self._last_capture)
            if self.auto_check.isChecked():
                self._set_status("Sugerencia lista. Auto analizara el proximo tablero estable.")
            else:
                self._set_status("Analisis completado.")

        self.move_value.setText(f"{result.best_move_san}  ({result.best_move_uci})")
        self.eval_value.setText(result.evaluation)
        if result.eval_bar_post_fraction is not None and result.eval_bar_fraction is not None:
            self.eval_bar.animate_dual(
                result.eval_bar_fraction,
                result.eval_bar_post_fraction,
            )
        else:
            self.eval_bar.animate_to(eval_to_fraction(result.evaluation))
        self.think_value.setText(_format_think_time(result.planned_ms))
        self.think_note.setText(result.think_note or "Tiempo sugerido")
        self.preview.start_think(result.planned_ms)
        self._flash_reveal()

        if result.humanized:
            if result.engine_mode == OVERLAY_ENGINE_MAIA3:
                crit = result.criticality or "—"
                self.meta_value.setText(
                    f"Maia-3 Elo ~{result.target_elo}  |  "
                    f"Criticality {crit}/10  |  {result.think_note or 'human'}"
                )
            elif result.engine_mode == OVERLAY_ENGINE_RECKLESS:
                phase = PHASE_LABELS.get(result.phase, result.phase or "\u2014")
                crit = CRITICALITY_LABELS.get(
                    result.criticality, result.criticality or "\u2014"
                )
                book = "  |  Book" if result.from_book else ""
                self.meta_value.setText(
                    f"Reckless Elo ~{result.target_elo}  |  {phase}  |  "
                    f"Criticality {crit}{book}"
                )
            else:
                phase = PHASE_LABELS.get(result.phase, result.phase or "\u2014")
                crit = CRITICALITY_LABELS.get(
                    result.criticality, result.criticality or "\u2014"
                )
                book = "  |  Book" if result.from_book else ""
                self.meta_value.setText(
                    f"Elo ~{result.target_elo}  |  {phase}  |  Criticality {crit}{book}"
                )
        else:
            if result.engine_mode == OVERLAY_ENGINE_RECKLESS:
                label = "Max effort (Reckless)"
            else:
                label = (
                    "Max effort (Stockfish)"
                    if is_max_effort(self._current_elo())
                    else "Full strength (Stockfish)"
                )
            self.meta_value.setText(label)

        fen_short = result.fen[:80] + ("..." if len(result.fen) > 80 else "")
        self.fen_value.setText(fen_short)
        mode_labels = {
            "full": "tablero completo",
            "incremental": f"{result.squares_updated} casillas",
            "cached": "sin cambios",
        }
        mode = mode_labels.get(result.inference_mode, result.inference_mode)
        self.times_value.setText(
            f"Vis: {result.vision_ms:.0f} ms ({mode})\n"
            f"FEN: {result.fen_ms:.0f} ms\n"
            f"Val: {result.validation_ms:.0f} ms\n"
            f"Motor: {result.engine_ms:.0f} ms\n"
            f"Total: {result.total_ms:.0f} ms"
        )
        if self.auto_check.isChecked():
            self._schedule_auto_poll()

    # -- lifecycle -------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._closing = True
        try:
            save_window_geometry(
                self.paths["root"], bytes(self.saveGeometry().toHex()).decode("ascii")
            )
        except Exception:
            log.exception("Failed to save window geometry")
        self._pump_timer.stop()
        self.preview.stop_think()
        if self._analysis_worker is not None:
            self._analysis_worker.shutdown()
        self._capture_service.shutdown()
        if self._analyzer is not None:
            self._analyzer.close()
        super().closeEvent(event)
