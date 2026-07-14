"""Lightweight transparent overlay to pick a screen capture region."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from src.app.capture_service import CaptureService
from src.app.logging_config import get_logger
from src.app.region_coords import region_from_drag
from src.app.screen_capture import ScreenRegion, virtual_screen_bounds

log = get_logger(__name__)


def _tk_geometry(width: int, height: int, x: int, y: int) -> str:
    """Build a Tk geometry string with explicit signed offsets."""
    return f"{width}x{height}{x:+d}{y:+d}"


class RegionSelector:
    """Transparent full-desktop picker; drag to select a screen region."""

    def __init__(
        self,
        master: tk.Misc,
        on_complete: Callable[[ScreenRegion | None], None],
    ):
        self._on_complete = on_complete
        self._bounds = virtual_screen_bounds()
        self._start_root_x = 0
        self._start_root_y = 0
        self._rect_id: int | None = None
        self._finished = False

        left, top, right, bottom = self._bounds
        self._screen_x = left
        self._screen_y = top
        width = max(1, right - left)
        height = max(1, bottom - top)

        self.root = tk.Toplevel(master)
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.28)
        except tk.TclError:
            pass
        self.root.geometry(_tk_geometry(width, height, left, top))
        self.root.configure(cursor="crosshair")
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._finish(None))
        self.root.bind("<Escape>", lambda _e: self._finish(None))

        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            bg="#050505",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_text(
            width // 2,
            34,
            text="Arrastra sobre el tablero  |  ESC cancelar",
            fill="white",
            font=("Segoe UI", 14, "bold"),
        )

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def show(self) -> None:
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.grab_set()
        self.root.lift()
        self.root.focus_force()

    def _on_press(self, event: tk.Event) -> None:
        self._start_root_x = int(event.x_root)
        self._start_root_y = int(event.y_root)
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        x, y = self._to_canvas(event.x_root, event.y_root)
        self._rect_id = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#e94560",
            width=3,
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self._rect_id is None:
            return
        start_x, start_y = self._to_canvas(self._start_root_x, self._start_root_y)
        end_x, end_y = self._to_canvas(event.x_root, event.y_root)
        self.canvas.coords(
            self._rect_id,
            start_x,
            start_y,
            end_x,
            end_y,
        )

    def _on_release(self, event: tk.Event) -> None:
        region = region_from_drag(
            self._start_root_x,
            self._start_root_y,
            int(event.x_root),
            int(event.y_root),
        )
        region = region.clamp_to_virtual_screen()
        self._finish(region if region.is_valid() else None)

    def _to_canvas(self, root_x: int, root_y: int) -> tuple[int, int]:
        return int(root_x - self._screen_x), int(root_y - self._screen_y)

    def _finish(self, region: ScreenRegion | None) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self.root.grab_release()
        except tk.TclError:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self._on_complete(region)


def open_region_selector(
    master: tk.Misc,
    capture_service: CaptureService,
    on_complete: Callable[[ScreenRegion | None], None],
    *,
    on_status: Callable[[str], None] | None = None,
    selector_holder: list | None = None,
    schedule_ui: Callable[[Callable[[], None]], None] | None = None,
) -> None:
    """Show a transparent region picker without capturing the desktop first."""
    _ = capture_service  # Kept for API compatibility with OverlayClient.
    saved_geo = master.geometry()
    try:
        was_topmost = bool(master.attributes("-topmost"))
    except tk.TclError:
        was_topmost = True

    def _run_on_ui(fn: Callable[[], None]) -> None:
        if schedule_ui is not None:
            schedule_ui(fn)
        else:
            master.after(0, fn)

    def _status(msg: str) -> None:
        if on_status is not None:
            on_status(msg)

    def _hide_main() -> None:
        try:
            master.attributes("-topmost", False)
            master.withdraw()
            master.update_idletasks()
        except tk.TclError:
            pass

    def _restore_main() -> None:
        try:
            master.geometry(saved_geo)
            master.deiconify()
            if was_topmost:
                master.attributes("-topmost", True)
            master.lift()
        except tk.TclError:
            pass

    def _wrapped_complete(region: ScreenRegion | None) -> None:
        if selector_holder is not None:
            selector_holder.clear()
        _restore_main()
        on_complete(region)

    def _show_selector() -> None:
        try:
            selector = RegionSelector(master, _wrapped_complete)
            if selector_holder is not None:
                selector_holder.append(selector)
            selector.show()
        except Exception:
            log.exception("Region selector failed")
            _wrapped_complete(None)

    def _start() -> None:
        _status("Selecciona region: arrastra sobre el tablero (ESC cancela).")
        _hide_main()
        _show_selector()

    _run_on_ui(_start)
