"""Windows DPI awareness helpers (call before Tk init)."""

from __future__ import annotations

import sys


def set_process_dpi_aware() -> None:
    """Enable per-monitor DPI awareness so mss and Tk share physical pixels."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        import ctypes

        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass
