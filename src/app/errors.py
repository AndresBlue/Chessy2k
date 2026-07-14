"""Domain exceptions for the overlay application."""

from __future__ import annotations


class ChessyError(Exception):
    """Base error for Chessy overlay."""


class CaptureError(ChessyError):
    """Screen capture failed."""


class EngineTimeout(ChessyError):
    """Chess engine did not respond in time."""


class EngineError(ChessyError):
    """Chess engine process error."""


class InvalidFEN(ChessyError):
    """FEN string is invalid or unusable."""


class AmbiguousTransition(ChessyError):
    """Multiple legal transitions match the detected board."""
