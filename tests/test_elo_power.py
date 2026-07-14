"""Tests for Elo power slider mapping."""

from src.app.elo_power import (
    ELO_MAX_HUMAN,
    ELO_MIN,
    MAX_EFFORT_ELO,
    STOCKFISH_UCI_ELO_MIN,
    elo_to_slider,
    format_elo_label,
    is_below_stockfish_floor,
    is_max_effort,
    slider_to_elo,
    stockfish_uci_elo,
)


def test_slider_roundtrip():
    for elo in (ELO_MIN, 800, 1320, 1500, 2000, ELO_MAX_HUMAN):
        pos = elo_to_slider(elo)
        assert slider_to_elo(pos) == elo


def test_max_effort_sentinel():
    assert is_max_effort(MAX_EFFORT_ELO)
    assert slider_to_elo(100) == MAX_EFFORT_ELO
    assert format_elo_label(MAX_EFFORT_ELO) == "Maximo esfuerzo"


def test_elo_label_human():
    assert "1850" in format_elo_label(1850)


def test_stockfish_uci_elo_floor():
    assert stockfish_uci_elo(600) == STOCKFISH_UCI_ELO_MIN
    assert stockfish_uci_elo(1000) == STOCKFISH_UCI_ELO_MIN
    assert stockfish_uci_elo(1500) == 1500
    assert is_below_stockfish_floor(800)
    assert not is_below_stockfish_floor(1500)
