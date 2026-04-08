"""Sanity checks: TWAP IS sign on synthetic monotonic paths (legacy mode)."""

from __future__ import annotations

from src.benchmarks import twap_execution
from src.scenario_paths import append_synthetic_scenarios, synthetic_panel


def test_twap_flat_legacy_near_zero():
    df = synthetic_panel("flat", n=40)
    p = {"order_notional_usd": 0.0, "order_start_bar": 0, "T": 10}
    r = twap_execution(1.0, 10, df, 0, params=p)
    assert abs(r["implementation_shortfall"]) < 1e-9


def test_twap_up_legacy_positive_is():
    df = synthetic_panel("up", n=40, step=0.5)
    p = {"order_notional_usd": 0.0, "order_start_bar": 0, "T": 10}
    r = twap_execution(1.0, 10, df, 0, params=p)
    assert r["implementation_shortfall"] > 1e-6


def test_append_synthetic_scenarios_extends_rows():
    base = synthetic_panel("flat", n=30)
    base["regime"] = 1
    out = append_synthetic_scenarios(base, "up", n_bars=25)
    assert len(out) == 55


def test_twap_down_legacy_negative_is():
    df = synthetic_panel("down", n=40, step=0.5)
    p = {"order_notional_usd": 0.0, "order_start_bar": 0, "T": 10}
    r = twap_execution(1.0, 10, df, 0, params=p)
    assert r["implementation_shortfall"] < -1e-6
