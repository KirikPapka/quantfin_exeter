from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmarks import (
    almgren_chriss_execution,
    compare_all,
    immediate_execution,
    twap_execution,
)


def _ep(T: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    return pd.DataFrame(
        {
            "Close": np.linspace(100, 100.4, T),
            "Volume": np.full(T, 1e6),
            "sigma_daily": np.full(T, 0.02),
            "amihud_illiquidity": np.full(T, 1e-5),
            "regime": np.zeros(T, dtype=int),
        },
        index=idx,
    )


def test_twap():
    r = twap_execution(1.0, 5, _ep(10), 0)
    assert np.isfinite(r["execution_cost_bps"])


def test_ac():
    r = almgren_chriss_execution(1.0, 10, _ep(10), 0.01, 0.001, 0.02, 0.5, 0)
    assert np.isfinite(r["implementation_shortfall"])


def test_immediate_twap_style_zero_vs_arrival():
    ep = _ep(10)
    r = immediate_execution(1.0, 5, ep, 0, params=None)
    assert r["implementation_shortfall"] == 0.0
    assert r["execution_cost_bps"] == 0.0


def test_immediate_physical_nonzero_with_size():
    ep = _ep(20)
    ep["Volume"] = np.full(len(ep), 5e4)
    params = {"order_notional_usd": 2e9, "order_start_bar": 0}
    r = immediate_execution(1.0, 10, ep, 5, params=params)
    assert np.isfinite(r["implementation_shortfall"])
    assert abs(r["execution_cost_bps"]) > 1e-3


def test_compare_all_fixed_row_starts():
    ep = _ep(25)
    rl = {"mean_is_bps": 1.0, "std_is_bps": 0.1, "completion_rate": 1.0}
    params = {
        "T": 5,
        "X_0": 1.0,
        "seed": 0,
        "n_starts": 99,
        "fixed_row_starts": [0, 2, 5],
    }
    df = compare_all(rl, ep, params)
    assert len(df) == 5
    assert df.iloc[0]["Strategy"] == "RL"
    assert np.isfinite(df.iloc[1]["Mean_IS_bps"])
