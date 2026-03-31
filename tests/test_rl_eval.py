from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl_agent import evaluate_agent, format_rl_eval_report
from src.trading_env import OptimalExecutionEnv


def _panel(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close": 100 + rng.standard_normal(n).cumsum() * 0.2,
            "Volume": rng.integers(1e6, 2e6, size=n).astype(float),
            "sigma_daily": np.full(n, 0.02),
            "realised_vol_20": np.full(n, 0.2),
            "amihud_illiquidity": np.full(n, 1e-5),
            "regime": rng.integers(0, 2, size=n),
        },
        index=idx,
    )


class _Rand:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def predict(self, obs, deterministic=False):  # noqa: ANN001
        return self._rng.uniform(0.0, 1.0, size=(1,)), None


def test_evaluate_agent_path_aligned_keys():
    env = OptimalExecutionEnv(_panel(), T=8, resample=True, seed=1)
    bp = {
        "T": 8,
        "X_0": 1.0,
        "eta": 0.01,
        "gamma": 0.001,
        "lam": 0.5,
        "order_notional_usd": 0.0,
        "order_start_bar": 0,
    }
    st = evaluate_agent(_Rand(), env, n_episodes=15, seed=3, bench_params=bp)
    assert "mean_rl_minus_twap_bps" in st
    assert "pct_beat_twap_is" in st
    assert 0.0 <= st["pct_beat_twap_is"] <= 1.0
    assert "mean_episode_return" in st
    txt = format_rl_eval_report(st)
    assert "RL_minus_TWAP" in txt
