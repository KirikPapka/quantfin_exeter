from __future__ import annotations

import numpy as np
import pandas as pd

from src.trading_env import OptimalExecutionEnv
from src.ui_rollout import rollout_episode


class _A:
    def predict(self, obs, deterministic=False):  # noqa: ANN001
        return np.array([0.4], dtype=np.float32), None


def _panel(n: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close": 100 + rng.standard_normal(n).cumsum() * 0.3,
            "Volume": rng.integers(1e6, 2e6, size=n).astype(float),
            "sigma_daily": np.full(n, 0.02),
            "realised_vol_20": np.full(n, 0.2),
            "amihud_illiquidity": np.full(n, 1e-5),
            "regime": rng.integers(0, 2, size=n),
        },
        index=idx,
    )


def test_rollout():
    env = OptimalExecutionEnv(_panel(), T=8, resample=False, seed=1)
    traj, summ = rollout_episode(env, _A(), seed=2, deterministic=True)
    assert len(traj) >= 1 and "is_bps" in summ
