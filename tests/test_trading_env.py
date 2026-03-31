from __future__ import annotations

import numpy as np
import pandas as pd

from src.trading_env import OptimalExecutionEnv, physical_institutional_kwargs


def _panel(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close": 100 + rng.standard_normal(n).cumsum() * 0.5,
            "Volume": rng.integers(1e6, 1e7, size=n).astype(float),
            "sigma_daily": np.abs(rng.standard_normal(n) * 0.01 + 0.015),
            "realised_vol_20": np.full(n, 0.2),
            "amihud_illiquidity": np.abs(rng.standard_normal(n) * 1e-6 + 1e-5),
            "regime": rng.integers(0, 2, size=n),
        },
        index=idx,
    )


def test_episode_terminates():
    env = OptimalExecutionEnv(_panel(), T=5, resample=False, seed=42)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (9,)
    for _ in range(30):
        obs, _, term, trunc, _ = env.step(np.array([0.5], dtype=np.float32))
        if term or trunc:
            break
    assert term or trunc


def test_reset_deterministic():
    env = OptimalExecutionEnv(_panel(40), T=5, seed=123)
    a, _ = env.reset(seed=7)
    env2 = OptimalExecutionEnv(_panel(40), T=5, seed=123)
    b, _ = env2.reset(seed=7)
    np.testing.assert_array_equal(a, b)


def test_physical_institutional_kwargs_off_when_no_notional():
    assert physical_institutional_kwargs(0.0) == {}
    assert physical_institutional_kwargs(None) == {}


def test_physical_institutional_defaults():
    d = physical_institutional_kwargs(1e9)
    assert d["max_inventory_fraction_per_step"] == 0.33
    assert d["is_reward_scale"] == 1.28
    assert d["twap_slice_bonus_coef"] == 0.30


def test_physical_twap_slice_bonus_runs():
    env = OptimalExecutionEnv(
        _panel(50),
        T=6,
        resample=False,
        seed=1,
        order_notional_usd=5e6,
        order_start_bar=0,
        twap_slice_bonus_coef=0.2,
        is_reward_scale=1.0,
        max_inventory_fraction_per_step=0.5,
    )
    obs, _ = env.reset(seed=0)
    for _ in range(8):
        obs, r, term, trunc, info = env.step(np.array([0.9], dtype=np.float32))
        assert "twap_slice_bonus" in info
        assert np.isfinite(r)
        if term or trunc:
            break
