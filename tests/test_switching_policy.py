from __future__ import annotations

import numpy as np

from src.switching_policy import SwitchingPolicy, TWAPPolicy, primary_if_trend_up


class _Const:
    def __init__(self, a: float) -> None:
        self.a = float(a)

    def predict(self, obs, deterministic: bool = False):  # noqa: ANN001
        return np.array([self.a], dtype=np.float32), None


def _obs(*, pva: float, rem: float = 1.0, regime: float = 0.0) -> np.ndarray:
    # [inventory, rem, S_ratio, liq_z, sig_z, regime, pva, twap_gap, news_z]
    return np.array([1.0, rem, 1.0, 0.0, 0.0, regime, pva, 0.0, 0.0], dtype=np.float32)


def test_twap_policy_equal_share_action():
    pol = TWAPPolicy(T=10, order_start_bar=0)
    # At t=0, remaining steps=10 => action=0.1
    a0, _ = pol.predict(_obs(pva=0.0, rem=1.0))
    assert np.isclose(float(a0[0]), 0.1)
    # At t=5, rem=(10-5)/10=0.5 => remaining steps=5 => action=0.2
    a5, _ = pol.predict(_obs(pva=0.0, rem=0.5))
    assert np.isclose(float(a5[0]), 0.2)


def test_switching_policy_uses_fallback_on_down_trend():
    primary = _Const(0.9)
    fallback = _Const(0.1)
    pol = SwitchingPolicy(primary=primary, fallback=fallback, decide=primary_if_trend_up())
    a_up, _ = pol.predict(_obs(pva=+0.02))
    a_dn, _ = pol.predict(_obs(pva=-0.02))
    assert np.isclose(float(a_up[0]), 0.9)
    assert np.isclose(float(a_dn[0]), 0.1)
