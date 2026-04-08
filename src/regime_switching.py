"""Meta-policy: route to uptrend / midtrend / downtrend sub-policies by price trend."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .trend_classifier import TREND_DOWN, TREND_MID, TREND_UP, classify_trend_at
from .trading_env import OptimalExecutionEnv

logger = logging.getLogger(__name__)


def _is_episode_start(obs: np.ndarray, eps: float = 1e-4) -> bool:
    """True when env is at first step of an episode (inventory full, full horizon remaining)."""
    if obs is None or len(obs) < 2:
        return True
    inv = float(obs[0])
    rem = float(obs[1])
    return rem >= 1.0 - eps and inv >= 1.0 - eps


class TWAPFallbackPolicy:
    """Deterministic schedule matching TWAP: residual env uses a=0.5; else fraction 1/(bars_left)."""

    def __init__(self, env: OptimalExecutionEnv) -> None:
        self._env = env

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if getattr(self._env, "_residual_bound", None) is not None:
            return np.array([0.5], dtype=np.float32), None
        rem = float(obs[1]) if len(obs) > 1 else 1.0
        T = max(int(self._env.T), 1)
        bars_left = max(float(rem) * T, 1e-12)
        a = min(1.0, 1.0 / bars_left)
        return np.array([a], dtype=np.float32), None


class RegimeSwitchingPolicy:
    """At each episode start, classify trend and delegate ``predict`` to the chosen sub-policy."""

    def __init__(
        self,
        uptrend_policy: Any,
        midtrend_policy: Any,
        downtrend_policy: Any,
        env: OptimalExecutionEnv,
        lookback: int = 20,
        up_pct: float = 0.02,
        down_pct: float = -0.02,
    ) -> None:
        self.uptrend_policy = uptrend_policy
        self.midtrend_policy = midtrend_policy
        self.downtrend_policy = downtrend_policy
        self.env = env
        self.lookback = int(lookback)
        self.up_pct = float(up_pct)
        self.down_pct = float(down_pct)
        self._active: Any = uptrend_policy
        self._active_trend: int = TREND_MID
        self.policy_selections: dict[str, int] = {
            "up": 0,
            "mid": 0,
            "down": 0,
        }

    def _select_for_row_start(self, row_start: int) -> None:
        trend = classify_trend_at(
            self.env.price_data,
            row_start,
            lookback=self.lookback,
            up_pct=self.up_pct,
            down_pct=self.down_pct,
        )
        self._active_trend = trend
        if trend == TREND_UP:
            self._active = self.uptrend_policy
            self.policy_selections["up"] += 1
        elif trend == TREND_DOWN:
            self._active = self.downtrend_policy
            self.policy_selections["down"] += 1
        else:
            self._active = self.midtrend_policy
            self.policy_selections["mid"] += 1
        logger.debug(
            "RegimeSwitchingPolicy row_start=%s trend=%s -> %s",
            row_start,
            trend,
            type(self._active).__name__,
        )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=np.float32)
        if _is_episode_start(obs):
            row_start = int(getattr(self.env, "_row_start", 0))
            self._select_for_row_start(row_start)
        return self._active.predict(obs, deterministic=deterministic, **kwargs)


def build_regime_switching_policy(
    uptrend_policy: Any,
    downtrend_policy: Any,
    env: OptimalExecutionEnv,
    *,
    midtrend_strategy: str = "twap",
    lookback: int = 20,
    up_pct: float = 0.02,
    down_pct: float = -0.02,
) -> RegimeSwitchingPolicy:
    """Construct ``RegimeSwitchingPolicy`` with TWAP mid (default) or downtrend model in mid slot."""
    twap = TWAPFallbackPolicy(env)
    mid = downtrend_policy if midtrend_strategy == "downtrend_model" else twap
    return RegimeSwitchingPolicy(
        uptrend_policy,
        mid,
        downtrend_policy,
        env,
        lookback=int(lookback),
        up_pct=float(up_pct),
        down_pct=float(down_pct),
    )
