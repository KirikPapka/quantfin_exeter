"""Lightweight policy wrappers (no retraining).

This module provides:
- A simple TWAP baseline implemented as a *policy* (``predict(obs)``) so it can be
  evaluated inside ``OptimalExecutionEnv`` alongside RL.
- A switching wrapper that routes decisions to RL vs TWAP based on a simple
  regime/trend signal extracted from the observation.

The goal is to support the narrative: *RL is conditionally superior, so deploy it
selectively*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np

# Observation layout from ``OptimalExecutionEnv._obs``:
# [inventory, rem, S_ratio, liq_z, sig_z, regime, pva, twap_gap, news_z]
OBS_INVENTORY_FRAC = 0
OBS_REM_FRAC = 1
OBS_REGIME = 5
OBS_PVA = 6


class PredictPolicy(Protocol):
    def predict(self, obs, deterministic: bool = False):  # noqa: ANN001
        """Return (action, state) per Stable-Baselines3 convention."""


TrendLabel = Literal["up", "down", "flat"]


def _infer_step_from_obs(obs: np.ndarray, T: int) -> int:
    """Infer integer step ``t`` from ``rem = (T-t)/T``.

    ``OptimalExecutionEnv`` computes ``rem`` using the *current* ``t`` before stepping.
    """
    rem = float(np.asarray(obs)[OBS_REM_FRAC])
    rem = float(np.clip(rem, 0.0, 1.0))
    t = int(round(float(T) * (1.0 - rem)))
    return int(np.clip(t, 0, max(int(T) - 1, 0)))


def detect_trend_from_obs(
    obs: np.ndarray,
    *,
    pva_threshold: float = 0.0,
) -> TrendLabel:
    """Classify a simple price trend using the env's PVA feature.

    ``pva`` is (close / arrival - 1) clipped to [-0.5, 0.5].
    """
    pva = float(np.asarray(obs)[OBS_PVA])
    th = float(max(pva_threshold, 0.0))
    if pva > th:
        return "up"
    if pva < -th:
        return "down"
    return "flat"


@dataclass
class TWAPPolicy:
    """TWAP as a *policy* for ``OptimalExecutionEnv``.

    The env action is a fraction of remaining inventory to sell at each step.
    Equal-share TWAP therefore uses ``action = 1 / remaining_steps`` (and 0.0 before
    ``order_start_bar`` when using the physical notional mode).
    """

    T: int
    order_start_bar: int = 0

    def predict(self, obs, deterministic: bool = False):  # noqa: ANN001
        o = np.asarray(obs)
        t = _infer_step_from_obs(o, int(self.T))
        if t < int(self.order_start_bar):
            a = 0.0
        else:
            remaining_steps = max(int(self.T) - t, 1)
            a = 1.0 / float(remaining_steps)
        return np.array([a], dtype=np.float32), None


@dataclass
class SwitchingPolicy:
    """Wrapper that switches between two policies based on a signal."""

    primary: PredictPolicy
    fallback: PredictPolicy
    decide: Callable[[np.ndarray], bool]

    def predict(self, obs, deterministic: bool = False):  # noqa: ANN001
        o = np.asarray(obs)
        use_primary = bool(self.decide(o))
        if use_primary:
            return self.primary.predict(o, deterministic=deterministic)
        return self.fallback.predict(o, deterministic=deterministic)


def primary_if_trend_up(
    *,
    pva_threshold: float = 0.0,
) -> Callable[[np.ndarray], bool]:
    """Decision function: use primary for up/flat, fallback for down."""

    def _fn(obs: np.ndarray) -> bool:
        return detect_trend_from_obs(obs, pva_threshold=pva_threshold) != "down"

    return _fn


def primary_if_regime_calm(
    *,
    calm_regime: int = 0,
) -> Callable[[np.ndarray], bool]:
    """Decision function: use primary if ``regime == calm_regime``."""

    def _fn(obs: np.ndarray) -> bool:
        r = int(round(float(np.asarray(obs)[OBS_REGIME])))
        return r == int(calm_regime)

    return _fn


def evaluate_switching_vs_baselines(
    rl_policy: PredictPolicy,
    env,
    *,
    n_episodes: int = 100,
    seed: int = 42,
    pva_threshold: float = 0.0,
):
    """Evaluate RL vs TWAP vs Switching on the same env (no training).

    Returns a dict mapping labels -> stats, where ``stats`` matches the output of
    ``src.rl_agent.evaluate_agent``.
    """

    from .rl_agent import evaluate_agent

    T = int(getattr(env, "T"))
    t0 = int(getattr(env, "order_start_bar", 0) or 0)
    twap = TWAPPolicy(T=T, order_start_bar=t0)
    switch = SwitchingPolicy(
        primary=rl_policy,
        fallback=twap,
        decide=primary_if_trend_up(pva_threshold=pva_threshold),
    )
    return {
        "RL": evaluate_agent(rl_policy, env, n_episodes=n_episodes, seed=seed),
        "TWAP": evaluate_agent(twap, env, n_episodes=n_episodes, seed=seed),
        "Switching": evaluate_agent(switch, env, n_episodes=n_episodes, seed=seed),
    }
