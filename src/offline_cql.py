"""Offline RL with Conservative Q-Learning (CQL) via d3rlpy.

Optional module -- requires ``pip install d3rlpy``.
Collects a dataset from TWAP + random-perturbation rollouts, then trains CQL-SAC offline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .trading_env import OptimalExecutionEnv

logger = logging.getLogger(__name__)


def collect_offline_dataset(
    env: OptimalExecutionEnv,
    n_episodes: int = 2000,
    seed: int = 42,
    twap_frac: float = 0.6,
    perturb_std: float = 0.1,
) -> dict[str, np.ndarray]:
    """Collect transitions from a mix of TWAP and perturbed-TWAP rollouts.

    Returns dict with keys: observations, actions, rewards, terminals, next_observations.
    """
    rng = np.random.default_rng(seed)
    t0 = env._t0
    T = env.T
    residual_mode = env._residual_bound is not None

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    rew_buf: list[float] = []
    term_buf: list[bool] = []
    next_obs_buf: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        use_twap = rng.random() < twap_frac
        terminated = False
        while not terminated:
            if residual_mode:
                base_a = 0.5
            else:
                T_eff = max(T - t0, 1)
                trading_step = env._t - t0
                if env._t < t0:
                    base_a = 0.0
                else:
                    bars_left = max(T_eff - trading_step, 1)
                    base_a = 1.0 / bars_left

            if use_twap:
                noise = float(rng.normal(0, perturb_std))
            else:
                noise = float(rng.normal(0, perturb_std * 3))

            action = np.array([float(np.clip(base_a + noise, 0.0, 1.0))], dtype=np.float32)

            obs_buf.append(obs.copy())
            act_buf.append(action.copy())

            next_obs, reward, term, trunc, _ = env.step(action)
            terminated = bool(term or trunc)

            rew_buf.append(float(reward))
            term_buf.append(terminated)
            next_obs_buf.append(next_obs.copy())
            obs = next_obs

    return {
        "observations": np.array(obs_buf, dtype=np.float32),
        "actions": np.array(act_buf, dtype=np.float32),
        "rewards": np.array(rew_buf, dtype=np.float32),
        "terminals": np.array(term_buf, dtype=np.float32),
        "next_observations": np.array(next_obs_buf, dtype=np.float32),
    }


class CQLAgent:
    """Wrapper around a d3rlpy CQL model for compatibility with ``evaluate_agent``."""

    def __init__(self, algo: Any) -> None:
        self._algo = algo

    def predict(
        self, observation: np.ndarray, deterministic: bool = True, **kwargs: Any
    ) -> tuple[np.ndarray, None]:
        obs = np.atleast_2d(observation).astype(np.float32)
        action = self._algo.predict(obs)[0]
        return np.array(action, dtype=np.float32).reshape(-1), None


def train_cql(
    env: OptimalExecutionEnv,
    n_episodes: int = 2000,
    seed: int = 42,
    n_steps: int = 50_000,
    save_path: Optional[str] = None,
    **cql_kwargs: Any,
) -> CQLAgent:
    """End-to-end: collect offline data and train CQL.

    Requires ``d3rlpy`` to be installed.
    """
    try:
        import d3rlpy
    except ImportError as e:
        raise ImportError(
            "d3rlpy is required for offline CQL. Install with: pip install d3rlpy"
        ) from e

    logger.info("Collecting offline dataset (%d episodes)...", n_episodes)
    data = collect_offline_dataset(env, n_episodes=n_episodes, seed=seed)
    logger.info(
        "Dataset: %d transitions, reward range [%.4f, %.4f]",
        len(data["rewards"]),
        float(np.min(data["rewards"])),
        float(np.max(data["rewards"])),
    )

    dataset = d3rlpy.dataset.MDPDataset(
        observations=data["observations"],
        actions=data["actions"],
        rewards=data["rewards"],
        terminals=data["terminals"],
    )

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=cql_kwargs.get("actor_lr", 1e-4),
        critic_learning_rate=cql_kwargs.get("critic_lr", 3e-4),
        alpha_learning_rate=cql_kwargs.get("alpha_lr", 1e-4),
        batch_size=cql_kwargs.get("batch_size", 256),
    ).create(device="cuda:0" if _has_cuda() else "cpu:0")

    logger.info("Training CQL for %d steps...", n_steps)
    cql.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=min(1000, n_steps),
        show_progress=True,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cql.save(save_path)
        logger.info("Saved CQL model to %s", save_path)

    return CQLAgent(cql)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
