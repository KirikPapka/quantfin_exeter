"""Ensemble policy: median-aggregation of multiple SB3 models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EnsemblePolicy:
    """Combine N trained models by taking the median action at inference.

    Usage::

        models = load_ensemble_from_dir("models/ensemble/", algo="PPO")
        ens = EnsemblePolicy(models)
        stats = evaluate_agent(ens, env, bench_params=bp)
    """

    def __init__(self, models: list[Any]) -> None:
        if not models:
            raise ValueError("EnsemblePolicy requires at least one model")
        self.models = models
        logger.info("EnsemblePolicy created with %d models", len(models))

    def predict(
        self, observation: np.ndarray, deterministic: bool = True, **kwargs: Any
    ) -> tuple[np.ndarray, None]:
        actions = []
        for m in self.models:
            act, _ = m.predict(observation, deterministic=deterministic)
            actions.append(act)
        median_action = np.median(np.array(actions), axis=0)
        return median_action, None


def load_ensemble_from_dir(
    model_dir: str | Path,
    algo: str = "PPO",
    device: str = "auto",
) -> list[Any]:
    """Load all ``.zip`` models from a directory."""
    from stable_baselines3 import PPO, SAC

    cls = PPO if algo.upper() == "PPO" else SAC
    model_dir = Path(model_dir)
    zips = sorted(model_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip models found in {model_dir}")
    models = []
    for p in zips:
        m = cls.load(str(p), device=device)
        models.append(m)
        logger.info("Loaded ensemble member: %s", p.name)
    return models


def train_ensemble(
    env_factory,
    n_seeds: int = 3,
    base_seed: int = 42,
    algo: str = "PPO",
    total_timesteps: int = 100_000,
    save_dir: str = "models/ensemble",
    **train_kwargs: Any,
) -> EnsemblePolicy:
    """Train ``n_seeds`` models with different seeds and return an ensemble."""
    from .rl_agent import train_agent

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    models = []
    for i in range(n_seeds):
        seed = base_seed + i * 7919
        logger.info("Training ensemble member %d/%d (seed=%d)", i + 1, n_seeds, seed)
        model = train_agent(
            env_factory,
            algorithm=algo,
            total_timesteps=total_timesteps,
            seed=seed,
            save_path=str(save_path / f"seed_{seed}"),
            **train_kwargs,
        )
        model.save(str(save_path / f"{algo.lower()}_seed{seed}.zip"))
        models.append(model)
    return EnsemblePolicy(models)
