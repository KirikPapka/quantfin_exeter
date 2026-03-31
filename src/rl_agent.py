"""SB3 training + evaluation."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .trading_env import OptimalExecutionEnv

logger = logging.getLogger(__name__)


def train_agent(
    env: OptimalExecutionEnv,
    algorithm: str = "PPO",
    total_timesteps: int = 200_000,
    save_path: str = "models/",
    log_path: str = "logs/",
    seed: int = 42,
):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    save_dir = Path(save_path)
    log_dir = Path(log_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    algo = algorithm.upper()
    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed,
        )
    ckpt = CheckpointCallback(save_freq=50_000, save_path=str(save_dir / "ckpt"), name_prefix=algo)
    model.learn(total_timesteps=total_timesteps, callback=ckpt, progress_bar=True)
    out = save_dir / f"{algo}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(str(out))
    logger.info("Saved %s", out)
    return model


def evaluate_agent(agent, env: OptimalExecutionEnv, n_episodes: int = 100, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    is_list: list[float] = []
    completed = 0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        arrival = env._arrival  # noqa: SLF001
        total_cost = 0.0
        terminated = False
        while not terminated:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            terminated = bool(term or trunc)
            v = float(info.get("v_t", 0.0))
            px = float(info.get("exec_price", arrival))
            total_cost += v * (px - arrival)
        x_final = env._X  # noqa: SLF001
        if x_final <= 0.01 * env.X_0:
            completed += 1
        else:
            rel = max(min(env._t, env.T) - 1, 0)  # noqa: SLF001
            last_px = float(env.price_data.iloc[env._row_start + rel]["Close"])  # noqa: SLF001
            total_cost += x_final * (last_px - arrival)
        is_list.append(((total_cost / env.X_0) / arrival * 1e4) if arrival else 0.0)
    return {
        "mean_is_bps": float(np.mean(is_list)) if is_list else 0.0,
        "std_is_bps": float(np.std(is_list)) if is_list else 0.0,
        "completion_rate": completed / max(n_episodes, 1),
    }
