"""Single-episode rollout for UI."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from .trading_env import OptimalExecutionEnv


def rollout_episode(
    env: OptimalExecutionEnv,
    agent: Any,
    *,
    seed: int | None = None,
    deterministic: bool = True,
) -> Tuple[pd.DataFrame, dict[str, float]]:
    obs, _ = env.reset(seed=seed)
    arrival = float(env._arrival)  # noqa: SLF001
    rows: List[dict[str, float]] = []
    step = 0
    total_cost = 0.0
    terminated = False
    while not terminated:
        x_before = float(env._X)  # noqa: SLF001
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)
        terminated = bool(term or trunc)
        v = float(info.get("v_t", 0.0))
        px = float(info.get("exec_price", arrival))
        total_cost += v * (px - arrival)
        rows.append(
            {
                "step": step,
                "inventory_before": x_before,
                "inventory_after": float(env._X),  # noqa: SLF001
                "action_frac": float(np.clip(action[0], 0, 1)),
                "reward": float(reward),
                "regime": float(info.get("regime", 0)),
                "v_t": v,
                "exec_price": px,
            }
        )
        step += 1
    x_final = float(env._X)  # noqa: SLF001
    if x_final > 0.01 * env.X_0:
        rel = max(min(env._t, env.T) - 1, 0)  # noqa: SLF001
        last_px = float(env.price_data.iloc[env._row_start + rel]["Close"])  # noqa: SLF001
        total_cost += x_final * (last_px - arrival)
    completed = x_final <= 0.01 * env.X_0
    is_bps = ((total_cost / env.X_0) / arrival * 1e4) if arrival else 0.0
    return pd.DataFrame(rows), {
        "is_bps": float(is_bps),
        "completed": bool(completed),
        "arrival": arrival,
        "steps": int(step),
        "x_final": x_final,
    }
