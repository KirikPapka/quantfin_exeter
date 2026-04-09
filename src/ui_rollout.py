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
    row_start: int | None = None,
) -> Tuple[pd.DataFrame, dict[str, float]]:
    obs, _ = env.reset(seed=seed)
    if row_start is not None:
        # Override the randomly-sampled start to a specific bar index.
        # Mirrors the fixed_starts logic used in evaluate_agent.
        env._row_start = row_start  # noqa: SLF001
        env._t = 0  # noqa: SLF001
        env._X = env.X_0  # noqa: SLF001
        env._S0 = float(env.price_data.iloc[row_start]["Close"])  # noqa: SLF001
        if getattr(env, "_physical", False):
            from .execution_impact import arrival_price_full
            env._arrival = float(arrival_price_full(env.price_data, row_start, env._t0))  # noqa: SLF001
            env._Q_shares = float(env.order_notional_usd) / max(env._arrival, 1e-12)  # noqa: SLF001
            env._notional_scale = max(env._Q_shares * env._arrival, 1e-12)  # noqa: SLF001
        else:
            env._arrival = env._S0  # noqa: SLF001
        env._ep_is_num = 0.0  # noqa: SLF001
        obs = env._obs()  # noqa: SLF001
    arrival = float(env._arrival)  # noqa: SLF001
    physical = bool(getattr(env, "_physical", False))
    rows: List[dict[str, float]] = []
    step = 0
    total_cost = 0.0
    terminated = False
    while not terminated:
        x_before = float(env._X)  # noqa: SLF001
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)
        terminated = bool(term or trunc)
        v = float(info.get("v_shares", info.get("v_t", 0.0)))
        px = float(info.get("exec_price", arrival))
        total_cost += v * (px - arrival)
        date_idx = env._row_start + step  # noqa: SLF001
        date_val = env.price_data.index[date_idx] if date_idx < len(env.price_data) else None
        rows.append(
            {
                "step": step,
                "date": str(date_val.date()) if date_val is not None else str(step),
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
        x_out = (x_final / max(env.X_0, 1e-12)) * float(env._Q_shares) if physical else x_final
        total_cost += x_out * (last_px - arrival)
    completed = x_final <= 0.01 * env.X_0
    if physical:
        denom = float(getattr(env, "_notional_scale", env.X_0 * arrival))
        is_bps = (total_cost / max(denom, 1e-12)) * 1e4 if arrival else 0.0
    else:
        is_bps = ((total_cost / env.X_0) / arrival * 1e4) if arrival else 0.0
    return pd.DataFrame(rows), {
        "is_bps": float(is_bps),
        "completed": bool(completed),
        "arrival": arrival,
        "steps": int(step),
        "x_final": x_final,
    }
