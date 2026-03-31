"""SB3 training + evaluation."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .benchmarks import twap_execution, vwap_execution
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
    policy_kwargs = {"net_arch": dict(pi=[128, 128], vf=[128, 128])}
    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=15,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.008,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            tau=0.005,
            policy_kwargs=policy_kwargs,
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


def _schedule_is_bps_on_path(
    env: OptimalExecutionEnv,
    start: int,
    bench_params: dict[str, Any],
    kind: str,
) -> float:
    try:
        if kind == "twap":
            r = twap_execution(
                env.X_0,
                env.T,
                env.price_data,
                start,
                params=bench_params,
            )
        else:
            r = vwap_execution(
                env.X_0,
                env.T,
                env.price_data,
                start,
                params=bench_params,
            )
        return float(r["implementation_shortfall"]) * 1e4
    except Exception as e:  # noqa: BLE001
        logger.debug("schedule %s skip start=%s: %s", kind, start, e)
        return float("nan")


def format_rl_eval_report(stats: dict[str, Any]) -> str:
    """Human-readable block for logs / UI (see README for field definitions)."""
    lines = [f"  mean_episode_return        {stats.get('mean_episode_return', float('nan')):.6f}"]
    if "mean_twap_is_bps_path" in stats:
        lines += [
            f"  std_episode_return         {stats.get('std_episode_return', float('nan')):.6f}",
            f"  mean_twap_is_bps (paths)   {stats['mean_twap_is_bps_path']:.4f}",
            f"  mean_vwap_is_bps (paths)   {stats['mean_vwap_is_bps_path']:.4f}",
            f"  mean_RL_minus_TWAP_bps     {stats['mean_rl_minus_twap_bps']:.4f}  "
            f"(>0 ⇒ RL higher avg sell vs arrival than TWAP on same windows)",
            f"  mean_RL_minus_VWAP_bps     {stats['mean_rl_minus_vwap_bps']:.4f}",
            f"  pct_beat_TWAP_IS           {stats['pct_beat_twap_is']:.2%}",
            f"  pct_beat_VWAP_IS           {stats['pct_beat_vwap_is']:.2%}",
        ]
    else:
        lines.append("  (path-aligned TWAP/VWAP: pass bench_params=... to evaluate_agent)")
    return "\n".join(lines)


def evaluate_agent(
    agent,
    env: OptimalExecutionEnv,
    n_episodes: int = 100,
    seed: int = 42,
    *,
    bench_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate policy on ``env``.

    When ``bench_params`` is set (same keys as ``compare_all`` / ``twap_execution``), computes
    TWAP and VWAP implementation shortfall on the **identical** ``(start, T)`` window as each RL
    episode so ``mean_rl_minus_twap_bps`` and ``pct_beat_twap_is`` are apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    is_list: list[float] = []
    ret_list: list[float] = []
    twap_path: list[float] = []
    vwap_path: list[float] = []
    diff_t: list[float] = []
    diff_v: list[float] = []
    beat_t = beat_v = 0
    valid_twap = 0
    valid_vwap = 0
    completed = 0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        arrival = env._arrival  # noqa: SLF001
        start = int(env._row_start)  # noqa: SLF001
        total_cost = 0.0
        ep_ret = 0.0
        terminated = False
        physical = bool(getattr(env, "_physical", False))
        while not terminated:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_ret += float(reward)
            terminated = bool(term or trunc)
            v = float(info.get("v_shares", info.get("v_t", 0.0)))
            px = float(info.get("exec_price", arrival))
            total_cost += v * (px - arrival)
        x_final = env._X  # noqa: SLF001
        if x_final <= 0.01 * env.X_0:
            completed += 1
        else:
            rel = max(min(env._t, env.T) - 1, 0)  # noqa: SLF001
            last_px = float(env.price_data.iloc[env._row_start + rel]["Close"])  # noqa: SLF001
            x_out = (x_final / max(env.X_0, 1e-12)) * float(env._Q_shares) if physical else x_final
            total_cost += x_out * (last_px - arrival)
        if physical:
            denom = float(getattr(env, "_notional_scale", env.X_0 * arrival))
            is_bps = (total_cost / max(denom, 1e-12)) * 1e4 if arrival else 0.0
        else:
            is_bps = ((total_cost / env.X_0) / arrival * 1e4) if arrival else 0.0
        is_bps_f = float(is_bps)
        is_list.append(is_bps_f)
        ret_list.append(ep_ret)

        if bench_params is not None:
            tb = _schedule_is_bps_on_path(env, start, bench_params, "twap")
            vb = _schedule_is_bps_on_path(env, start, bench_params, "vwap")
            twap_path.append(tb)
            vwap_path.append(vb)
            if np.isfinite(tb):
                valid_twap += 1
                d = is_bps_f - tb
                diff_t.append(d)
                if is_bps_f > tb:
                    beat_t += 1
            if np.isfinite(vb):
                valid_vwap += 1
                diff_v.append(is_bps_f - vb)
                if is_bps_f > vb:
                    beat_v += 1

    out: dict[str, Any] = {
        "mean_is_bps": float(np.mean(is_list)) if is_list else 0.0,
        "std_is_bps": float(np.std(is_list)) if is_list else 0.0,
        "completion_rate": completed / max(n_episodes, 1),
        "mean_episode_return": float(np.mean(ret_list)) if ret_list else 0.0,
        "std_episode_return": float(np.std(ret_list)) if ret_list else 0.0,
    }
    if bench_params is not None:
        out["mean_twap_is_bps_path"] = float(np.nanmean(twap_path)) if twap_path else float("nan")
        out["mean_vwap_is_bps_path"] = float(np.nanmean(vwap_path)) if vwap_path else float("nan")
        out["mean_rl_minus_twap_bps"] = float(np.mean(diff_t)) if diff_t else float("nan")
        out["mean_rl_minus_vwap_bps"] = float(np.mean(diff_v)) if diff_v else float("nan")
        out["std_rl_minus_twap_bps"] = float(np.std(diff_t)) if diff_t else float("nan")
        out["pct_beat_twap_is"] = beat_t / max(valid_twap, 1)
        out["pct_beat_vwap_is"] = beat_v / max(valid_vwap, 1)
    return out
