"""SB3 training + evaluation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .benchmarks import twap_execution, vwap_execution
from .trading_env import OptimalExecutionEnv

logger = logging.getLogger(__name__)


class PathAlignedEvalCallback:
    """Save model when ``mean_RL_minus_TWAP_bps`` improves (requires ``bench_params``)."""

    def __init__(
        self,
        eval_env: OptimalExecutionEnv,
        bench_params: dict[str, Any],
        eval_freq_timesteps: int,
        save_path: Path,
        n_eval_episodes: int = 24,
        eval_seed: int = 0,
        verbose: int = 0,
    ) -> None:
        self._eval_env = eval_env
        self._bench_params = bench_params
        self._eval_freq = int(eval_freq_timesteps)
        self._save_path = Path(save_path)
        self._n_eval_episodes = int(n_eval_episodes)
        self._eval_seed = int(eval_seed)
        self._verbose = int(verbose)
        self._last_eval_ts = 0
        self.best_mean_gap = float("-inf")

    def make_callback(self):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _PathAlignedEvalCallback(BaseCallback):
            def __init__(inner_self) -> None:
                super().__init__(outer._verbose)

            def _on_step(inner_self) -> bool:
                if outer._eval_freq <= 0:
                    return True
                ts = int(inner_self.num_timesteps)
                if ts < outer._eval_freq:
                    return True
                if ts - outer._last_eval_ts < outer._eval_freq:
                    return True
                outer._last_eval_ts = ts
                stats = evaluate_agent(
                    inner_self.model,
                    outer._eval_env,
                    n_episodes=outer._n_eval_episodes,
                    seed=outer._eval_seed + ts,
                    bench_params=outer._bench_params,
                )
                gap = float(stats.get("mean_rl_minus_twap_bps", float("nan")))
                if inner_self.logger is not None and np.isfinite(gap):
                    inner_self.logger.record("eval/mean_rl_minus_twap_bps", gap)
                if np.isfinite(gap) and gap > outer.best_mean_gap:
                    outer.best_mean_gap = gap
                    inner_self.model.save(str(outer._save_path))
                    logger.info(
                        "Best checkpoint mean_RL_minus_TWAP_bps=%.4f → %s",
                        gap,
                        outer._save_path,
                    )
                return True

        return _PathAlignedEvalCallback()


def _linear_schedule(initial: float, final: float) -> Callable[[float], float]:
    """Linear LR decay from ``initial`` to ``final`` over training."""
    def _schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining
    return _schedule


def train_agent(
    env_factory: Callable[[int], OptimalExecutionEnv],
    algorithm: str = "PPO",
    total_timesteps: int = 200_000,
    n_envs: int = 1,
    save_path: str = "models/",
    log_path: str = "logs/",
    seed: int = 42,
    *,
    eval_env: OptimalExecutionEnv | None = None,
    bench_params: dict[str, Any] | None = None,
    eval_freq_timesteps: int = 0,
    n_eval_episodes: int = 24,
    best_model_filename: str = "best_ppo_twap_gap.zip",
    lr_schedule: str = "linear",
    ent_coef: float | None = None,
    bc_state_dict: Any | None = None,
) -> Any:
    """Train PPO/SAC on a vectorized env built from ``env_factory(rank)``.

    When ``eval_freq_timesteps > 0`` and ``eval_env`` + ``bench_params`` are set, saves the
    model that maximizes path-aligned ``mean_RL_minus_TWAP_bps`` to ``save_path/best_model_filename``.
    """
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    save_dir = Path(save_path)
    log_dir = Path(log_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    algo = algorithm.upper()
    n_envs = max(1, int(n_envs))

    thunks: list[Callable[[], OptimalExecutionEnv]] = []
    for rank in range(n_envs):

        def _make_thunk(r: int = rank) -> Callable[[], OptimalExecutionEnv]:
            def _init() -> OptimalExecutionEnv:
                return env_factory(r)

            return _init

        thunks.append(_make_thunk())

    vec_env = DummyVecEnv(thunks)

    policy_kwargs = {"net_arch": dict(pi=[128, 128], vf=[128, 128])}

    lr: Any
    if lr_schedule == "linear":
        lr = _linear_schedule(3e-4, 5e-5)
    else:
        lr = 2.5e-4

    ppo_ent = ent_coef if ent_coef is not None else 0.003

    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=128,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ppo_ent,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed,
        )
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            tau=0.005,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=seed,
        )

    if bc_state_dict is not None:
        model.policy.load_state_dict(bc_state_dict, strict=False)
        logger.info("Loaded BC warmstart weights into policy")
    ckpt = CheckpointCallback(
        save_freq=50_000, save_path=str(save_dir / "ckpt"), name_prefix=algo
    )
    callbacks: list[Any] = [ckpt]
    if (
        eval_freq_timesteps > 0
        and eval_env is not None
        and bench_params is not None
    ):
        best_path = save_dir / best_model_filename
        ev = PathAlignedEvalCallback(
            eval_env,
            bench_params,
            eval_freq_timesteps,
            best_path,
            n_eval_episodes=n_eval_episodes,
            eval_seed=seed,
            verbose=0,
        )
        callbacks.append(ev.make_callback())
    cb = CallbackList(callbacks)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True)
    out = save_dir / f"{algo}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(str(out))
    logger.info("Saved %s", out)
    vec_env.close()
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
            f"(>0 => RL higher avg sell vs arrival than TWAP on same windows)",
            f"  std_RL_minus_TWAP_bps      {stats.get('std_rl_minus_twap_bps', float('nan')):.4f}",
            f"  mean_RL_minus_VWAP_bps     {stats['mean_rl_minus_vwap_bps']:.4f}",
            f"  pct_beat_TWAP_IS           {stats['pct_beat_twap_is']:.2%}",
            f"  pct_beat_VWAP_IS           {stats['pct_beat_vwap_is']:.2%}",
        ]
        ci = stats.get("ci95_rl_minus_twap_bps")
        if ci is not None:
            lo, hi = ci
            lines.append(f"  IS-gap Sharpe              {stats.get('is_gap_sharpe', float('nan')):.4f}")
            lines.append(f"  95% CI (TWAP gap)          [{lo:.4f}, {hi:.4f}]")
        for key in sorted(stats):
            if key.startswith("mean_rl_minus_twap_bps_regime"):
                regime = key.split("regime")[-1]
                n_key = f"n_episodes_regime{regime}"
                n_ep = stats.get(n_key, "?")
                lines.append(f"  TWAP gap regime {regime}         {stats[key]:.4f}  (n={n_ep})")
    else:
        lines.append("  (path-aligned TWAP/VWAP: pass bench_params=... to evaluate_agent)")
    return "\n".join(lines)


def generate_fixed_eval_starts(
    env: OptimalExecutionEnv,
    n: int = 200,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Pre-sample deterministic ``(row_start, reset_seed)`` pairs for reproducible eval."""
    rng = np.random.default_rng(seed)
    max_start = len(env.price_data) - env.T - 1
    if max_start < 0:
        raise ValueError("price_data too short for fixed starts")
    starts: list[tuple[int, int]] = []
    for _ in range(n):
        row = int(rng.integers(0, max_start + 1))
        sd = int(rng.integers(0, 2**31 - 1))
        starts.append((row, sd))
    return starts


def save_fixed_eval_starts(starts: list[tuple[int, int]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(starts, f)


def load_fixed_eval_starts(path: str | Path) -> list[tuple[int, int]]:
    with open(path) as f:
        return [tuple(pair) for pair in json.load(f)]


def _bootstrap_ci(
    values: list[float], n_bootstrap: int = 10_000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap ``1-alpha`` CI for the mean."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.array(values)
    rng = np.random.default_rng(0)
    means = np.array(
        [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_bootstrap)]
    )
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def evaluate_agent(
    agent,
    env: OptimalExecutionEnv,
    n_episodes: int = 100,
    seed: int = 42,
    *,
    bench_params: Optional[dict[str, Any]] = None,
    fixed_starts: Optional[list[tuple[int, int]]] = None,
) -> dict[str, Any]:
    """Evaluate policy on ``env``.

    When ``bench_params`` is set (same keys as ``compare_all`` / ``twap_execution``), computes
    TWAP and VWAP implementation shortfall on the **identical** ``(start, T)`` window as each RL
    episode so ``mean_rl_minus_twap_bps`` and ``pct_beat_twap_is`` are apples-to-apples.

    When ``fixed_starts`` is provided, each episode resets with the pre-sampled
    ``(row_start, reset_seed)`` pair for deterministic evaluation.
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

    regime_diff_t: dict[int, list[float]] = {}

    effective_n = len(fixed_starts) if fixed_starts else n_episodes

    for ep_idx in range(effective_n):
        if fixed_starts is not None:
            row_start, ep_seed = fixed_starts[ep_idx]
            obs, _ = env.reset(seed=ep_seed)
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
        else:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

        arrival = env._arrival  # noqa: SLF001
        start = int(env._row_start)  # noqa: SLF001
        total_cost = 0.0
        ep_ret = 0.0
        terminated = False
        physical = bool(getattr(env, "_physical", False))
        ep_regime = int(env.price_data.iloc[start].get("regime", 0))
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
                regime_diff_t.setdefault(ep_regime, []).append(d)
            if np.isfinite(vb):
                valid_vwap += 1
                diff_v.append(is_bps_f - vb)
                if is_bps_f > vb:
                    beat_v += 1

    out: dict[str, Any] = {
        "mean_is_bps": float(np.mean(is_list)) if is_list else 0.0,
        "std_is_bps": float(np.std(is_list)) if is_list else 0.0,
        "completion_rate": completed / max(effective_n, 1),
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

        if diff_t:
            std_t = float(np.std(diff_t))
            out["is_gap_sharpe"] = float(np.mean(diff_t)) / std_t if std_t > 1e-12 else 0.0
            lo, hi = _bootstrap_ci(diff_t)
            out["ci95_rl_minus_twap_bps"] = (lo, hi)
        else:
            out["is_gap_sharpe"] = float("nan")
            out["ci95_rl_minus_twap_bps"] = (float("nan"), float("nan"))

        for regime, diffs in sorted(regime_diff_t.items()):
            r_key = f"mean_rl_minus_twap_bps_regime{regime}"
            out[r_key] = float(np.mean(diffs))
            out[f"n_episodes_regime{regime}"] = len(diffs)

    return out
