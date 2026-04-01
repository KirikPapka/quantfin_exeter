#!/usr/bin/env python3
"""Run RL + classical benchmarks on synthetic flat / up / down price paths."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.benchmarks import compare_all
from src.rl_agent import evaluate_agent, format_rl_eval_report
from src.scenario_paths import synthetic_panel
from src.trading_env import OptimalExecutionEnv, physical_institutional_kwargs
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def _bench_params(
    T: int,
    notional: float,
    t0: int,
    seed: int,
    n_starts: int,
) -> dict:
    return {
        "T": T,
        "X_0": 1.0,
        "eta": 0.01,
        "gamma": 0.001,
        "lam": 0.5,
        "order_notional_usd": float(notional),
        "order_start_bar": int(t0),
        "seed": int(seed),
        "n_starts": int(n_starts),
    }


def _eval_params(bp: dict) -> dict:
    return {k: v for k, v in bp.items() if k != "n_starts"}


class _RandomAgent:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=False):  # noqa: ANN001
        return self._rng.uniform(0.0, 1.0, size=(1,)), None


def _load_trained_policy(path: str):
    """Load a Stable-Baselines3 PPO or SAC zip from ``path``."""
    from stable_baselines3 import PPO, SAC

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Model file not found: {p}")
    last_err: Exception | None = None
    for cls in (PPO, SAC):
        try:
            return cls.load(str(p))
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise SystemExit(f"Could not load {p} as PPO or SAC: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--bars", type=int, default=80, help="Length of synthetic series.")
    ap.add_argument("--step", type=float, default=0.2, help="Daily close step for up/down paths.")
    ap.add_argument("--order-notional-usd", type=float, default=0.0)
    ap.add_argument("--order-start-bar", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-starts", type=int, default=40, help="Random windows for classical benchmarks.")
    ap.add_argument("--n-episodes", type=int, default=40, help="RL eval episodes per scenario.")
    ap.add_argument("--no-rl", action="store_true", help="Classical benchmarks only (faster).")
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PPO or SAC .zip (e.g. models/PPO_*.zip). Must match env: "
        "same T, order-notional-usd, order-start-bar, institutional kwargs, and 9-dim obs.",
    )
    ap.add_argument(
        "--no-per-step-cap",
        action="store_true",
        help="With USD notional, match train.py --no-per-step-cap.",
    )
    ap.add_argument(
        "--max-inventory-frac-per-step",
        type=float,
        default=None,
        help="Override per-step inventory cap (default 0.33 when notional>0).",
    )
    ap.add_argument(
        "--is-reward-scale",
        type=float,
        default=None,
        help="Override IS reward scale (default 1.28 when notional>0).",
    )
    ap.add_argument(
        "--twap-slice-bonus",
        type=float,
        default=None,
        help="Override TWAP slice bonus coef (default 0.60 when notional>0).",
    )
    ap.add_argument(
        "--residual-bound",
        type=float,
        default=None,
        help="TWAP-residual action: a=0.5 is TWAP, deviation bounded (must match training).",
    )
    ap.add_argument(
        "--relative-is-scale",
        type=float,
        default=None,
        help="Relative-IS reward scale (must match training; 0 = disabled).",
    )
    ap.add_argument(
        "--terminal-penalty",
        type=float,
        default=None,
        help="Terminal inventory penalty (must match training).",
    )
    ap.add_argument(
        "--lam-risk",
        type=float,
        default=None,
        help="Inventory risk weight (must match training).",
    )
    args = ap.parse_args()

    setup_logging()
    on = args.order_notional_usd if args.order_notional_usd > 0 else None
    env_kw = dict(
        T=int(args.T),
        seed=int(args.seed),
        resample=True,
        order_notional_usd=on,
        order_start_bar=max(0, args.order_start_bar),
    )
    if on is not None:
        env_kw.update(
            physical_institutional_kwargs(
                float(args.order_notional_usd),
                no_per_step_cap=bool(args.no_per_step_cap),
                max_inventory_fraction_per_step=args.max_inventory_frac_per_step,
                is_reward_scale=args.is_reward_scale,
                twap_slice_bonus_coef=args.twap_slice_bonus,
                terminal_inventory_penalty=args.terminal_penalty,
                lam=args.lam_risk,
                residual_bound=args.residual_bound,
                relative_is_scale=args.relative_is_scale,
            )
        )
    if args.residual_bound is not None and on is None:
        env_kw["residual_bound"] = float(args.residual_bound)
    if args.relative_is_scale is not None and on is None:
        env_kw["relative_is_scale"] = float(args.relative_is_scale)

    for kind in ("flat", "up", "down"):
        df = synthetic_panel(kind, n=int(args.bars), step=float(args.step))
        bp = _bench_params(
            int(args.T),
            float(args.order_notional_usd),
            int(args.order_start_bar),
            int(args.seed),
            min(int(args.n_starts), len(df) - int(args.T)),
        )
        max_start = len(df) - int(args.T) - 1
        if max_start < 0:
            raise SystemExit("bars too short for T")

        logger.info("======== Scenario: %s (Close %s) ========", kind.upper(), kind)
        if args.no_rl:
            rl_stats = {
                "mean_is_bps": float("nan"),
                "std_is_bps": float("nan"),
                "completion_rate": float("nan"),
            }
        else:
            env = OptimalExecutionEnv(df, **env_kw)
            if args.model:
                agent = _load_trained_policy(args.model)
                policy_label = f"trained ({Path(args.model).name})"
            else:
                agent = _RandomAgent(int(args.seed))
                policy_label = "random policy"
            rl_stats = evaluate_agent(
                agent,
                env,
                n_episodes=int(args.n_episodes),
                seed=int(args.seed),
                bench_params=_eval_params(bp),
            )
            logger.info("RL (%s) path-aligned:\n%s", policy_label, format_rl_eval_report(rl_stats))

        table = compare_all(rl_stats, df, params=bp)
        logger.info("Benchmark table:\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
