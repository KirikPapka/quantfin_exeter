#!/usr/bin/env python3
"""Regime → env → eval / optional PPO train."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks import compare_all
from src.data_pipeline import load_split
from src.regime_detector import RegimeDetector, regime_sanity_summary
from src.rl_agent import evaluate_agent, format_rl_eval_report, train_agent
from src.trading_env import OptimalExecutionEnv, physical_institutional_kwargs
from src.utils import set_global_seed, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--train", action="store_true")
    ap.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="PPO/SAC gradient steps (use 300k+ for meaningful IS vs benchmarks).",
    )
    ap.add_argument("--algo", default="PPO", choices=("PPO", "SAC"))
    ap.add_argument("--no-bbo", action="store_true", help="Skip BBO merge even if parquet exists")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--order-notional-usd",
        type=float,
        default=0.0,
        help="USD notional to liquidate; 0 = legacy normalized units (no participation model).",
    )
    ap.add_argument(
        "--order-start-bar",
        type=int,
        default=0,
        help="First bar within horizon (0..T-1) when trading starts; prior close is arrival benchmark.",
    )
    ap.add_argument(
        "--lam-risk",
        type=float,
        default=None,
        help="Inventory risk weight in env (default: env default 0.22).",
    )
    ap.add_argument(
        "--no-per-step-cap",
        action="store_true",
        help="With USD notional, disable max fraction of inventory per bar (default cap 0.33).",
    )
    ap.add_argument(
        "--max-inventory-frac-per-step",
        type=float,
        default=None,
        help="Cap fraction of X_0 per step when physical (default 0.33 unless --no-per-step-cap).",
    )
    ap.add_argument(
        "--is-reward-scale",
        type=float,
        default=None,
        help="Scale physical IS term in reward (default 1.28 when notional>0, else 1.0 via env).",
    )
    ap.add_argument(
        "--twap-slice-bonus",
        type=float,
        default=None,
        help="TWAP same-bar reference shaping coef (default 0.30 when notional>0; 0 disables).",
    )
    ap.add_argument(
        "--no-news",
        action="store_true",
        help="Do not merge Finnhub news parquet (see scripts/fetch_finnhub_news.py).",
    )
    args = ap.parse_args()

    setup_logging()
    set_global_seed(args.seed)

    train_df = load_split(
        "train",
        ticker=args.ticker,
        use_bbo=not args.no_bbo,
        use_news=not args.no_news,
    )
    test_df = load_split(
        "test",
        ticker=args.ticker,
        use_bbo=not args.no_bbo,
        use_news=not args.no_news,
    )

    det = RegimeDetector(n_components=2, fallback_threshold=0.24)
    det.fit(train_df)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["regime"] = det.predict(train_df)
    test_df["regime"] = det.predict(test_df)
    logger.info("Regime sanity:\n%s", regime_sanity_summary(train_df, train_df["regime"].to_numpy()))

    on = args.order_notional_usd if args.order_notional_usd > 0 else None
    env_kw = dict(
        T=10,
        seed=args.seed,
        order_notional_usd=on,
        order_start_bar=max(0, args.order_start_bar),
    )
    if args.lam_risk is not None:
        env_kw["lam"] = float(args.lam_risk)
    if on is not None:
        env_kw.update(
            physical_institutional_kwargs(
                float(args.order_notional_usd),
                no_per_step_cap=bool(args.no_per_step_cap),
                max_inventory_fraction_per_step=args.max_inventory_frac_per_step,
                is_reward_scale=args.is_reward_scale,
                twap_slice_bonus_coef=args.twap_slice_bonus,
            )
        )
    else:
        if args.is_reward_scale is not None:
            env_kw["is_reward_scale"] = float(args.is_reward_scale)
        if args.twap_slice_bonus is not None:
            env_kw["twap_slice_bonus_coef"] = float(args.twap_slice_bonus)
        if args.max_inventory_frac_per_step is not None:
            env_kw["max_inventory_fraction_per_step"] = float(
                args.max_inventory_frac_per_step
            )
        elif args.no_per_step_cap:
            env_kw["max_inventory_fraction_per_step"] = None
    env = OptimalExecutionEnv(test_df, **env_kw)
    from gymnasium.utils.env_checker import check_env

    check_env(env)

    bench_params = {
        "T": 10,
        "X_0": 1.0,
        "eta": 0.01,
        "gamma": 0.001,
        "lam": 0.5,
        "order_notional_usd": float(args.order_notional_usd),
        "order_start_bar": int(args.order_start_bar),
    }

    if args.train:
        model = train_agent(
            OptimalExecutionEnv(train_df, **env_kw),
            algorithm=args.algo,
            total_timesteps=args.timesteps,
            save_path=str(ROOT / "models"),
            log_path=str(ROOT / "logs"),
            seed=args.seed,
        )
        rl_stats = evaluate_agent(
            model, env, n_episodes=50, seed=args.seed, bench_params=bench_params
        )
    else:
        rng = __import__("numpy").random.default_rng(args.seed)

        class R:
            def predict(self, obs, deterministic=False):  # noqa: ANN001
                return rng.uniform(0.0, 1.0, size=(1,)), None

        rl_stats = evaluate_agent(
            R(), env, n_episodes=50, seed=args.seed, bench_params=bench_params
        )

    logger.info("RL evaluation (path-aligned vs same windows):\n%s", format_rl_eval_report(rl_stats))
    logger.info(
        "Benchmarks:\n%s",
        compare_all(
            rl_stats,
            test_df,
            params={**bench_params, "seed": args.seed},
        ).to_string(index=False),
    )


if __name__ == "__main__":
    main()
