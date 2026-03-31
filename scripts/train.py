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
from src.rl_agent import evaluate_agent, train_agent
from src.trading_env import OptimalExecutionEnv
from src.utils import set_global_seed, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--timesteps", type=int, default=20_000)
    ap.add_argument("--algo", default="PPO", choices=("PPO", "SAC"))
    ap.add_argument("--no-bbo", action="store_true", help="Skip BBO merge even if parquet exists")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging()
    set_global_seed(args.seed)

    train_df = load_split("train", ticker=args.ticker, use_bbo=not args.no_bbo)
    test_df = load_split("test", ticker=args.ticker, use_bbo=not args.no_bbo)

    det = RegimeDetector(n_components=2, fallback_threshold=0.24)
    det.fit(train_df)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["regime"] = det.predict(train_df)
    test_df["regime"] = det.predict(test_df)
    logger.info("Regime sanity:\n%s", regime_sanity_summary(train_df, train_df["regime"].to_numpy()))

    env = OptimalExecutionEnv(test_df, T=10, seed=args.seed)
    from gymnasium.utils.env_checker import check_env

    check_env(env)

    if args.train:
        model = train_agent(
            OptimalExecutionEnv(train_df, T=10, seed=args.seed),
            algorithm=args.algo,
            total_timesteps=args.timesteps,
            save_path=str(ROOT / "models"),
            log_path=str(ROOT / "logs"),
            seed=args.seed,
        )
        rl_stats = evaluate_agent(model, env, n_episodes=50, seed=args.seed)
    else:
        rng = __import__("numpy").random.default_rng(args.seed)

        class R:
            def predict(self, obs, deterministic=False):  # noqa: ANN001
                return rng.uniform(0.0, 1.0, size=(1,)), None

        rl_stats = evaluate_agent(R(), env, n_episodes=50, seed=args.seed)

    logger.info(
        "Benchmarks:\n%s",
        compare_all(
            rl_stats,
            test_df,
            params={"T": 10, "X_0": 1.0, "eta": 0.01, "gamma": 0.001, "lam": 0.5, "seed": args.seed},
        ).to_string(index=False),
    )


if __name__ == "__main__":
    main()
