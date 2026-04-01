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
from src.bc_warmstart import bc_warmstart_state_dict
from src.ensemble import EnsemblePolicy, load_ensemble_from_dir, train_ensemble
from src.rl_agent import (
    evaluate_agent,
    format_rl_eval_report,
    generate_fixed_eval_starts,
    load_fixed_eval_starts,
    save_fixed_eval_starts,
    train_agent,
)
from src.scenario_paths import append_synthetic_scenarios
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
    ap.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Parallel Gym envs (DummyVecEnv) for more diverse rollouts per PPO/SAC update.",
    )
    ap.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Timesteps between val evals; 0 disables best-TWAP-gap checkpoint (train only).",
    )
    ap.add_argument(
        "--n-eval-episodes",
        type=int,
        default=24,
        help="Episodes per eval when saving best mean_RL_minus_TWAP_bps.",
    )
    ap.add_argument(
        "--eval-is-coef",
        type=float,
        default=0.0,
        help="Terminal bonus scale matching evaluate_agent IS (0=off; try 0.02–0.08).",
    )
    ap.add_argument(
        "--append-synthetic",
        type=str,
        default="",
        help="Comma-separated scenario kinds appended to train panel: flat,up,down.",
    )
    ap.add_argument(
        "--synthetic-bars",
        type=int,
        default=120,
        help="Length of each synthetic strip when using --append-synthetic.",
    )
    ap.add_argument(
        "--lr-schedule",
        choices=("linear", "constant"),
        default="linear",
        help="LR schedule: linear decays 3e-4->5e-5; constant uses 2.5e-4.",
    )
    ap.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help="PPO entropy coefficient (default 0.003).",
    )
    ap.add_argument(
        "--terminal-penalty",
        type=float,
        default=None,
        help="Terminal inventory penalty (default 5.0 for physical, 12.0 legacy).",
    )
    ap.add_argument(
        "--fixed-eval-starts",
        type=str,
        default="",
        help="Path to JSON with fixed (row_start, seed) eval pairs; empty = generate fresh.",
    )
    ap.add_argument(
        "--n-fixed-eval",
        type=int,
        default=200,
        help="Number of fixed eval windows to generate if --fixed-eval-starts is empty.",
    )
    ap.add_argument(
        "--bc-warmstart",
        action="store_true",
        help="Pre-train policy with behavioral cloning from TWAP before RL fine-tuning.",
    )
    ap.add_argument(
        "--bc-episodes",
        type=int,
        default=500,
        help="Number of TWAP demo episodes for BC warmstart.",
    )
    ap.add_argument(
        "--bc-epochs",
        type=int,
        default=20,
        help="BC supervised training epochs.",
    )
    ap.add_argument(
        "--residual-bound",
        type=float,
        default=None,
        help="Enable TWAP-residual action: a=0.5 is TWAP, deviation bounded by this (e.g. 0.15).",
    )
    ap.add_argument(
        "--relative-is-scale",
        type=float,
        default=None,
        help="Scale for relative-IS reward (vs TWAP same-bar); 0 = disabled, 2.0 typical.",
    )
    ap.add_argument(
        "--ensemble",
        type=int,
        default=0,
        help="Train N models with different seeds and ensemble via median action (0=off).",
    )
    ap.add_argument(
        "--ensemble-dir",
        type=str,
        default="",
        help="Load pre-trained ensemble from this dir (all .zip files) instead of training.",
    )
    ap.add_argument(
        "--cql",
        action="store_true",
        help="Train offline CQL instead of online PPO/SAC (requires d3rlpy).",
    )
    ap.add_argument(
        "--cql-episodes",
        type=int,
        default=2000,
        help="Number of rollout episodes for CQL offline dataset.",
    )
    ap.add_argument(
        "--cql-steps",
        type=int,
        default=50_000,
        help="CQL gradient steps.",
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
    try:
        train_df = append_synthetic_scenarios(
            train_df, str(args.append_synthetic), int(args.synthetic_bars)
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    logger.info("Train rows after optional synthetic append: %s", len(train_df))
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
                terminal_inventory_penalty=args.terminal_penalty,
                lam=args.lam_risk,
                residual_bound=args.residual_bound,
                relative_is_scale=args.relative_is_scale,
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
    if float(args.eval_is_coef) > 0.0:
        env_kw["eval_is_reward_coef"] = float(args.eval_is_coef)
    if args.terminal_penalty is not None and on is None:
        env_kw["terminal_inventory_penalty"] = float(args.terminal_penalty)
    if args.residual_bound is not None and on is None:
        env_kw["residual_bound"] = float(args.residual_bound)
    if args.relative_is_scale is not None and on is None:
        env_kw["relative_is_scale"] = float(args.relative_is_scale)
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

    fixed_starts = None
    if args.fixed_eval_starts and Path(args.fixed_eval_starts).exists():
        fixed_starts = load_fixed_eval_starts(args.fixed_eval_starts)
        logger.info("Loaded %d fixed eval starts from %s", len(fixed_starts), args.fixed_eval_starts)
    else:
        fixed_starts = generate_fixed_eval_starts(env, n=int(args.n_fixed_eval), seed=args.seed)
        starts_path = ROOT / "models" / "fixed_eval_starts.json"
        save_fixed_eval_starts(fixed_starts, starts_path)
        logger.info("Generated %d fixed eval starts → %s", len(fixed_starts), starts_path)

    if args.cql:
        from src.offline_cql import train_cql
        cql_env = OptimalExecutionEnv(train_df, **{**env_kw, "seed": args.seed + 77_777})
        cql_agent = train_cql(
            cql_env,
            n_episodes=args.cql_episodes,
            seed=args.seed,
            n_steps=args.cql_steps,
            save_path=str(ROOT / "models" / "cql_model.d3"),
        )
        rl_stats = evaluate_agent(
            cql_agent, env, n_episodes=50, seed=args.seed,
            bench_params=bench_params, fixed_starts=fixed_starts,
        )
    elif args.train:

        def _make_train_env(rank: int) -> OptimalExecutionEnv:
            return OptimalExecutionEnv(
                train_df, **{**env_kw, "seed": int(args.seed) + rank * 10_007}
            )

        try:
            val_df = load_split(
                "val",
                ticker=args.ticker,
                use_bbo=not args.no_bbo,
                use_news=not args.no_news,
            )
            val_df = val_df.copy()
            val_df["regime"] = det.predict(val_df)
            eval_df = val_df
            logger.info("Using val split for eval callback (%s rows)", len(eval_df))
        except FileNotFoundError:
            eval_df = test_df
            logger.info("No val split; eval callback uses test (%s rows)", len(eval_df))

        eval_callback_env = OptimalExecutionEnv(eval_df, **env_kw)
        eval_freq = int(args.eval_freq) if int(args.eval_freq) > 0 else 0

        bc_sd = None
        if args.bc_warmstart:
            bc_env = OptimalExecutionEnv(train_df, **{**env_kw, "seed": args.seed + 99_999})
            bc_sd = bc_warmstart_state_dict(
                bc_env,
                n_episodes=args.bc_episodes,
                seed=args.seed,
                epochs=args.bc_epochs,
            )

        model = train_agent(
            _make_train_env,
            algorithm=args.algo,
            total_timesteps=args.timesteps,
            n_envs=int(args.n_envs),
            save_path=str(ROOT / "models"),
            log_path=str(ROOT / "logs"),
            seed=args.seed,
            eval_env=eval_callback_env if eval_freq > 0 else None,
            bench_params=bench_params if eval_freq > 0 else None,
            eval_freq_timesteps=eval_freq,
            n_eval_episodes=int(args.n_eval_episodes),
            best_model_filename=f"best_{args.algo.lower()}_twap_gap.zip",
            lr_schedule=args.lr_schedule,
            ent_coef=args.ent_coef,
            bc_state_dict=bc_sd,
        )
        if args.ensemble > 1:
            logger.info("Training %d-seed ensemble...", args.ensemble)
            ens = train_ensemble(
                _make_train_env,
                n_seeds=args.ensemble,
                base_seed=args.seed,
                algo=args.algo,
                total_timesteps=args.timesteps,
                save_dir=str(ROOT / "models" / "ensemble"),
                n_envs=int(args.n_envs),
                log_path=str(ROOT / "logs"),
                lr_schedule=args.lr_schedule,
                ent_coef=args.ent_coef,
                bc_state_dict=bc_sd,
            )
            rl_stats = evaluate_agent(
                ens, env, n_episodes=50, seed=args.seed,
                bench_params=bench_params, fixed_starts=fixed_starts,
            )
        else:
            rl_stats = evaluate_agent(
                model, env, n_episodes=50, seed=args.seed,
                bench_params=bench_params, fixed_starts=fixed_starts,
            )
    else:
        if args.ensemble_dir and Path(args.ensemble_dir).is_dir():
            models = load_ensemble_from_dir(args.ensemble_dir, algo=args.algo)
            agent = EnsemblePolicy(models)
        else:
            rng = __import__("numpy").random.default_rng(args.seed)

            class R:
                def predict(self, obs, deterministic=False):  # noqa: ANN001
                    return rng.uniform(0.0, 1.0, size=(1,)), None

            agent = R()

        rl_stats = evaluate_agent(
            agent, env, n_episodes=50, seed=args.seed,
            bench_params=bench_params, fixed_starts=fixed_starts,
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
