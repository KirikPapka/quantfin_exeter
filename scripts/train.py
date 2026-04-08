#!/usr/bin/env python3
"""Regime → env → eval / optional PPO train."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPO_CHECKPOINT = ROOT / "models" / "best_ppo_twap_gap.zip"
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
from src.regime_switching import TWAPFallbackPolicy, build_regime_switching_policy
from src.scenario_paths import append_synthetic_scenarios
from src.trend_classifier import compute_trend_regime
from src.trading_env import OptimalExecutionEnv, physical_institutional_kwargs
from src.utils import set_global_seed, setup_logging

logger = logging.getLogger(__name__)


def _load_sb3_model(path: Path, algo: str) -> Any:
    from stable_baselines3 import PPO, SAC

    if not path.is_file():
        raise FileNotFoundError(path)
    last_err: Exception | None = None
    for cls in (PPO, SAC) if algo.upper() == "PPO" else (SAC, PPO):
        try:
            return cls.load(str(path))
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not load {path}: {last_err}")


def _compare_all_params(
    bench_params: dict[str, Any],
    fixed_starts: list[tuple[int, int]] | None,
    seed: int,
) -> dict[str, Any]:
    """Align printed benchmark table with ``evaluate_agent`` / web case study (``fixed_row_starts``)."""
    p = {**bench_params, "seed": int(seed)}
    if fixed_starts:
        row_starts = [int(r) for r, _ in fixed_starts]
        p["fixed_row_starts"] = row_starts
        p["n_starts"] = len(row_starts)
    return p


def _build_final_eval_agent(
    primary: Any,
    env: OptimalExecutionEnv,
    args: Any,
    specialist_model: Any | None,
) -> Any:
    """Apply optional trend-based regime switching around ``primary`` (uptrend slot)."""
    downtrend: Any = TWAPFallbackPolicy(env)
    if specialist_model is not None:
        downtrend = specialist_model
    elif str(args.downtrend_model).strip():
        p = Path(args.downtrend_model)
        if p.is_file():
            downtrend = _load_sb3_model(p, args.algo)
        else:
            logger.warning("downtrend-model not found: %s — using TWAP fallback", p)
    if args.regime_switch:
        return build_regime_switching_policy(
            primary,
            downtrend,
            env,
            midtrend_strategy=str(args.midtrend_strategy),
            lookback=int(args.trend_lookback),
            up_pct=float(args.trend_up_pct),
            down_pct=float(args.trend_down_pct),
        )
    return primary


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
        help="TWAP-residual action: a=0.5 is TWAP, deviation bounded by this. "
        "When --order-notional-usd > 0, default 0.15 (same as web case study).",
    )
    ap.add_argument(
        "--relative-is-scale",
        type=float,
        default=None,
        help="Relative-IS reward scale vs TWAP same-bar. "
        "When --order-notional-usd > 0, default 2.0 (same as web case study).",
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
    ap.add_argument(
        "--regime-switch",
        action="store_true",
        help="Wrap policy in trend-based RegimeSwitchingPolicy at eval (uptrend=main/load, mid/down configurable).",
    )
    ap.add_argument(
        "--downtrend-model",
        type=str,
        default="",
        help="Path to PPO/SAC .zip for downtrend slot when not using a freshly trained down specialist.",
    )
    ap.add_argument(
        "--midtrend-strategy",
        choices=("twap", "downtrend_model"),
        default="twap",
        help="Mid-trend slot: TWAP fallback or same policy as downtrend slot.",
    )
    ap.add_argument("--trend-lookback", type=int, default=20)
    ap.add_argument("--trend-up-pct", type=float, default=0.02)
    ap.add_argument("--trend-down-pct", type=float, default=-0.02)
    ap.add_argument(
        "--train-downtrend-specialist",
        action="store_true",
        help="Train a separate PPO on synthetic down+flat paths (9-D obs), save models/down_specialist.zip.",
    )
    ap.add_argument(
        "--downtrend-specialist-timesteps",
        type=int,
        default=200_000,
        help="Timesteps for --train-downtrend-specialist.",
    )
    ap.add_argument(
        "--downtrend-synthetic-bars",
        type=int,
        default=120,
        help="Bars per synthetic strip for down+flat specialist training data.",
    )
    ap.add_argument(
        "--load-model",
        type=str,
        default="",
        help="Load this PPO/SAC .zip for eval-only or as uptrend policy with --regime-switch.",
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
    train_before_synth = train_df.copy()
    try:
        train_df = append_synthetic_scenarios(
            train_df, str(args.append_synthetic), int(args.synthetic_bars)
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    logger.info("Train rows after optional synthetic append: %s", len(train_df))
    logger.info("Regime sanity:\n%s", regime_sanity_summary(train_df, train_df["regime"].to_numpy()))

    # Match web case study / deployment (web/precompute.py ORDER_NOTIONAL + RESIDUAL_BOUND + RELATIVE_IS_SCALE).
    if float(args.order_notional_usd) > 0:
        if args.residual_bound is None:
            args.residual_bound = 0.15
            logger.info("Default --residual-bound 0.15 (case-study / deployment alignment).")
        if args.relative_is_scale is None:
            args.relative_is_scale = 2.0
            logger.info("Default --relative-is-scale 2.0 (case-study / deployment alignment).")

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

    test_df = compute_trend_regime(
        test_df,
        lookback=int(args.trend_lookback),
        up_pct=float(args.trend_up_pct),
        down_pct=float(args.trend_down_pct),
    )
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
    default_starts_path = ROOT / "models" / "fixed_eval_starts.json"
    if args.fixed_eval_starts and Path(args.fixed_eval_starts).exists():
        fixed_starts = load_fixed_eval_starts(Path(args.fixed_eval_starts))
        logger.info("Loaded %d fixed eval starts from %s", len(fixed_starts), args.fixed_eval_starts)
    elif default_starts_path.is_file():
        fixed_starts = load_fixed_eval_starts(default_starts_path)
        logger.info(
            "Loaded %d fixed eval starts from %s (reuse for stable benchmarks vs case study)",
            len(fixed_starts),
            default_starts_path,
        )
    else:
        fixed_starts = generate_fixed_eval_starts(env, n=int(args.n_fixed_eval), seed=args.seed)
        save_fixed_eval_starts(fixed_starts, default_starts_path)
        logger.info("Generated %d fixed eval starts → %s", len(fixed_starts), default_starts_path)

    T_horizon = int(env_kw["T"])
    max_start = len(test_df) - T_horizon - 1
    if fixed_starts and max_start >= 0:
        n_before = len(fixed_starts)
        fixed_starts = [(int(r), int(s)) for r, s in fixed_starts if 0 <= int(r) <= max_start]
        dropped = n_before - len(fixed_starts)
        if dropped:
            logger.warning(
                "Dropped %d fixed eval starts outside valid range [0, %d] for this panel",
                dropped,
                max_start,
            )
        if not fixed_starts:
            logger.warning("No valid fixed starts after filter; regenerating.")
            fixed_starts = generate_fixed_eval_starts(env, n=int(args.n_fixed_eval), seed=args.seed)
            save_fixed_eval_starts(fixed_starts, default_starts_path)
            logger.info("Regenerated %d fixed eval starts → %s", len(fixed_starts), default_starts_path)

    eval_trend_kw = dict(
        trend_lookback=int(args.trend_lookback),
        trend_up_pct=float(args.trend_up_pct),
        trend_down_pct=float(args.trend_down_pct),
    )

    specialist_model = None
    if args.train_downtrend_specialist and args.cql:
        logger.warning("Ignoring --train-downtrend-specialist when --cql is set.")
    if args.train_downtrend_specialist and not args.cql:
        down_train = append_synthetic_scenarios(
            train_before_synth.copy(), "down,flat", int(args.downtrend_synthetic_bars)
        )
        spec_kw = {
            **env_kw,
            "lam": 0.5,
            "residual_bound": None,
            "relative_is_scale": 0.0,
        }
        if on is not None:
            spec_kw.update(
                physical_institutional_kwargs(
                    float(args.order_notional_usd),
                    no_per_step_cap=bool(args.no_per_step_cap),
                    max_inventory_fraction_per_step=args.max_inventory_frac_per_step,
                    is_reward_scale=args.is_reward_scale,
                    twap_slice_bonus_coef=args.twap_slice_bonus,
                    terminal_inventory_penalty=args.terminal_penalty,
                    lam=0.5,
                    residual_bound=None,
                    relative_is_scale=0.0,
                )
            )

        def _make_spec_env(rank: int) -> OptimalExecutionEnv:
            return OptimalExecutionEnv(
                down_train,
                **{**spec_kw, "seed": int(args.seed) + rank * 10_007 + 333_333},
            )

        specialist_model = train_agent(
            _make_spec_env,
            algorithm="PPO",
            total_timesteps=int(args.downtrend_specialist_timesteps),
            n_envs=1,
            save_path=str(ROOT / "models"),
            log_path=str(ROOT / "logs"),
            seed=int(args.seed) + 4242,
            eval_env=None,
            bench_params=None,
            eval_freq_timesteps=0,
            lr_schedule=args.lr_schedule,
            ent_coef=0.01,
        )
        specialist_path = ROOT / "models" / "down_specialist.zip"
        specialist_model.save(str(specialist_path))
        logger.info("Saved downtrend specialist model → %s", specialist_path)

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
            cql_agent,
            env,
            n_episodes=50,
            seed=args.seed,
            bench_params=bench_params,
            fixed_starts=fixed_starts,
            **eval_trend_kw,
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
            val_df = compute_trend_regime(
                val_df,
                lookback=int(args.trend_lookback),
                up_pct=float(args.trend_up_pct),
                down_pct=float(args.trend_down_pct),
            )
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
            primary = train_ensemble(
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
        else:
            primary = model
        final_agent = _build_final_eval_agent(primary, env, args, specialist_model)
        if args.regime_switch and hasattr(final_agent, "policy_selections"):
            logger.info("Regime switch selections: %s", final_agent.policy_selections)
        rl_stats = evaluate_agent(
            final_agent,
            env,
            n_episodes=50,
            seed=args.seed,
            bench_params=bench_params,
            fixed_starts=fixed_starts,
            **eval_trend_kw,
        )
    else:
        if args.ensemble_dir and Path(args.ensemble_dir).is_dir():
            models = load_ensemble_from_dir(args.ensemble_dir, algo=args.algo)
            primary: Any = EnsemblePolicy(models)
        elif str(args.load_model).strip() and Path(args.load_model).is_file():
            primary = _load_sb3_model(Path(args.load_model), args.algo)
        elif DEFAULT_PPO_CHECKPOINT.is_file():
            logger.info(
                "Eval-only: loading default checkpoint %s (omit --load-model to use another path)",
                DEFAULT_PPO_CHECKPOINT,
            )
            primary = _load_sb3_model(DEFAULT_PPO_CHECKPOINT, args.algo)
        elif args.regime_switch:
            primary = TWAPFallbackPolicy(env)
        else:
            rng = __import__("numpy").random.default_rng(args.seed)

            class R:
                def predict(self, obs, deterministic=False):  # noqa: ANN001
                    return rng.uniform(0.0, 1.0, size=(1,)), None

            primary = R()
            logger.warning(
                "No PPO checkpoint found at %s and no --load-model — using random policy for RL row",
                DEFAULT_PPO_CHECKPOINT,
            )
        final_agent = _build_final_eval_agent(primary, env, args, specialist_model)
        if args.regime_switch and hasattr(final_agent, "policy_selections"):
            logger.info("Regime switch selections: %s", final_agent.policy_selections)
        rl_stats = evaluate_agent(
            final_agent,
            env,
            n_episodes=50,
            seed=args.seed,
            bench_params=bench_params,
            fixed_starts=fixed_starts,
            **eval_trend_kw,
        )

    logger.info("RL evaluation (path-aligned vs same windows):\n%s", format_rl_eval_report(rl_stats))
    logger.info(
        "Benchmarks:\n%s",
        compare_all(
            rl_stats,
            test_df,
            params=_compare_all_params(bench_params, fixed_starts, args.seed),
        ).to_string(index=False),
    )


if __name__ == "__main__":
    main()
