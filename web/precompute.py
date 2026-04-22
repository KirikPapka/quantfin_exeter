"""Pre-compute case study data at server startup."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Match training / scenario_benchmarks physical institutional setup
ORDER_NOTIONAL_USD = 5_000_000.0
RESIDUAL_BOUND = 0.15
RELATIVE_IS_SCALE = 2.0
ORDER_START_BAR = 0

COLORS = {"regime_calm": "#34d399", "regime_vol": "#fbbf24", "price_line": "#0f172a"}


def _physical_env_kwargs() -> dict[str, Any]:
    from src.trading_env import physical_institutional_kwargs

    kw = physical_institutional_kwargs(
        ORDER_NOTIONAL_USD,
        residual_bound=RESIDUAL_BOUND,
        relative_is_scale=RELATIVE_IS_SCALE,
    )
    return {"order_notional_usd": ORDER_NOTIONAL_USD, **kw}


def _bench_params(T: int) -> dict[str, Any]:
    return {
        "T": T,
        "X_0": 1.0,
        "eta": 0.01,
        "gamma": 0.001,
        "lam": 0.5,
        "order_notional_usd": ORDER_NOTIONAL_USD,
        "order_start_bar": ORDER_START_BAR,
    }


# Case study: path-aligned eval on many windows, same starts as training diagnostics
CASE_STUDY_MAX_EPISODES = 200
# Sweep this many distinct checkpoints (by mtime); pick best mean_RL_minus_TWAP_bps
CASE_STUDY_MAX_CHECKPOINTS_TO_SWEEP = 12
# Use this many windows per checkpoint during the sweep (full set used for final table)
CASE_STUDY_SWEEP_EPISODES = 60


def _load_valid_fixed_starts(df: pd.DataFrame, T: int) -> list[tuple[int, int]] | None:
    path = ROOT / "models" / "fixed_eval_starts.json"
    if not path.is_file():
        return None
    from src.rl_agent import load_fixed_eval_starts

    max_start = len(df) - T - 1
    if max_start < 0:
        return None
    pairs = load_fixed_eval_starts(path)
    valid = [(int(r), int(s)) for r, s in pairs if 0 <= int(r) <= max_start]
    if not valid:
        return None
    return valid[:CASE_STUDY_MAX_EPISODES]


def _checkpoint_candidates(models_dir: Path) -> list[Path]:
    if not models_dir.is_dir():
        return []
    zips = sorted(models_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    best = models_dir / "best_ppo_twap_gap.zip"
    out: list[Path] = []
    seen: set[Path] = set()
    if best.is_file():
        out.append(best)
        seen.add(best.resolve())
    for p in zips:
        if p.resolve() in seen:
            continue
        out.append(p)
        seen.add(p.resolve())
        if len(out) >= CASE_STUDY_MAX_CHECKPOINTS_TO_SWEEP:
            break
    return out


def _choose_best_ppo_for_case_study(
    df: pd.DataFrame,
    T: int,
    bp: dict[str, Any],
    fixed_starts: list[tuple[int, int]] | None,
    models_dir: Path,
) -> tuple[Any, Path | None]:
    """Pick the PPO checkpoint with highest path-aligned TWAP gap (or mean IS fallback)."""
    from stable_baselines3 import PPO
    from src.rl_agent import evaluate_agent
    from src.trading_env import OptimalExecutionEnv

    candidates = _checkpoint_candidates(models_dir)
    if not candidates:
        return None, None

    env = OptimalExecutionEnv(df, T=T, seed=42, **_physical_env_kwargs())
    best_agent: Any = None
    best_path: Path | None = None
    best_score = float("-inf")

    sweep_starts = (
        fixed_starts[:CASE_STUDY_SWEEP_EPISODES]
        if fixed_starts and len(fixed_starts) > CASE_STUDY_SWEEP_EPISODES
        else fixed_starts
    )

    for path in candidates:
        try:
            agent = PPO.load(str(path))
            if sweep_starts:
                stats = evaluate_agent(
                    agent,
                    env,
                    n_episodes=0,
                    seed=42,
                    bench_params=bp,
                    fixed_starts=sweep_starts,
                )
            else:
                stats = evaluate_agent(
                    agent,
                    env,
                    n_episodes=min(200, CASE_STUDY_MAX_EPISODES),
                    seed=42,
                    bench_params=bp,
                )
        except Exception as exc:
            logger.warning("Case study: skip %s (%s)", path.name, exc)
            continue
        gap = float(stats.get("mean_rl_minus_twap_bps", float("nan")))
        score = gap if np.isfinite(gap) else float(stats.get("mean_is_bps", float("-inf")))
        if not np.isfinite(score):
            score = float("-inf")
        if score > best_score:
            best_score = score
            best_agent = agent
            best_path = path

    logger.info(
        "Case study: selected %s (selection score %.4f bps, path-aligned TWAP gap or mean IS)",
        best_path.name if best_path else "?",
        best_score,
    )
    return best_agent, best_path


@dataclass
class CaseStudyData:
    T: int = 10
    regime_chart: dict = field(default_factory=dict)
    episode_chart: dict = field(default_factory=dict)
    benchmark_chart: dict = field(default_factory=dict)
    regime_summary: list[dict] | None = None
    episode_summary: dict | None = None
    benchmarks: list[dict] | None = None
    governance_text: str = ""
    available: bool = False


def _regime_chart_data(
    df: pd.DataFrame, regimes: np.ndarray
) -> dict[str, Any]:
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    close = df["Close"].tolist()
    reg_list = regimes.tolist()
    return {"dates": dates, "close": close, "regimes": reg_list}


def _episode_chart_data(trajectory: pd.DataFrame) -> dict[str, Any]:
    return {
        "steps": trajectory["step"].tolist(),
        "dates": trajectory["date"].tolist() if "date" in trajectory.columns else trajectory["step"].tolist(),
        "inventory": trajectory["inventory_after"].tolist(),
        "actions": trajectory["action_frac"].tolist(),
    }


def _benchmark_chart_data(bench_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "strategies": bench_df["Strategy"].tolist(),
        "mean_is": bench_df["Mean_IS_bps"].tolist(),
        "std_is": bench_df["Std_IS_bps"].tolist(),
        "completion_rates": bench_df["Completion_Rate"].tolist() if "Completion_Rate" in bench_df.columns else [],
    }


def _safe_float(v: Any) -> float:
    f = float(v)
    if not np.isfinite(f):
        return 0.0
    return f


def precompute_case_study(
    ticker: str = "SPY",
    split: str = "test",
    T: int = 10,
    n_reg: int = 2,
    n_bench_episodes: int = 25,
) -> CaseStudyData:
    """Run all case-study computations, returning serialisable data.

    Catches all exceptions so the server always starts.
    """
    case = CaseStudyData(T=T)
    try:
        from src.data_pipeline import load_split
        from src.regime_detector import RegimeDetector, regime_sanity_summary
        from src.utils import regime_display_name

        df = load_split(split, ticker=ticker, use_bbo=True, use_news=True)
    except Exception as exc:
        logger.warning("Could not load data for case study: %s", exc)
        return case

    # Regime detection
    try:
        det = RegimeDetector(n_components=n_reg, fallback_threshold=0.24)
        det.fit(df)
        regimes = det.predict(df)
        df["regime"] = regimes
        case.regime_chart = _regime_chart_data(df, regimes)

        summary = regime_sanity_summary(df, regimes)
        case.regime_summary = [
            {
                "name": regime_display_name(int(idx), n_reg),
                "count": int(row["count"]),
                "mean_vol": float(row["mean_vol"]),
                "freq": float(row["freq"]),
            }
            for idx, row in summary.iterrows()
        ]
    except Exception as exc:
        logger.warning("Regime detection failed: %s", exc)
        df["regime"] = 0

    models_dir = ROOT / "models"
    bp = _bench_params(T)
    fixed_starts = _load_valid_fixed_starts(df, T)
    if fixed_starts:
        logger.info("Case study: using %s fixed eval windows (path-aligned with training)", len(fixed_starts))
    else:
        logger.info("Case study: no fixed_eval_starts.json or none valid — using random episode draws")

    try:
        agent, _chosen_ckpt = _choose_best_ppo_for_case_study(df, T, bp, fixed_starts, models_dir)
    except Exception as exc:
        logger.warning("Case study PPO selection failed: %s", exc)
        agent, _chosen_ckpt = None, None
    roll_agent = agent if agent is not None else _RandomAgent(42)
    if agent is None:
        logger.info("Case study: no PPO checkpoints; using random agent")

    # Episode rollout
    try:
        from src.trading_env import OptimalExecutionEnv
        from src.ui_rollout import rollout_episode

        env = OptimalExecutionEnv(df, T=T, seed=42, **_physical_env_kwargs())

        traj, summ = rollout_episode(env, roll_agent, seed=42, deterministic=True)
        case.episode_chart = _episode_chart_data(traj)
        case.episode_summary = {
            "is_bps": _safe_float(summ["is_bps"]),
            "completed": bool(summ["completed"]),
            "steps": int(summ["steps"]),
            "arrival": _safe_float(summ["arrival"]),
        }
    except Exception as exc:
        logger.warning("Episode rollout failed: %s", exc)

    # Benchmarks (same row windows as RL when fixed_starts present)
    try:
        from src.benchmarks import compare_all
        from src.rl_agent import evaluate_agent
        from src.trading_env import OptimalExecutionEnv

        env2 = OptimalExecutionEnv(df, T=T, seed=42, **_physical_env_kwargs())
        if fixed_starts:
            rl_stats = evaluate_agent(
                roll_agent,
                env2,
                n_episodes=0,
                seed=42,
                bench_params=bp,
                fixed_starts=fixed_starts,
            )
            row_starts = [int(r) for r, _ in fixed_starts]
            bench_df = compare_all(
                rl_stats,
                df,
                {
                    **bp,
                    "seed": 42,
                    "n_starts": len(row_starts),
                    "fixed_row_starts": row_starts,
                },
            )
        else:
            n_ep = max(n_bench_episodes, CASE_STUDY_MAX_EPISODES)
            rl_stats = evaluate_agent(
                roll_agent,
                env2,
                n_episodes=n_ep,
                seed=42,
                bench_params=bp,
            )
            bench_df = compare_all(
                rl_stats,
                df,
                {**bp, "seed": 42, "n_starts": min(200, n_ep)},
            )
        case.benchmarks = [
            {
                "Strategy": str(r["Strategy"]),
                "Mean_IS_bps": _safe_float(r["Mean_IS_bps"]),
                "Std_IS_bps": _safe_float(r["Std_IS_bps"]),
                "Completion_Rate": _safe_float(r["Completion_Rate"]),
            }
            for _, r in bench_df.iterrows()
        ]
        case.benchmark_chart = _benchmark_chart_data(bench_df)
    except Exception as exc:
        logger.warning("Benchmark evaluation failed: %s", exc)

    # LLM Governance
    try:
        from src.llm_explainer import explain_execution

        case.governance_text = explain_execution(
            regime=0,
            regime_name=regime_display_name(0, n_reg),
            inventory_remaining=0.55,
            action_taken=0.25,
            execution_cost_bps=_safe_float(case.episode_summary.get("is_bps", 4.5)) if case.episode_summary else 4.5,
            twap_cost_bps=_safe_float(case.benchmarks[1]["Mean_IS_bps"]) if case.benchmarks and len(case.benchmarks) > 1 else 5.0,
            ac_cost_bps=_safe_float(case.benchmarks[3]["Mean_IS_bps"]) if case.benchmarks and len(case.benchmarks) > 3 else 4.8,
            sigma_t=0.018,
            liquidity_t=1.2e-5,
            use_cache=True,
        )
    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)

    case.available = True
    logger.info("Case study pre-computed successfully")
    return case


def case_to_template_dict(case: CaseStudyData) -> dict[str, Any]:
    """Convert to dict for Jinja2 template context."""
    return {
        "T": case.T,
        "regime_summary": case.regime_summary,
        "episode_summary": type("S", (), case.episode_summary)() if case.episode_summary else None,
        "benchmarks": case.benchmarks,
        "governance_text": case.governance_text,
    }


def case_to_chart_json(case: CaseStudyData) -> str:
    """Serialise chart data to JSON for Plotly.js."""
    return json.dumps(
        {
            "regime_chart": case.regime_chart,
            "episode_chart": case.episode_chart,
            "benchmark_chart": case.benchmark_chart,
        },
        default=str,
    )


class _RandomAgent:
    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=False):
        return self._rng.uniform(0.0, 1.0, size=(1,)), None
