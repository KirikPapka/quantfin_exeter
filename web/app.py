"""Flask web application for QuantFin Exeter.

Run (from the ``quantfin_exeter`` directory so imports resolve):

    cd quantfin_exeter
    python -m web.app

The dev server listens on port **5001** (not 5000). User Manual: ``/user-manual`` (also ``/manual`` and ``/user_manual``).
Restart the process after code changes; a stale server will not have new routes.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify

from web.precompute import (
    CaseStudyData,
    ORDER_NOTIONAL_USD,
    precompute_case_study,
    case_to_template_dict,
    case_to_chart_json,
    _RandomAgent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder=str(ROOT / "web" / "static"),
    template_folder=str(ROOT / "web" / "templates"),
)
app.config["SECRET_KEY"] = "quantfin-exeter-team18"

# Match training / scenario_benchmarks physical institutional setup
ORDER_NOTIONAL_USD = 5_000_000.0
RESIDUAL_BOUND = 0.15
RELATIVE_IS_SCALE = 2.0
ORDER_START_BAR = 0


def _physical_env_kwargs() -> dict:
    from src.trading_env import physical_institutional_kwargs

    kw = physical_institutional_kwargs(
        ORDER_NOTIONAL_USD,
        residual_bound=RESIDUAL_BOUND,
        relative_is_scale=RELATIVE_IS_SCALE,
    )
    return {"order_notional_usd": ORDER_NOTIONAL_USD, **kw}


def _bench_params(T: int) -> dict:
    return {
        "T": T,
        "X_0": 1.0,
        "eta": 0.01,
        "gamma": 0.001,
        "lam": 0.5,
        "order_notional_usd": ORDER_NOTIONAL_USD,
        "order_start_bar": ORDER_START_BAR,
    }


# ---------------------------------------------------------------------------
# Startup: pre-compute case study
# ---------------------------------------------------------------------------
logger.info("Pre-computing case study data (this may take a moment)...")
CASE_STUDY: CaseStudyData = precompute_case_study()
logger.info("Case study ready (available=%s)", CASE_STUDY.available)

# ---------------------------------------------------------------------------
# Exec Lab cache (master process only — avoids PPO.load / torch in gunicorn workers)
# ---------------------------------------------------------------------------
_LAB_CACHE_DF: dict[str, Any] = {}
_LAB_CACHE_PPO_AGENT: object | None = None
_LAB_CACHE_PPO_RESOLVED: str | None = None


def _init_exec_lab_cache() -> None:
    """Load splits and PPO once at import so forked workers never call torch.load."""
    global _LAB_CACHE_DF, _LAB_CACHE_PPO_AGENT, _LAB_CACHE_PPO_RESOLVED
    from src.data_pipeline import load_split

    for split in ("train", "val", "test"):
        try:
            _LAB_CACHE_DF[split] = load_split(split, ticker="SPY", use_bbo=True, use_news=True)
        except Exception as exc:
            logger.warning("Exec lab cache: could not load split %s: %s", split, exc)

    best = ROOT / "models" / "best_ppo_twap_gap.zip"
    if best.is_file():
        try:
            from stable_baselines3 import PPO

            _LAB_CACHE_PPO_RESOLVED = str(best.resolve())
            logger.info("Exec lab cache: loading PPO once at %s", _LAB_CACHE_PPO_RESOLVED)
            _LAB_CACHE_PPO_AGENT = PPO.load(_LAB_CACHE_PPO_RESOLVED)
            logger.info("Exec lab cache: PPO ready (workers will reuse; no torch.load per request)")
        except Exception as exc:
            _LAB_CACHE_PPO_RESOLVED = None
            _LAB_CACHE_PPO_AGENT = None
            logger.warning("Exec lab cache: could not preload PPO (%s)", exc)
    else:
        logger.info("Exec lab cache: no best_ppo_twap_gap.zip — PPO routes cold-load or use random")


_init_exec_lab_cache()


def _lab_df_for_split(split: str):
    """Copy of cached panel for this split, or None if not cached."""
    base = _LAB_CACHE_DF.get(split)
    if base is None:
        return None
    return base.copy()


def _lab_agent_for_policy(policy_path: str, seed: int):
    """Agent for rollout/benchmarks. Random is always fresh; PPO hits cache when path matches best zip."""
    if not policy_path or policy_path == "random":
        return _RandomAgent(seed)

    try:
        req_resolved = str(Path(policy_path).resolve())
    except Exception:
        req_resolved = str(policy_path)

    if (
        _LAB_CACHE_PPO_AGENT is not None
        and _LAB_CACHE_PPO_RESOLVED is not None
        and req_resolved == _LAB_CACHE_PPO_RESOLVED
    ):
        return _LAB_CACHE_PPO_AGENT

    from stable_baselines3 import PPO

    return PPO.load(policy_path)


def _list_models() -> list[dict]:
    d = ROOT / "models"
    if not d.is_dir():
        return []
    best = d / "best_ppo_twap_gap.zip"
    zips = sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    ordered: list[Path] = []
    if best.is_file():
        ordered.append(best)
    for p in zips:
        if p.resolve() != best.resolve():
            ordered.append(p)
    return [{"name": p.name, "path": str(p)} for p in ordered]


def _safe_float(v) -> float:
    f = float(v)
    return 0.0 if not np.isfinite(f) else f


import pandas as _pd  # noqa: E402  (already imported transitively; explicit for type checker)

def _resolve_start_date(df: "_pd.DataFrame", start_date: str | None, T: int) -> int:
    """Map a YYYY-MM-DD string to a valid row_start index in df.

    Non-trading days snap backward (ffill) to the nearest previous session.
    Raises ValueError with a user-friendly message if the date is out of range.
    """
    if not start_date:
        raise ValueError("Please select an execution start date.")
    import pandas as pd
    target = pd.Timestamp(start_date)
    idx = int(df.index.get_indexer([target], method="ffill")[0])
    if idx < 0:
        raise ValueError(
            f"Date {start_date} is before the data starts ({df.index[0].date()}). "
            "Pick a later date."
        )
    max_start = len(df) - T - 1
    if idx > max_start:
        raise ValueError(
            f"Date {start_date} is too close to the end of the data: "
            f"need at least {T} bars after it "
            f"(last valid start: {df.index[max_start].date()})."
        )
    return idx


def _split_date_context(split: str = "test", T: int = 10) -> dict:
    """Return min/max/default date strings for the date picker."""
    import pandas as pd
    df = _lab_df_for_split(split)
    if df is None or len(df) < T + 2:
        return {"split_date_min": "", "split_date_max": "", "split_date_default": "", "horizon_T": T}
    date_min = df.index[0].date().isoformat()
    date_max = df.index[max(0, len(df) - T - 1)].date().isoformat()
    # Default: ~25% into the split (a reasonable mid-point)
    default_idx = min(len(df) // 4, len(df) - T - 1)
    date_default = df.index[default_idx].date().isoformat()
    return {
        "split_date_min": date_min,
        "split_date_max": date_max,
        "split_date_default": date_default,
        "horizon_T": T,
    }


def _showcase_benchmark_rows(benchmarks: list | None, notional: float = 5_000_000.0) -> list[dict] | None:
    if not benchmarks:
        return None
    by_name = {str(b["Strategy"]): b for b in benchmarks}
    order = ["RL", "TWAP", "Immediate"]
    out = []
    for s in order:
        if s not in by_name:
            continue
        row = dict(by_name[s])
        row["Mean_IS_usd"] = _safe_float(row["Mean_IS_bps"]) * notional / 10_000.0
        row["Std_IS_usd"] = _safe_float(row["Std_IS_bps"]) * notional / 10_000.0
        out.append(row)
    return out or None


def _usd_showcase_context(benchmarks: list | None, notional: float) -> dict | None:
    """Dollar interpretation for sell IS (higher bps = better avg price vs arrival)."""
    if not benchmarks:
        return None
    by_name = {str(b["Strategy"]): b for b in benchmarks}
    rl = by_name.get("RL")
    tw = by_name.get("TWAP")
    vw = by_name.get("VWAP")
    imm = by_name.get("Immediate")
    if not rl or not tw:
        return None
    rl_bps = float(rl["Mean_IS_bps"])
    tw_bps = float(tw["Mean_IS_bps"])
    gap_vs_twap = rl_bps - tw_bps
    usd_vs_twap = notional * gap_vs_twap / 10_000.0
    ctx: dict[str, float | str] = {
        "notional": notional,
        "rl_bps": rl_bps,
        "twap_bps": tw_bps,
        "gap_vs_twap_bps": gap_vs_twap,
        "usd_vs_twap": usd_vs_twap,
    }
    if vw:
        vw_bps = float(vw["Mean_IS_bps"])
        ctx["vwap_bps"] = vw_bps
        ctx["gap_vs_vwap_bps"] = rl_bps - vw_bps
        ctx["usd_vs_vwap"] = notional * (rl_bps - vw_bps) / 10_000.0
    if imm:
        imm_bps = float(imm["Mean_IS_bps"])
        ctx["immediate_bps"] = imm_bps
        ctx["gap_vs_immediate_bps"] = rl_bps - imm_bps
        ctx["usd_vs_immediate"] = notional * (rl_bps - imm_bps) / 10_000.0
    return ctx


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", active_page="home")


@app.route("/case-study")
def case_study():
    bench = CASE_STUDY.benchmarks
    return render_template(
        "case_study.html",
        active_page="case_study",
        case=case_to_template_dict(CASE_STUDY),
        case_json=case_to_chart_json(CASE_STUDY),
        showcase_rows=_showcase_benchmark_rows(bench, ORDER_NOTIONAL_USD),
        usd_showcase=_usd_showcase_context(bench, ORDER_NOTIONAL_USD),
        order_notional_usd=ORDER_NOTIONAL_USD,
    )


@app.route("/run")
def run_page():
    return render_template(
        "run.html",
        active_page="run",
        models=_list_models(),
        **_split_date_context("test", T=10),
    )


@app.route("/user-manual")
@app.route("/user_manual")
@app.route("/manual")
def user_manual():
    """Primary URL is /user-manual; aliases help bookmarks and proxies that mishandle hyphens."""
    return render_template("user_manual.html", active_page="user_manual")


# ---------------------------------------------------------------------------
# API: Regime Detection
# ---------------------------------------------------------------------------
@app.route("/api/regimes", methods=["POST"])
def api_regimes():
    try:
        from src.data_pipeline import load_split
        from src.regime_detector import RegimeDetector, regime_sanity_summary
        from src.utils import regime_display_name

        split = request.form.get("split", "test")
        n_reg = int(request.form.get("n_reg", 2))

        df = _lab_df_for_split(split)
        if df is None:
            df = load_split(split, ticker="SPY", use_bbo=True, use_news=True)
        det = RegimeDetector(n_components=n_reg, fallback_threshold=0.24)
        det.fit(df)
        regimes = det.predict(df)

        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        chart_data = {
            "dates": dates,
            "close": df["Close"].tolist(),
            "regimes": regimes.tolist(),
        }

        summary_df = regime_sanity_summary(df, regimes)
        summary = [
            {
                "name": regime_display_name(int(idx), n_reg),
                "count": int(row["count"]),
                "mean_vol": float(row["mean_vol"]),
            }
            for idx, row in summary_df.iterrows()
        ]

        return render_template(
            "partials/regime_chart.html",
            error=None,
            chart_json=json.dumps(chart_data),
            summary=summary,
        )
    except Exception as exc:
        logger.exception("Regime detection failed")
        return render_template("partials/regime_chart.html", error=str(exc))


# ---------------------------------------------------------------------------
# API: Episode Rollout
# ---------------------------------------------------------------------------
@app.route("/api/episode", methods=["POST"])
def api_episode():
    try:
        from src.data_pipeline import load_split
        from src.regime_detector import RegimeDetector
        from src.trading_env import OptimalExecutionEnv
        from src.ui_rollout import rollout_episode

        split = request.form.get("split", "test")
        n_reg = int(request.form.get("n_reg", 2))
        T = int(request.form.get("horizon", 10))
        start_date = request.form.get("start_date", "").strip()
        policy_path = request.form.get("policy", "random")

        df = _lab_df_for_split(split)
        if df is None:
            df = load_split(split, ticker="SPY", use_bbo=True, use_news=True)
        row_start = _resolve_start_date(df, start_date, T)

        det = RegimeDetector(n_components=n_reg, fallback_threshold=0.24)
        det.fit(df)
        df["regime"] = det.predict(df)

        env = OptimalExecutionEnv(df, T=T, resample=False, **_physical_env_kwargs())
        agent = _lab_agent_for_policy(policy_path, 42)

        traj, summ = rollout_episode(env, agent, seed=42, deterministic=True, row_start=row_start)
        chart_data = {
            "steps": traj["step"].tolist(),
            "dates": traj["date"].tolist() if "date" in traj.columns else traj["step"].tolist(),
            "inventory": traj["inventory_after"].tolist(),
            "actions": traj["action_frac"].tolist(),
        }

        summary = type("S", (), {
            "is_bps": _safe_float(summ["is_bps"]),
            "completed": bool(summ["completed"]),
            "steps": int(summ["steps"]),
            "arrival": _safe_float(summ["arrival"]),
        })()

        return render_template(
            "partials/episode_result.html",
            error=None,
            chart_json=json.dumps(chart_data),
            summary=summary,
        )
    except Exception as exc:
        logger.exception("Episode rollout failed")
        return render_template("partials/episode_result.html", error=str(exc))


# ---------------------------------------------------------------------------
# API: Benchmarks
# ---------------------------------------------------------------------------
@app.route("/api/benchmarks", methods=["POST"])
def api_benchmarks():
    try:
        from src.data_pipeline import load_split
        from src.regime_detector import RegimeDetector
        from src.trading_env import OptimalExecutionEnv
        from src.benchmarks import compare_all
        from src.rl_agent import evaluate_agent, format_rl_eval_report

        split = request.form.get("split", "test")
        n_reg = int(request.form.get("n_reg", 2))
        T = int(request.form.get("horizon", 10))
        start_date = request.form.get("start_date", "").strip()
        policy_path = request.form.get("policy", "random")

        df = _lab_df_for_split(split)
        if df is None:
            df = load_split(split, ticker="SPY", use_bbo=True, use_news=True)
        row_start = _resolve_start_date(df, start_date, T)

        det = RegimeDetector(n_components=n_reg, fallback_threshold=0.24)
        det.fit(df)
        df["regime"] = det.predict(df)

        env = OptimalExecutionEnv(df, T=T, resample=False, **_physical_env_kwargs())
        agent = _lab_agent_for_policy(policy_path, 42)

        bp = _bench_params(T)
        rl_stats = evaluate_agent(
            agent, env, n_episodes=0, seed=42, bench_params=bp,
            fixed_starts=[(row_start, 42)],
        )
        bench_df = compare_all(rl_stats, df, {**bp, "seed": 42, "n_starts": 1, "fixed_row_starts": [row_start]})

        rows = [
            {
                "Strategy": str(r["Strategy"]),
                "Mean_IS_bps": _safe_float(r["Mean_IS_bps"]),
                "Std_IS_bps": _safe_float(r["Std_IS_bps"]),
                "Completion_Rate": _safe_float(r["Completion_Rate"]),
            }
            for _, r in bench_df.iterrows()
        ]
        chart_data = {
            "strategies": bench_df["Strategy"].tolist(),
            "mean_is": [_safe_float(v) for v in bench_df["Mean_IS_bps"]],
            "std_is": [_safe_float(v) for v in bench_df["Std_IS_bps"]],
            "completion_rates": [_safe_float(v) for v in bench_df["Completion_Rate"]] if "Completion_Rate" in bench_df.columns else [],
        }
        rl_report = format_rl_eval_report(rl_stats)

        return render_template(
            "partials/benchmark_table.html",
            error=None,
            rows=rows,
            chart_json=json.dumps(chart_data),
            rl_report=rl_report,
        )
    except Exception as exc:
        logger.exception("Benchmark evaluation failed")
        return render_template("partials/benchmark_table.html", error=str(exc))


# ---------------------------------------------------------------------------
# API: Run All (full pipeline)
# ---------------------------------------------------------------------------
@app.route("/api/run-all", methods=["POST"])
def api_run_all():
    try:
        from src.data_pipeline import load_split
        from src.regime_detector import RegimeDetector, regime_sanity_summary
        from src.trading_env import OptimalExecutionEnv
        from src.ui_rollout import rollout_episode
        from src.benchmarks import compare_all
        from src.rl_agent import evaluate_agent, format_rl_eval_report
        from src.llm_explainer import explain_execution
        from src.utils import regime_display_name

        split = request.form.get("split", "test")
        n_reg = int(request.form.get("n_reg", 2))
        T = int(request.form.get("horizon", 10))
        start_date = request.form.get("start_date", "").strip()
        policy_path = request.form.get("policy", "random")

        df = _lab_df_for_split(split)
        if df is None:
            df = load_split(split, ticker="SPY", use_bbo=True, use_news=True)
        row_start = _resolve_start_date(df, start_date, T)

        det = RegimeDetector(n_components=n_reg, fallback_threshold=0.24)
        det.fit(df)
        regimes = det.predict(df)
        df["regime"] = regimes

        # Regime chart data
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        regime_chart_json = json.dumps({
            "dates": dates,
            "close": df["Close"].tolist(),
            "regimes": regimes.tolist(),
        })
        summary_df = regime_sanity_summary(df, regimes)
        regime_summary = [
            {
                "name": regime_display_name(int(idx), n_reg),
                "count": int(row["count"]),
                "mean_vol": float(row["mean_vol"]),
            }
            for idx, row in summary_df.iterrows()
        ]

        agent = _lab_agent_for_policy(policy_path, 42)

        # Episode — anchored to the user-selected start date
        env = OptimalExecutionEnv(df, T=T, resample=False, **_physical_env_kwargs())
        traj, summ = rollout_episode(env, agent, seed=42, deterministic=True, row_start=row_start)
        episode_chart_json = json.dumps({
            "steps": traj["step"].tolist(),
            "dates": traj["date"].tolist() if "date" in traj.columns else traj["step"].tolist(),
            "inventory": traj["inventory_after"].tolist(),
            "actions": traj["action_frac"].tolist(),
        })
        episode_summary = type("S", (), {
            "is_bps": _safe_float(summ["is_bps"]),
            "completed": bool(summ["completed"]),
            "steps": int(summ["steps"]),
            "arrival": _safe_float(summ["arrival"]),
        })()

        # Benchmarks — same single window as the episode
        env2 = OptimalExecutionEnv(df, T=T, resample=False, **_physical_env_kwargs())
        bp = _bench_params(T)
        rl_stats = evaluate_agent(
            agent, env2, n_episodes=0, seed=42, bench_params=bp,
            fixed_starts=[(row_start, 42)],
        )
        bench_df = compare_all(rl_stats, df, {**bp, "seed": 42, "n_starts": 1, "fixed_row_starts": [row_start]})
        bench_rows = [
            {
                "Strategy": str(r["Strategy"]),
                "Mean_IS_bps": _safe_float(r["Mean_IS_bps"]),
                "Std_IS_bps": _safe_float(r["Std_IS_bps"]),
                "Completion_Rate": _safe_float(r["Completion_Rate"]),
            }
            for _, r in bench_df.iterrows()
        ]
        bench_chart_json = json.dumps({
            "strategies": bench_df["Strategy"].tolist(),
            "mean_is": [_safe_float(v) for v in bench_df["Mean_IS_bps"]],
            "std_is": [_safe_float(v) for v in bench_df["Std_IS_bps"]],
            "completion_rates": [_safe_float(v) for v in bench_df["Completion_Rate"]] if "Completion_Rate" in bench_df.columns else [],
        })
        rl_report = format_rl_eval_report(rl_stats)

        # Governance
        gov_text = ""
        try:
            twap_bps = bench_rows[1]["Mean_IS_bps"] if len(bench_rows) > 1 else 5.0
            ac_bps = bench_rows[3]["Mean_IS_bps"] if len(bench_rows) > 3 else 4.8
            gov_text = explain_execution(
                regime=0,
                regime_name=regime_display_name(0, n_reg),
                inventory_remaining=0.55,
                action_taken=0.25,
                execution_cost_bps=_safe_float(summ["is_bps"]),
                twap_cost_bps=twap_bps,
                ac_cost_bps=ac_bps,
                sigma_t=0.018,
                liquidity_t=1.2e-5,
                use_cache=True,
            )
        except Exception as exc:
            logger.warning("Governance explanation failed: %s", exc)
            gov_text = f"Could not generate explanation: {exc}"

        return render_template(
            "partials/run_all_result.html",
            # regime
            error=None,
            chart_json=regime_chart_json,
            summary=regime_summary,
            # episode
            episode_chart_json=episode_chart_json,
            episode_summary=episode_summary,
            # benchmarks
            rows=bench_rows,
            bench_chart_json=bench_chart_json,
            rl_report=rl_report,
            # governance
            text=gov_text,
        )
    except Exception as exc:
        logger.exception("Full pipeline failed")
        return f'<div class="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">{exc}</div>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
