"""Judge-facing Streamlit dashboard. Run: streamlit run apps/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.benchmarks import compare_all
from src.data_pipeline import default_data_root, load_split
from src.llm_explainer import explain_execution
from src.regime_detector import RegimeDetector
from src.rl_agent import evaluate_agent
from src.trading_env import OptimalExecutionEnv
from src.ui_rollout import rollout_episode
from src.utils import regime_display_name


def _models():
    d = ROOT / "models"
    return sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []


class RandomAgent:
    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=False):  # noqa: ANN001
        return self._rng.uniform(0.0, 1.0, size=(1,)), None


def _load_ppo(path: str):
    from stable_baselines3 import PPO

    return PPO.load(path)


st.set_page_config(page_title="QuantFin Exeter — Execution Lab", layout="wide")
st.markdown(
    "<style>.main{background:linear-gradient(180deg,#f7f9fc 0%,#fff 45%);} h1{color:#1f3a5f;}</style>",
    unsafe_allow_html=True,
)
st.title("QuantFin Exeter — Execution lab")
st.caption("Team 18 · Regimes · RL · BBO imbalance (from 2019-01-02) · LLM governance")

dr = default_data_root()
if not (dr / "features").exists():
    st.error(f"Set `CFA_DATA_ROOT`. Missing: `{dr}/features`")
    st.stop()

sb = st.sidebar
ticker = sb.selectbox("Ticker", ["SPY", "AAPL"])
split_viz = sb.selectbox("Chart split", ["train", "val", "test"])
split_eval = sb.selectbox("Eval split", ["test", "val", "train"], index=0)
use_bbo = sb.checkbox("Merge BBO daily (needs `data/processed/bbo_daily.parquet`)", value=True)

try:
    df_viz = load_split(split_viz, ticker=ticker, use_bbo=use_bbo)
    df_eval = load_split(split_eval, ticker=ticker, use_bbo=use_bbo)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

n_reg = sb.slider("HMM states", 2, 3, 2)
T = sb.number_input("Horizon T", 3, 30, 10)
ep_seed = sb.number_input("Episode seed", 0, 999999, 42)
mods = _models()
pol = sb.radio(
    "Policy",
    ["Trained PPO", "Random"],
    index=0 if mods else 1,
)
ppo_p = str(mods[sb.selectbox("Checkpoint", range(len(mods)), format_func=lambda i: mods[i].name)]) if mods and pol == "Trained PPO" else None

_fit_key = (ticker, split_viz, n_reg, use_bbo)
if sb.button("Re-fit HMM on chart split"):
    st.session_state.fit_key = None
if st.session_state.get("fit_key") != _fit_key:
    det = RegimeDetector(n_components=n_reg, fallback_threshold=0.24)
    det.fit(df_viz)
    st.session_state.det = det
    st.session_state.reg_v = det.predict(df_viz)
    st.session_state.fit_key = _fit_key

df_viz = df_viz.copy()
df_viz["regime"] = st.session_state.reg_v
df_eval = df_eval.copy()
df_eval["regime"] = st.session_state.det.predict(df_eval)

tabs = st.tabs(["Overview", "Episode", "Benchmarks", "Governance"])

with tabs[0]:
    st.markdown(
        "**BBO note:** NASDAQ ITCH top-of-book 1m bars start **2019-01-02**. "
        "Earlier daily rows use neutral OBI (0) unless you rebuild features. "
        "This is *not* full LOBSTER depth—Level-1 only."
    )
    colors = {0: "#cfead8", 1: "#ffe9b3", 2: "#f8bcbc"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz["Close"], name="Close", line=dict(color="#1f3a5f")))
    for r in sorted(df_viz["regime"].unique()):
        idx = np.where(df_viz["regime"].to_numpy() == r)[0]
        for block in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(block):
                fig.add_vrect(
                    x0=df_viz.index[block[0]],
                    x1=df_viz.index[block[-1]],
                    fillcolor=colors.get(int(r), "#ddd"),
                    opacity=0.35,
                    layer="below",
                    line_width=0,
                )
    fig.update_layout(title=f"{ticker} price & regimes", template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        f2 = go.Figure(go.Scatter(x=df_viz.index, y=df_viz["realised_vol_20"], name="Vol"))
        f2.update_layout(title="Realised vol 20d (ann.)", template="plotly_white", height=280)
        st.plotly_chart(f2, use_container_width=True)
    with c2:
        vc = df_viz["regime"].value_counts().sort_index()
        f3 = go.Figure(go.Bar(x=[regime_display_name(int(i), n_reg) for i in vc.index], y=vc.values))
        f3.update_layout(title="Regime mix", template="plotly_white", height=280)
        st.plotly_chart(f3, use_container_width=True)

    if "order_imbalance_daily" in df_viz.columns and df_viz["order_imbalance_daily"].abs().sum() > 0:
        f4 = go.Figure(
            go.Scatter(
                x=df_viz.index,
                y=df_viz["order_imbalance_daily"],
                name="OBI",
                line=dict(color="#6a4c93"),
            )
        )
        f4.update_layout(
            title="Daily order imbalance (BBO 1m mean, [-1,1])",
            template="plotly_white",
            height=260,
            yaxis_range=[-1, 1],
        )
        st.plotly_chart(f4, use_container_width=True)
    elif use_bbo:
        st.info("Build `bbo_daily.parquet` with `python scripts/build_bbo_daily.py` to see OBI here.")

with tabs[1]:
    if st.button("Run episode", type="primary", key="ep"):
        with st.spinner("Rollout…"):
            try:
                env = OptimalExecutionEnv(df_eval, T=int(T), seed=int(ep_seed))
                agent = _load_ppo(ppo_p) if ppo_p else RandomAgent(int(ep_seed))
                tr, sm = rollout_episode(env, agent, seed=int(ep_seed), deterministic=True)
                st.session_state.traj = tr
                st.session_state.summ = sm
            except Exception as e:
                st.error(str(e))
                st.session_state.traj = None
    if st.session_state.get("traj") is not None and len(st.session_state.traj):
        sm = st.session_state.summ
        a, b, c, d = st.columns(4)
        a.metric("IS bps", f"{sm['is_bps']:.2f}")
        b.metric("Done", "Yes" if sm["completed"] else "No")
        c.metric("Steps", sm["steps"])
        d.metric("Arrival", f"{sm['arrival']:.4f}")
        tr = st.session_state.traj
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=tr["step"], y=tr["inventory_after"], name="Inv"), secondary_y=False)
        fig.add_trace(go.Bar(x=tr["step"], y=tr["action_frac"], name="a", opacity=0.7), secondary_y=True)
        fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click **Run episode**.")

with tabs[2]:
    n_ep = st.slider("Episodes", 10, 80, 25)
    if st.button("Run benchmarks", type="primary", key="bn"):
        with st.spinner("Evaluating…"):
            try:
                env = OptimalExecutionEnv(df_eval, T=int(T), seed=42)
                ag = _load_ppo(ppo_p) if ppo_p else RandomAgent(42)
                rl = evaluate_agent(ag, env, n_episodes=n_ep, seed=42)
                st.session_state.btab = compare_all(
                    rl,
                    df_eval,
                    params={"T": int(T), "seed": 42, "n_starts": min(50, n_ep)},
                )
            except Exception as e:
                st.error(str(e))
    if st.session_state.get("btab") is not None:
        st.dataframe(st.session_state.btab, hide_index=True, use_container_width=True)

with tabs[3]:
    rg = st.selectbox("Regime", [0, 1, 2], 0)
    inv = st.slider("Inventory left", 0.05, 1.0, 0.55)
    act = st.slider("Action", 0.05, 1.0, 0.25)
    ec, tw, ac = st.columns(3)
    g_ec = ec.number_input("Cost bps", value=4.5)
    g_tw = tw.number_input("TWAP bps", value=5.0)
    g_ac = ac.number_input("AC bps", value=4.8)
    sig = st.number_input("Daily vol", value=0.018, format="%.4f")
    liq = st.number_input("Amihud", value=1.2e-5, format="%.6f")
    if st.button("Generate explanation", type="primary", key="llm"):
        st.session_state.gtxt = explain_execution(
            int(rg),
            regime_display_name(int(rg), n_reg),
            float(inv),
            float(act),
            float(g_ec),
            float(g_tw),
            float(g_ac),
            float(sig),
            float(liq),
            use_cache=True,
        )
    if st.session_state.get("gtxt"):
        st.info(st.session_state.gtxt)
