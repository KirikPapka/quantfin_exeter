"""TWAP, VWAP, Almgren–Chriss."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _slice_episode(price_data: pd.DataFrame, start: int, T: int) -> pd.DataFrame:
    return price_data.iloc[start : start + T].copy()


def twap_execution(
    X_0: float, T: int, price_data: pd.DataFrame, start: int = 0
) -> dict[str, float]:
    ep = _slice_episode(price_data, start, T)
    arrival = float(ep.iloc[0]["Close"])
    per = X_0 / T
    x_rem = X_0
    num = den = 0.0
    for _, row in ep.iterrows():
        px = float(row["Close"])
        v = min(per, x_rem)
        num += v * px
        den += v
        x_rem -= v
    avg_px = num / den if den > 0 else arrival
    is_frac = (avg_px - arrival) / arrival if arrival else 0.0
    return {"execution_cost_bps": float(is_frac * 1e4), "implementation_shortfall": float(is_frac)}


def vwap_execution(
    X_0: float, T: int, price_data: pd.DataFrame, start: int = 0
) -> dict[str, float]:
    ep = _slice_episode(price_data, start, T)
    vol = ep["Volume"].astype(float).to_numpy()
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, 1.0)
    w = vol / vol.sum()
    arrival = float(ep.iloc[0]["Close"])
    x_rem = X_0
    num = den = 0.0
    for wi, (_, row) in zip(w, ep.iterrows()):
        px = float(row["Close"])
        v = min(X_0 * float(wi), x_rem)
        num += v * px
        den += v
        x_rem -= v
    avg_px = num / den if den > 0 else arrival
    is_frac = (avg_px - arrival) / arrival if arrival else 0.0
    return {"execution_cost_bps": float(is_frac * 1e4), "implementation_shortfall": float(is_frac)}


def almgren_chriss_execution(
    X_0: float,
    T: int,
    price_data: pd.DataFrame,
    eta: float,
    gamma: float,
    sigma: float,
    lam: float,
    start: int = 0,
) -> dict[str, float]:
    ep = _slice_episode(price_data, start, T)
    arrival = float(ep.iloc[0]["Close"])
    kappa = np.sqrt(max(lam * (sigma**2) / eta, 1e-16))
    idx = np.arange(T + 1)
    traj = X_0 * np.sinh(kappa * (T - idx)) / np.sinh(kappa * T + 1e-16)
    trades = -np.diff(traj)
    num = den = 0.0
    for t_i, (_, row) in enumerate(ep.iterrows()):
        if t_i >= len(trades):
            break
        v = float(trades[t_i])
        px = float(row["Close"])
        num += v * px
        den += v
    avg_px = num / den if den > 0 else arrival
    is_frac = (avg_px - arrival) / arrival if arrival else 0.0
    return {"execution_cost_bps": float(is_frac * 1e4), "implementation_shortfall": float(is_frac)}


def compare_all(
    rl_results: dict[str, Any],
    price_data: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    T = int(params.get("T", 10))
    X_0 = float(params.get("X_0", 1.0))
    eta = float(params.get("eta", 0.01))
    gamma = float(params.get("gamma", 0.001))
    lam = float(params.get("lam", 0.5))
    n_starts = int(params.get("n_starts", 50))
    sigma = float(
        params.get(
            "sigma",
            float(price_data["sigma_daily"].mean())
            if "sigma_daily" in price_data.columns
            else 0.01,
        )
    )
    max_start = len(price_data) - T - 1
    if max_start < 0:
        raise ValueError("price_data too short.")
    rng = np.random.default_rng(params.get("seed", 42))
    starts = rng.choice(max_start + 1, size=min(n_starts, max_start + 1), replace=False)

    def _collect(fn, **kw: Any) -> list[float]:
        out: list[float] = []
        for s in starts:
            try:
                r = fn(X_0=X_0, T=T, price_data=price_data, start=int(s), **kw)
                out.append(float(r["implementation_shortfall"]) * 1e4)
            except Exception as e:  # noqa: BLE001
                logger.debug("skip %s: %s", s, e)
        return out

    twap_vals = _collect(twap_execution)
    vwap_vals = _collect(vwap_execution)
    ac_vals = _collect(
        almgren_chriss_execution, eta=eta, gamma=gamma, sigma=sigma, lam=lam
    )
    rows = [
        {
            "Strategy": "RL",
            "Mean_IS_bps": float(rl_results.get("mean_is_bps", np.nan)),
            "Std_IS_bps": float(rl_results.get("std_is_bps", np.nan)),
            "Completion_Rate": float(rl_results.get("completion_rate", np.nan)),
        },
        {
            "Strategy": "TWAP",
            "Mean_IS_bps": float(np.mean(twap_vals)) if twap_vals else np.nan,
            "Std_IS_bps": float(np.std(twap_vals)) if twap_vals else np.nan,
            "Completion_Rate": 1.0,
        },
        {
            "Strategy": "VWAP",
            "Mean_IS_bps": float(np.mean(vwap_vals)) if vwap_vals else np.nan,
            "Std_IS_bps": float(np.std(vwap_vals)) if vwap_vals else np.nan,
            "Completion_Rate": 1.0,
        },
        {
            "Strategy": "Almgren-Chriss",
            "Mean_IS_bps": float(np.mean(ac_vals)) if ac_vals else np.nan,
            "Std_IS_bps": float(np.std(ac_vals)) if ac_vals else np.nan,
            "Completion_Rate": 1.0,
        },
    ]
    return pd.DataFrame(rows)
