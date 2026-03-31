"""TWAP, VWAP, Almgren–Chriss, immediate execution with optional physical notional + impact."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .execution_impact import (
    ImpactHyper,
    arrival_price_full,
    dollar_bar_volume,
    effective_horizon,
    impact_hyper_from_dict,
    sell_effective_close,
)

logger = logging.getLogger(__name__)


def _slice_episode(price_data: pd.DataFrame, start: int, T: int) -> pd.DataFrame:
    return price_data.iloc[start : start + T].copy()


@dataclass
class _ExecBundle:
    legacy: bool
    arrival: float
    Q: float
    t0: int
    T_eff: int
    hyper: ImpactHyper


def _exec_bundle(
    price_data: pd.DataFrame, start: int, T: int, params: dict[str, Any]
) -> _ExecBundle:
    t0 = int(params.get("order_start_bar", 0))
    t0 = max(0, min(t0, max(T - 1, 0)))
    notional = float(params.get("order_notional_usd", 0.0) or 0.0)
    hyper = impact_hyper_from_dict(params)
    x0 = float(params.get("X_0", 1.0))
    if notional <= 0:
        ep = _slice_episode(price_data, start, T)
        arr = float(ep.iloc[0]["Close"]) if len(ep) else 0.0
        return _ExecBundle(True, arr, x0, 0, T, hyper)
    arr = arrival_price_full(price_data, start, t0)
    q = notional / max(arr, 1e-12)
    teff = effective_horizon(T, t0)
    return _ExecBundle(False, float(arr), float(q), t0, teff, hyper)


def twap_execution(
    X_0: float,
    T: int,
    price_data: pd.DataFrame,
    start: int = 0,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    ep = _slice_episode(price_data, start, T)
    if ep.empty:
        return {"execution_cost_bps": 0.0, "implementation_shortfall": 0.0}
    p = params or {}
    b = _exec_bundle(price_data, start, T, {**p, "X_0": X_0})
    arrival = b.arrival
    if b.legacy:
        per = b.Q / T
        x_rem = b.Q
        num = den = 0.0
        for _, row in ep.iterrows():
            px = float(row["Close"])
            v = min(per, x_rem)
            num += v * px
            den += v
            x_rem -= v
    else:
        n_b = max(b.T_eff, 1)
        per = b.Q / n_b
        x_rem = b.Q
        num = den = 0.0
        for k in range(b.t0, T):
            row = ep.iloc[k]
            base = float(row["Close"])
            v = min(per, x_rem)
            px = sell_effective_close(base, v, row, b.hyper)
            num += v * px
            den += v
            x_rem -= v
    avg_px = num / den if den > 0 else arrival
    is_frac = (avg_px - arrival) / arrival if arrival else 0.0
    return {"execution_cost_bps": float(is_frac * 1e4), "implementation_shortfall": float(is_frac)}


def vwap_execution(
    X_0: float,
    T: int,
    price_data: pd.DataFrame,
    start: int = 0,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    ep = _slice_episode(price_data, start, T)
    if ep.empty:
        return {"execution_cost_bps": 0.0, "implementation_shortfall": 0.0}
    p = params or {}
    b = _exec_bundle(price_data, start, T, {**p, "X_0": X_0})
    arrival = b.arrival
    if b.legacy:
        vol = ep["Volume"].astype(float).to_numpy()
        vol = np.where(np.isfinite(vol) & (vol > 0), vol, 1.0)
        w = vol / vol.sum()
        x_rem = b.Q
        num = den = 0.0
        for wi, (_, row) in zip(w, ep.iterrows()):
            px = float(row["Close"])
            v = min(b.Q * float(wi), x_rem)
            num += v * px
            den += v
            x_rem -= v
    else:
        sub = ep.iloc[b.t0 : T]
        vol = sub["Volume"].astype(float).to_numpy()
        vol = np.where(np.isfinite(vol) & (vol > 0), vol, 1.0)
        w = vol / vol.sum()
        x_rem = b.Q
        num = den = 0.0
        for wi, (_, row) in zip(w, sub.iterrows()):
            base = float(row["Close"])
            v = min(b.Q * float(wi), x_rem)
            px = sell_effective_close(base, v, row, b.hyper)
            num += v * px
            den += v
            x_rem -= v
    avg_px = num / den if den > 0 else arrival
    is_frac = (avg_px - arrival) / arrival if arrival else 0.0
    return {"execution_cost_bps": float(is_frac * 1e4), "implementation_shortfall": float(is_frac)}


def immediate_execution(
    X_0: float,
    T: int,
    price_data: pd.DataFrame,
    start: int = 0,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    ep = _slice_episode(price_data, start, T)
    if ep.empty:
        return {"execution_cost_bps": 0.0, "implementation_shortfall": 0.0}
    p = params or {}
    b = _exec_bundle(price_data, start, T, {**p, "X_0": X_0})
    arrival = b.arrival
    if b.legacy:
        px = float(ep.iloc[0]["Close"])
        v = float(b.Q)
        num, den = v * px, v
        avg_px = num / den if den > 0 else arrival
    else:
        row = ep.iloc[b.t0]
        base = float(row["Close"])
        avg_px = sell_effective_close(base, b.Q, row, b.hyper)
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
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    ep = _slice_episode(price_data, start, T)
    if ep.empty:
        return {"execution_cost_bps": 0.0, "implementation_shortfall": 0.0}
    p = params or {}
    b = _exec_bundle(price_data, start, T, {**p, "X_0": X_0})
    arrival = b.arrival
    teff = b.T_eff if not b.legacy else T
    kappa = np.sqrt(max(lam * (sigma**2) / max(eta, 1e-16), 1e-16))
    idx = np.arange(teff + 1)
    traj = b.Q * np.sinh(kappa * (teff - idx)) / np.sinh(kappa * teff + 1e-16)
    trades = -np.diff(traj)
    num = den = 0.0
    if b.legacy:
        for t_i, (_, row) in enumerate(ep.iterrows()):
            if t_i >= len(trades):
                break
            v = float(trades[t_i])
            px = float(row["Close"])
            num += v * px
            den += v
    else:
        for t_i in range(len(trades)):
            row = ep.iloc[b.t0 + t_i]
            base = float(row["Close"])
            v = float(trades[t_i])
            px = sell_effective_close(base, v, row, b.hyper)
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
    bench_params = {k: v for k, v in params.items()}
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
                r = fn(
                    X_0=X_0,
                    T=T,
                    price_data=price_data,
                    start=int(s),
                    params=bench_params,
                    **kw,
                )
                out.append(float(r["implementation_shortfall"]) * 1e4)
            except Exception as e:  # noqa: BLE001
                logger.debug("skip %s: %s", s, e)
        return out

    twap_vals = _collect(twap_execution)
    vwap_vals = _collect(vwap_execution)
    ac_vals = _collect(
        almgren_chriss_execution, eta=eta, gamma=gamma, sigma=sigma, lam=lam
    )
    imm_vals = _collect(immediate_execution)
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
        {
            "Strategy": "Immediate",
            "Mean_IS_bps": float(np.mean(imm_vals)) if imm_vals else np.nan,
            "Std_IS_bps": float(np.std(imm_vals)) if imm_vals else np.nan,
            "Completion_Rate": 1.0,
        },
    ]
    return pd.DataFrame(rows)


def execution_summary_row(price_data: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Single-row context for UI: notional → shares at a representative price."""
    T = int(params.get("T", 10))
    start = max(0, len(price_data) // 2)
    if start + T > len(price_data):
        start = max(0, len(price_data) - T - 1)
    b = _exec_bundle(price_data, start, T, params)
    t0 = b.t0
    exec_bar = min(start + t0, len(price_data) - 1)
    ref_close = float(price_data.iloc[exec_bar]["Close"])
    dv = dollar_bar_volume(price_data.iloc[exec_bar])
    part = (b.Q * ref_close) / dv if not b.legacy and b.Q > 0 else 0.0
    return {
        "mode": "legacy_normalized" if b.legacy else "physical_usd",
        "arrival_price_usd": b.arrival,
        "shares_approx": b.Q,
        "order_start_bar": t0,
        "exec_horizon_bars": b.T_eff if not b.legacy else T,
        "ref_close_usd": ref_close,
        "approx_participation_first_bar": float(part),
    }
