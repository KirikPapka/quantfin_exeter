"""Shared microstructure-style impact for benchmarks and (optional) RL env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class ImpactHyper:
    """Tunable impact law (sell / liquidation: effective price below base)."""

    sigma_coef: float = 0.65
    amihud_coef: float = 0.35
    part_cap_sqrt: float = 12.0
    max_impact_frac: float = 0.35


def impact_hyper_from_dict(d: dict[str, Any]) -> ImpactHyper:
    return ImpactHyper(
        sigma_coef=float(d.get("impact_sigma_coef", ImpactHyper().sigma_coef)),
        amihud_coef=float(d.get("impact_amihud_coef", ImpactHyper().amihud_coef)),
        part_cap_sqrt=float(d.get("impact_part_cap_sqrt", ImpactHyper().part_cap_sqrt)),
        max_impact_frac=float(d.get("impact_max_frac", ImpactHyper().max_impact_frac)),
    )


def arrival_index_global(start: int, t0: int) -> int:
    """Bar index in full series for benchmark arrival (prior close before trading)."""
    if t0 > 0:
        return max(0, start + t0 - 1)
    return max(0, start - 1)


def arrival_price_full(price_data: pd.DataFrame, start: int, t0: int) -> float:
    """Decision-time benchmark: previous bar close before the first execution bar."""
    ai = arrival_index_global(start, t0)
    return float(price_data.iloc[ai]["Close"])


def first_exec_bar_index(start: int, t0: int) -> int:
    return int(start + t0)


def effective_horizon(T: int, t0: int) -> int:
    return max(1, int(T) - int(t0))


def dollar_bar_volume(row: pd.Series) -> float:
    c = float(row["Close"])
    v = float(row["Volume"])
    if not np.isfinite(c) or c <= 0:
        return EPS
    if not np.isfinite(v) or v <= 0:
        v = 1.0
    return max(v * c, EPS)


def sell_effective_close(
    base_close: float,
    v_shares: float,
    row: pd.Series,
    hyper: ImpactHyper,
) -> float:
    """Average cash per share when selling ``v_shares`` into bar ``row`` (close benchmark)."""
    base = float(base_close)
    if base <= 0 or v_shares <= 0:
        return max(base, EPS)
    td = abs(float(v_shares)) * base
    dv = dollar_bar_volume(row)
    part = td / dv
    sig = float(row.get("sigma_daily", 0.02))
    if not np.isfinite(sig) or sig <= 0:
        sig = 0.02
    ami = float(row.get("amihud_illiquidity", 0.0))
    if not np.isfinite(ami) or ami < 0:
        ami = 0.0
    sqrt_part = np.sqrt(min(part, hyper.part_cap_sqrt) + EPS)
    impact_frac = hyper.sigma_coef * sig * sqrt_part + hyper.amihud_coef * ami * min(part, hyper.part_cap_sqrt)
    impact_frac = float(np.clip(impact_frac, 0.0, hyper.max_impact_frac))
    return max(base * (1.0 - impact_frac), EPS)
