"""Synthetic daily panels for controlled up / down / flat price experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ScenarioKind = Literal["flat", "up", "down"]


def append_synthetic_scenarios(
    train_df: pd.DataFrame, kinds_csv: str, n_bars: int
) -> pd.DataFrame:
    """Concatenate ``synthetic_panel`` strips onto ``train_df`` (comma-separated kinds)."""
    if not kinds_csv.strip():
        return train_df
    parts: list[pd.DataFrame] = [train_df.reset_index(drop=True)]
    for raw in kinds_csv.split(","):
        k = raw.strip().lower()
        if not k:
            continue
        if k not in ("flat", "up", "down"):
            raise ValueError(f"unknown synthetic kind {raw!r} (expected flat, up, down)")
        sp = synthetic_panel(k, n=int(n_bars)).reset_index(drop=True)
        parts.append(sp)
    return pd.concat(parts, axis=0, ignore_index=True)


def synthetic_panel(
    kind: ScenarioKind,
    n: int = 80,
    *,
    base_close: float = 100.0,
    step: float = 0.2,
    volume: float = 5e6,
) -> pd.DataFrame:
    """Build a reproducible CRSP-style panel with monotonic or flat ``Close``.

    - **flat**: constant close.
    - **up**: close increases by ``step`` each day.
    - **down**: close decreases by ``step`` each day.

    All rows include columns required by ``OptimalExecutionEnv`` and benchmarks.
    """
    if n < 15:
        raise ValueError("n should be at least 15 for T=10 and arrival lookback.")
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    t = np.arange(n, dtype=float)
    if kind == "flat":
        close = np.full(n, base_close, dtype=float)
    elif kind == "up":
        close = base_close + t * step
    elif kind == "down":
        close = base_close - t * step
    else:
        raise ValueError(kind)
    close = np.maximum(close, 1.0)
    vol = np.full(n, volume, dtype=float)
    rv = np.full(n, 0.2, dtype=float)
    sig = rv / np.sqrt(252.0)
    ami = np.full(n, 1e-5, dtype=float)
    return pd.DataFrame(
        {
            "Close": close,
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Volume": vol,
            "realised_vol_20": rv,
            "sigma_daily": sig,
            "amihud_illiquidity": ami,
            "regime": np.zeros(n, dtype=np.int64),
            "news_count": np.zeros(n, dtype=float),
        },
        index=idx,
    )
