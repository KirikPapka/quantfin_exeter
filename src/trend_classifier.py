"""Price trend labels from rolling returns (independent of HMM volatility regime)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trend bucket indices (used in eval reports and switching policy)
TREND_DOWN = 0
TREND_MID = 1
TREND_UP = 2


def _rolling_return(close: pd.Series, lookback: int) -> pd.Series:
    prev = close.shift(int(lookback))
    return close / prev.replace(0.0, np.nan) - 1.0


def classify_return(
    ret: float,
    *,
    up_pct: float = 0.02,
    down_pct: float = -0.02,
) -> int:
    """Map a single lookback return to ``TREND_DOWN`` / ``TREND_MID`` / ``TREND_UP``."""
    if not np.isfinite(ret):
        return TREND_MID
    if ret >= up_pct:
        return TREND_UP
    if ret <= down_pct:
        return TREND_DOWN
    return TREND_MID


def classify_trend_at(
    price_data: pd.DataFrame,
    row_start: int,
    lookback: int = 20,
    up_pct: float = 0.02,
    down_pct: float = -0.02,
) -> int:
    """Trend label at ``row_start`` using ``Close`` lookback return (same row as episode start)."""
    if row_start < 0 or row_start >= len(price_data):
        return TREND_MID
    if "Close" not in price_data.columns:
        return TREND_MID
    close = price_data["Close"].astype(float)
    i = int(row_start)
    j = i - int(lookback)
    if j < 0:
        return TREND_MID
    c_now = float(close.iloc[i])
    c_prev = float(close.iloc[j])
    if not np.isfinite(c_now) or not np.isfinite(c_prev) or c_prev == 0.0:
        return TREND_MID
    ret = c_now / c_prev - 1.0
    return classify_return(ret, up_pct=up_pct, down_pct=down_pct)


def compute_trend_regime(
    price_data: pd.DataFrame,
    lookback: int = 20,
    up_pct: float = 0.02,
    down_pct: float = -0.02,
    *,
    column_name: str = "trend_regime",
) -> pd.DataFrame:
    """Return a copy of ``price_data`` with ``column_name`` (0=down, 1=mid, 2=up)."""
    out = price_data.copy()
    if "Close" not in out.columns:
        out[column_name] = TREND_MID
        return out
    close = out["Close"].astype(float)
    rr = _rolling_return(close, lookback)
    labels = np.full(len(out), TREND_MID, dtype=np.int64)
    valid = rr.notna().to_numpy()
    vals = rr.to_numpy()
    for k in np.where(valid)[0]:
        labels[k] = classify_return(float(vals[k]), up_pct=up_pct, down_pct=down_pct)
    out[column_name] = labels
    return out
