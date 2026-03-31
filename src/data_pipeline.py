"""Ingestion, CRSP-style parquet, optional BBO merge."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .bbo_pipeline import load_bbo_daily, merge_bbo_into_features
from .utils import default_data_root, project_root

logger = logging.getLogger(__name__)
EPS = 1e-8


def _ensure_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], utc=False)
        if getattr(out[date_col].dtype, "tz", None) is not None:
            out[date_col] = out[date_col].dt.tz_localize(None)
        out = out.set_index(date_col).sort_index()
    out.index.name = "date"
    return out


def load_features_parquet(path: Path, ticker: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if ticker is not None and "ticker" in df.columns:
        df = df.loc[df["ticker"].astype(str) == ticker].copy()
    out = _ensure_datetime_index(df, "date")
    close = out["prc"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    vol = out["vol"].astype(float).replace(0, np.nan)
    amihud = (
        out["amihud_20d"]
        if "amihud_20d" in out.columns
        else out.get("amihud_daily", pd.Series(np.nan, index=out.index))
    ).astype(float)
    rv = out["real_vol_20d"].astype(float)
    vix_s = out["vix"].astype(float) if "vix" in out.columns else pd.Series(np.nan, index=out.index)
    bid_ask_proxy = (high - low) / (close + EPS)
    volume_to_spread = vol / (bid_ask_proxy + EPS)
    panel = pd.DataFrame(
        {
            "Close": close,
            "Volume": vol,
            "realised_vol_20": rv,
            "sigma_daily": rv / np.sqrt(252.0),
            "amihud_illiquidity": amihud.fillna(0.0),
            "bid_ask_proxy": bid_ask_proxy,
            "volume_to_spread": volume_to_spread.replace([np.inf, -np.inf], np.nan),
            "vix_aligned": vix_s.ffill(),
        }
    )
    panel = panel.dropna(subset=["Close", "realised_vol_20"])
    logger.info(
        "Loaded %s: %s rows %s → %s",
        path.name,
        len(panel),
        panel.index.min().date(),
        panel.index.max().date(),
    )
    return panel


def load_split(
    split: str,
    *,
    data_root: Optional[Path] = None,
    ticker: Optional[str] = "SPY",
    use_bbo: bool = True,
    bbo_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    data_root = data_root or default_data_root()
    path = data_root / "features" / f"features_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    panel = load_features_parquet(path, ticker=ticker)
    if use_bbo and ticker:
        try:
            bp = bbo_parquet or (project_root() / "data" / "processed" / "bbo_daily.parquet")
            bbo = load_bbo_daily(bp)
            panel = merge_bbo_into_features(panel, bbo, ticker=ticker, fill_missing_obi=0.0)
            logger.info("Merged BBO order imbalance for %s", ticker)
        except FileNotFoundError:
            logger.warning(
                "No BBO parquet at %s — run: python scripts/build_bbo_daily.py",
                project_root() / "data" / "processed" / "bbo_daily.parquet",
            )
    return panel
