"""
NASDAQ ITCH-derived **BBO (Level-1) 1-minute** pipeline.

This is *not* full LOBSTER depth (multi-level LOB); it ingests top-of-book
``bid_sz_00`` / ``ask_sz_00`` bars and builds **daily order imbalance** features.

Coverage starts **2019-01-02** (first timestamp in typical XNAS BBO dumps), so
dates before that have no BBO rows — callers should ``merge`` and ``fillna`` as
documented.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .utils import default_data_root, project_root

logger = logging.getLogger(__name__)

EPS = 1e-9

DEFAULT_BBO_GLOB = "raw/XNAS-*/xnas-itch-*bbo-1m.csv"


def default_bbo_csv_path(data_root: Optional[Path] = None) -> Path:
    """First matching BBO CSV under ``CFADATA/raw/XNAS-*/``."""
    root = data_root or default_data_root()
    matches = sorted(root.glob(DEFAULT_BBO_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No BBO CSV matching {root}/{DEFAULT_BBO_GLOB}. "
            "Set CFA_DATA_ROOT or pass bbo_csv_path explicitly."
        )
    return matches[0]


def _iter_bbo_chunks(
    path: Path,
    *,
    chunksize: int,
    usecols: list[str],
) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False):
        yield chunk


def compute_minute_obi(bid_sz: pd.Series, ask_sz: pd.Series) -> pd.Series:
    """BBO size imbalance in ``[-1, 1]``."""
    b = bid_sz.astype(float).clip(lower=0)
    a = ask_sz.astype(float).clip(lower=0)
    return (b - a) / (b + a + EPS)


def aggregate_bbo_csv_to_daily(
    bbo_csv: Path,
    *,
    symbols: Optional[list[str]] = None,
    chunksize: int = 400_000,
    tz_trading: str = "America/New_York",
) -> pd.DataFrame:
    """
    Scan the full BBO CSV in chunks; return daily panel.

    Columns: ``date`` (datetime64[ns] normalized), ``symbol``, ``order_imbalance_daily``
    (mean OBI over 1m bars that day in ``tz_trading``), ``n_bbo_bars``, optional
    ``spread_bps_mean`` (quoted spread relative to mid, in bps).

    **Trading calendar:** ``ts_recv`` is parsed as UTC, converted to
    ``tz_trading``, then **calendar date** in that zone defines the row's day
    (aligns with US equity session dates for NASDAQ data).
    """
    usecols = [
        "ts_recv",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "symbol",
    ]
    parts: list[pd.DataFrame] = []

    for chunk in _iter_bbo_chunks(bbo_csv, chunksize=chunksize, usecols=usecols):
        if symbols is not None:
            chunk = chunk.loc[chunk["symbol"].astype(str).isin(symbols)]
        if chunk.empty:
            continue

        ts = pd.to_datetime(chunk["ts_recv"], utc=True).dt.tz_convert(tz_trading)
        chunk = chunk.assign(
            trade_date=ts.dt.normalize(),
        )
        bid = chunk["bid_sz_00"].astype(float)
        ask = chunk["ask_sz_00"].astype(float)
        chunk["obi"] = compute_minute_obi(bid, ask)

        bp = chunk["bid_px_00"].astype(float)
        ap = chunk["ask_px_00"].astype(float)
        mid = (bp + ap) / 2.0
        chunk["spread_bps"] = np.where(
            mid > 0, (ap - bp) / mid * 1e4, np.nan
        )

        g = chunk.groupby(["symbol", "trade_date"], sort=False).agg(
            obi_sum=("obi", "sum"),
            n_bbo_bars=("obi", "size"),
            spread_sum=("spread_bps", lambda s: float(np.nansum(s))),
            spread_n=("spread_bps", lambda s: int(np.sum(np.isfinite(s)))),
        )
        parts.append(g.reset_index())

    if not parts:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "order_imbalance_daily",
                "n_bbo_bars",
                "spread_bps_mean",
            ]
        )

    all_daily = pd.concat(parts, ignore_index=True)
    all_daily = all_daily.groupby(["symbol", "trade_date"], sort=False).agg(
        obi_sum=("obi_sum", "sum"),
        n_bbo_bars=("n_bbo_bars", "sum"),
        spread_sum=("spread_sum", "sum"),
        spread_n=("spread_n", "sum"),
    )
    all_daily = all_daily.reset_index()
    all_daily["order_imbalance_daily"] = all_daily["obi_sum"] / np.maximum(
        all_daily["n_bbo_bars"].astype(float), 1.0
    )
    all_daily["spread_bps_mean"] = np.where(
        all_daily["spread_n"] > 0,
        all_daily["spread_sum"] / all_daily["spread_n"],
        np.nan,
    )
    all_daily = all_daily.drop(
        columns=["obi_sum", "spread_sum", "spread_n"], errors="ignore"
    )
    all_daily = all_daily.rename(columns={"trade_date": "date"})
    all_daily["date"] = pd.to_datetime(all_daily["date"]).dt.tz_localize(None)
    logger.info(
        "BBO daily aggregation: %s symbol-days from %s (first bar file: %s)",
        len(all_daily),
        bbo_csv.name,
        bbo_csv,
    )
    return all_daily


def merge_bbo_into_features(
    features: pd.DataFrame,
    bbo_daily: pd.DataFrame,
    *,
    ticker: str,
    fill_missing_obi: float = 0.0,
) -> pd.DataFrame:
    """
    Left-join daily BBO features onto a single-ticker feature panel.

    ``features`` must have DatetimeIndex (name ``date``) or column ``date``.
    Rows before BBO coverage (e.g. prior to **2019-01-02**) get ``fill_missing_obi``.
    """
    out = features.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    date_col = "date" if "date" in out.columns else out.columns[0]
    out[date_col] = pd.to_datetime(out[date_col]).dt.tz_localize(None).dt.normalize()

    sub = bbo_daily.loc[bbo_daily["symbol"].astype(str) == str(ticker)].copy()
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    sub = sub[["date", "order_imbalance_daily", "n_bbo_bars", "spread_bps_mean"]]

    out = out.merge(sub, on="date", how="left")
    out["order_imbalance_daily"] = out["order_imbalance_daily"].fillna(fill_missing_obi)
    out["n_bbo_bars"] = out["n_bbo_bars"].fillna(0).astype(int)
    out["spread_bps_mean"] = out["spread_bps_mean"].fillna(np.nan)

    out = out.set_index(date_col)
    out.index.name = "date"
    return out


def save_bbo_daily(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (project_root() / "data" / "processed" / "bbo_daily.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %s rows to %s", len(df), path)
    return path


def load_bbo_daily(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (project_root() / "data" / "processed" / "bbo_daily.parquet")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
