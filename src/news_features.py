"""Optional daily news counts (Finnhub) merged into feature panels."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def merge_news_daily(panel: pd.DataFrame, news_parquet: Path) -> pd.DataFrame:
    """Left-join ``news_count`` onto ``panel`` index (trading dates)."""
    path = Path(news_parquet)
    if not path.is_file():
        raise FileNotFoundError(path)
    nd = pd.read_parquet(path)
    if nd.index.name != "date" and "date" in nd.columns:
        nd = nd.set_index(pd.to_datetime(nd["date"], utc=False))
        if getattr(nd.index.dtype, "tz", None) is not None:
            nd.index = nd.index.tz_localize(None)
        nd.index.name = "date"
    nd = nd.sort_index()
    if "news_count" not in nd.columns:
        raise ValueError(f"{path} must contain column 'news_count'")
    counts = nd["news_count"].astype(float).reindex(panel.index).fillna(0.0)
    out = panel.copy()
    out["news_count"] = counts
    logger.info("Merged news_count from %s (nonzero days: %s)", path.name, int((counts > 0).sum()))
    return out
