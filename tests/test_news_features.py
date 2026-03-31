from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.news_features import merge_news_daily


def test_merge_news_daily_aligns_index():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    panel = pd.DataFrame(
        {
            "Close": np.arange(5, dtype=float) + 100,
            "Volume": np.full(5, 1e6),
            "realised_vol_20": np.full(5, 0.2),
            "sigma_daily": np.full(5, 0.02),
            "amihud_illiquidity": np.full(5, 1e-5),
            "regime": np.zeros(5, dtype=np.int64),
        },
        index=idx,
    )
    nidx = pd.to_datetime(["2020-01-01", "2020-01-03"])
    news = pd.DataFrame({"news_count": [3.0, 7.0]}, index=nidx)
    news.index.name = "date"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "n.parquet"
        news.to_parquet(p)
        out = merge_news_daily(panel, p)
    assert list(out["news_count"]) == [3.0, 0.0, 7.0, 0.0, 0.0]
