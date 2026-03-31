"""BBO aggregation + merge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.bbo_pipeline import aggregate_bbo_csv_to_daily, merge_bbo_into_features


def test_aggregate_and_merge(tmp_path: Path) -> None:
    csv = tmp_path / "bbo.csv"
    csv.write_text(
        "ts_recv,ts_event,rtype,publisher_id,instrument_id,side,price,size,flags,sequence,"
        "bid_px_00,ask_px_00,bid_sz_00,ask_sz_00,bid_ct_00,ask_ct_00,symbol\n"
        "2019-01-02T14:31:00.000000000Z,,1,2,1,A,1,1,0,1,100,100.5,10,30,1,1,TEST\n"
        "2019-01-02T14:32:00.000000000Z,,1,2,1,A,1,1,0,1,100,100.5,20,20,1,1,TEST\n"
    )
    daily = aggregate_bbo_csv_to_daily(csv, symbols=["TEST"], chunksize=10_000)
    assert len(daily) == 1
    assert daily.iloc[0]["symbol"] == "TEST"
    assert "order_imbalance_daily" in daily.columns

    feat = pd.DataFrame(
        {"Close": [100.0], "realised_vol_20": [0.2], "volume_to_spread": [1e6]},
        index=pd.DatetimeIndex([pd.Timestamp("2019-01-02")], name="date"),
    )
    merged = merge_bbo_into_features(feat, daily, ticker="TEST", fill_missing_obi=0.0)
    assert "order_imbalance_daily" in merged.columns
    assert merged.iloc[0]["order_imbalance_daily"] != 0.0 or daily.iloc[0]["n_bbo_bars"] == 0
