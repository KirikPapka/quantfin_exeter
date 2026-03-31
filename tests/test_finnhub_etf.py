from __future__ import annotations

from src.finnhub_etf import _parse_holdings_payload, top_holdings_renormalized


def test_top_holdings_renormalized_sums_to_one():
    h = [("A", 0.5), ("B", 0.3), ("C", 0.2)]
    t = top_holdings_renormalized(h, 2)
    assert [x[0] for x in t] == ["A", "B"]
    assert abs(sum(w for _, w in t) - 1.0) < 1e-9


def test_parse_holdings_payload_weight_and_percent():
    raw = {
        "holdings": [
            {"symbol": "AAPL", "weight": 0.05},
            {"symbol": "MSFT", "percentage": 7.0},
        ]
    }
    out = _parse_holdings_payload(raw)
    assert len(out) == 2
    syms = {s: w for s, w in out}
    assert abs(syms["AAPL"] - 0.05) < 1e-9
    assert abs(syms["MSFT"] - 0.07) < 1e-9
