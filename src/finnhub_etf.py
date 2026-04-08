"""Finnhub ETF holdings + fallback weights for constituent-level news aggregation."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

logger = logging.getLogger(__name__)

FINNHUB_ETF_HOLDINGS = "https://finnhub.io/api/v1/stock/etf/holdings"

# Approximate large-cap slice when API is unavailable (renormalize after top-N trim).
_FALLBACK_SPY = [
    ("NVDA", 0.072),
    ("AAPL", 0.065),
    ("MSFT", 0.060),
    ("AMZN", 0.038),
    ("META", 0.028),
    ("GOOGL", 0.021),
    ("AVGO", 0.025),
    ("BRK.B", 0.016),
    ("TSLA", 0.014),
    ("JPM", 0.013),
    ("UNH", 0.011),
    ("V", 0.010),
    ("XOM", 0.009),
    ("PG", 0.009),
    ("JNJ", 0.009),
    ("MA", 0.009),
    ("HD", 0.008),
    ("COST", 0.008),
    ("ABBV", 0.007),
    ("MRK", 0.007),
]


def fetch_etf_holdings(
    etf_symbol: str,
    token: str,
    *,
    max_retries: int = 6,
) -> list[tuple[str, float]]:
    """Return ``(constituent_symbol, weight)`` with weights in (0, 1], newest snapshot."""
    q = urlencode({"symbol": etf_symbol.upper(), "token": token})
    url = f"{FINNHUB_ETF_HOLDINGS}?{q}"
    for attempt in range(max_retries):
        try:
            req = urlopen(url, timeout=60)
            raw = json.loads(req.read().decode("utf-8"))
            if not isinstance(raw, dict):
                logger.warning("Unexpected ETF holdings JSON type — using fallback slice")
                return list(_FALLBACK_SPY)
            out = _parse_holdings_payload(raw)
            if out:
                return out
            logger.warning("ETF holdings empty for %s — using fallback slice", etf_symbol)
            return list(_FALLBACK_SPY)
        except HTTPError as e:
            if e.code == 429 and attempt + 1 < max_retries:
                time.sleep(min(120.0, 15.0 * (2**attempt)))
                continue
            logger.warning(
                "ETF holdings HTTP %s for %s (%s) — using fallback slice",
                e.code,
                etf_symbol,
                getattr(e, "reason", ""),
            )
            return list(_FALLBACK_SPY)
        except (URLError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("ETF holdings error for %s: %s — using fallback slice", etf_symbol, e)
            return list(_FALLBACK_SPY)
    logger.warning("ETF holdings retries exhausted for %s — using fallback slice", etf_symbol)
    return list(_FALLBACK_SPY)


def _parse_holdings_payload(raw: dict[str, Any]) -> list[tuple[str, float]]:
    rows = raw.get("holdings")
    if not isinstance(rows, list):
        return []
    out: list[tuple[str, float]] = []
    for h in rows:
        if not isinstance(h, dict):
            continue
        sym = h.get("symbol") or h.get("asset")
        if not sym:
            continue
        sym = str(sym).strip().upper()
        w = h.get("weight")
        if w is None:
            w = h.get("percentage")
        try:
            wf = float(w)
        except (TypeError, ValueError):
            continue
        if wf > 1.0 + 1e-6:
            wf /= 100.0
        if wf <= 0:
            continue
        out.append((sym, wf))
    return out


def top_holdings_renormalized(
    holdings: list[tuple[str, float]],
    max_names: int,
) -> list[tuple[str, float]]:
    """Keep largest weights and renormalize to sum to 1 over the kept set."""
    if not holdings:
        return []
    ranked = sorted(holdings, key=lambda x: -x[1])[: max(1, int(max_names))]
    s = sum(w for _, w in ranked)
    if s <= 0:
        return []
    return [(sym, w / s) for sym, w in ranked]
