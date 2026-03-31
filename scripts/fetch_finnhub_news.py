#!/usr/bin/env python3
"""Download company news counts from Finnhub (free API) into a daily parquet.

Register at https://finnhub.io/register for a token. Set ``FINNHUB_API_KEY`` in ``.env``.

The free tier is strict: stay near **1 request / second** and retry **429** with backoff.
News is aggregated
to **calendar-day article counts** aligned to your features index after merge in ``load_split``.

For **ETFs** (e.g. SPY), direct company-news is often empty historically. Use ``--etf-proxy`` to
pull **ETF holdings** (API or fallback slice), take the top **N** names by weight, **renormalize**
weights, then set ``news_count`` = Σ weight\\ :sub:`i` × (articles for name *i* that day).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.finnhub_etf import fetch_etf_holdings, top_holdings_renormalized
from src.utils import default_data_root, project_root

logger = logging.getLogger(__name__)

FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"


def _day_chunks(start_d: date, end_d: date, chunk_days: int = 365) -> list[tuple[str, str]]:
    """Fewer, wider chunks ⇒ fewer HTTP calls (important for free-tier limits)."""
    out: list[tuple[str, str]] = []
    d = start_d
    while d <= end_d:
        d2 = min(d + timedelta(days=chunk_days - 1), end_d)
        out.append((d.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")))
        d = d2 + timedelta(days=1)
    return out


def fetch_company_news(
    symbol: str,
    date_from: str,
    date_to: str,
    token: str,
    *,
    max_retries: int = 8,
) -> list[dict]:
    q = urlencode(
        {"symbol": symbol.upper(), "from": date_from, "to": date_to, "token": token}
    )
    url = f"{FINNHUB_NEWS}?{q}"
    last_err: HTTPError | None = None
    for attempt in range(max_retries):
        try:
            req = urlopen(url, timeout=90)
            data = req.read().decode("utf-8")
            return json.loads(data)
        except HTTPError as e:
            last_err = e
            if e.code == 429 and attempt + 1 < max_retries:
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(ra) if ra is not None else 0.0
                except (TypeError, ValueError):
                    wait = 0.0
                if wait <= 0:
                    wait = min(120.0, 15.0 * (2**attempt))
                logger.warning(
                    "429 Too Many Requests for %s..%s — sleeping %.0fs (retry %s/%s)",
                    date_from,
                    date_to,
                    wait,
                    attempt + 2,
                    max_retries,
                )
                time.sleep(wait)
                continue
            raise
    assert last_err is not None
    raise last_err


def aggregate_daily(items: list[dict]) -> dict[str, int]:
    by_day: dict[str, int] = {}
    for it in items:
        ts = it.get("datetime")
        if ts is None:
            continue
        try:
            dt = datetime.utcfromtimestamp(int(ts))
        except (TypeError, ValueError, OSError):
            continue
        day = dt.strftime("%Y-%m-%d")
        by_day[day] = by_day.get(day, 0) + 1
    return by_day


def fetch_weighted_constituent_news(
    constituents: list[tuple[str, float]],
    ranges: list[tuple[str, str]],
    token: str,
    sleep: float,
) -> dict[str, float]:
    """Σ w_i * article_count_i(day); one HTTP call per (symbol, chunk)."""
    acc: dict[str, float] = {}
    n_req = len(constituents) * len(ranges)
    done = 0
    for sym, w in constituents:
        for d0, d1 in ranges:
            try:
                items = fetch_company_news(sym, d0, d1, token)
            except HTTPError as e:
                logger.error("HTTP %s for %s %s..%s: %s", e.code, sym, d0, d1, e.reason)
                raise
            except URLError as e:
                logger.error("Network error %s %s..%s: %s", sym, d0, d1, e.reason)
                raise
            chunk = aggregate_daily(items if isinstance(items, list) else [])
            for day, c in chunk.items():
                acc[day] = acc.get(day, 0.0) + w * float(c)
            done += 1
            if done % max(1, n_req // 20) == 0 or done == n_req:
                logger.info("ETF proxy progress %s/%s HTTP calls", done, n_req)
            time.sleep(max(0.0, float(sleep)))
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="SPY", help="Ticker (e.g. SPY, AAPL)")
    ap.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="YYYY-MM-DD start (default: min date from features_train if available)",
    )
    ap.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="YYYY-MM-DD end (default: today UTC)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output parquet path (default: data/processed/news_daily_{SYMBOL}.parquet)",
    )
    ap.add_argument(
        "--chunk-days",
        type=int,
        default=365,
        help="Days per API request (larger ⇒ fewer calls; default 365).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.1,
        help="Seconds to wait after each successful request (free tier: stay ~≥1 s).",
    )
    ap.add_argument(
        "--etf-proxy",
        action="store_true",
        help="For an ETF --symbol (e.g. SPY): aggregate weighted company-news of top holdings.",
    )
    ap.add_argument(
        "--max-constituents",
        type=int,
        default=20,
        help="With --etf-proxy: number of largest holdings to include (renormalized weights).",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    token = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not token:
        raise SystemExit(
            "Set FINNHUB_API_KEY in .env (get a free key at https://finnhub.io/register)"
        )

    sym = args.symbol.upper()
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * 10)

    if args.to_date:
        end = datetime.strptime(args.to_date, "%Y-%m-%d").date()
    if args.from_date:
        start = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    else:
        train_parquet = default_data_root() / "features" / "features_train.parquet"
        if train_parquet.is_file():
            import pandas as pd

            df = pd.read_parquet(train_parquet)
            if "ticker" in df.columns and sym in set(df["ticker"].astype(str).unique()):
                df = df.loc[df["ticker"].astype(str) == sym]
            if "date" in df.columns:
                d = pd.to_datetime(df["date"], utc=False)
                start = d.min().date()
                logger.info("Using train parquet min date: %s", start)

    out_path = (
        Path(args.out)
        if args.out
        else project_root() / "data" / "processed" / f"news_daily_{sym}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_by_day: dict[str, float] = {}
    chunk_days = max(7, int(args.chunk_days))
    ranges = _day_chunks(start, end, chunk_days=chunk_days)

    if args.etf_proxy:
        raw_h = fetch_etf_holdings(sym, token)
        constituents = top_holdings_renormalized(raw_h, int(args.max_constituents))
        logger.info(
            "ETF proxy %s: %s names (e.g. %s)",
            sym,
            len(constituents),
            ", ".join(f"{s}:{w:.3f}" for s, w in constituents[:5]),
        )
        est_calls = len(constituents) * len(ranges)
        logger.info(
            "~%s HTTP calls from %s to %s (~%.0fs minimum pacing)",
            est_calls,
            start,
            end,
            est_calls * float(args.sleep),
        )
        try:
            total_by_day = fetch_weighted_constituent_news(constituents, ranges, token, args.sleep)
        except (HTTPError, URLError):
            raise SystemExit(1)
    else:
        logger.info(
            "Fetching %s chunks from %s to %s (~%.0fs minimum pacing)",
            len(ranges),
            start,
            end,
            len(ranges) * float(args.sleep),
        )
        for i, (d0, d1) in enumerate(ranges):
            try:
                items = fetch_company_news(sym, d0, d1, token)
            except HTTPError as e:
                logger.error("HTTP %s for %s..%s: %s", e.code, d0, d1, e.reason)
                raise SystemExit(1) from e
            except URLError as e:
                logger.error("Network error %s..%s: %s", d0, d1, e.reason)
                raise SystemExit(1) from e
            chunk = aggregate_daily(items if isinstance(items, list) else [])
            for k, v in chunk.items():
                total_by_day[k] = total_by_day.get(k, 0.0) + float(v)
            logger.info(
                "Chunk %s/%s %s..%s → %s article-days",
                i + 1,
                len(ranges),
                d0,
                d1,
                len(chunk),
            )
            time.sleep(max(0.0, float(args.sleep)))

    import pandas as pd

    if not total_by_day:
        logger.warning("No news rows returned — check symbol and date range.")
    idx = pd.to_datetime(sorted(total_by_day.keys()))
    ser = pd.Series([total_by_day[d.strftime("%Y-%m-%d")] for d in idx], index=idx)
    ser.index.name = "date"
    out_df = ser.to_frame("news_count")
    out_df.to_parquet(out_path)
    logger.info("Wrote %s rows to %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
