#!/usr/bin/env python3
"""
Build ``data/processed/bbo_daily.parquet`` from NASDAQ ITCH BBO 1-minute CSV.

BBO coverage begins **2019-01-02** (not 2018). Earlier CRSP dates get ``order_imbalance_daily=0``
when merged (neutral fill).

Usage:
    # Optional: override data root (defaults to ./deploy_data if present)
    export CFA_DATA_ROOT=/path/to/deploy_data
  python scripts/build_bbo_daily.py
  python scripts/build_bbo_daily.py --symbols SPY AAPL
  python scripts/build_bbo_daily.py --csv /path/to/xnas-itch-...bbo-1m.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bbo_pipeline import (  # noqa: E402
    aggregate_bbo_csv_to_daily,
    default_bbo_csv_path,
    save_bbo_daily,
)
from src.utils import default_data_root, setup_logging


def main() -> None:
    setup_logging()
    log = logging.getLogger(__name__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="BBO CSV path")
    ap.add_argument(
        "--symbols",
        nargs="*",
        default=["SPY", "AAPL"],
        help="Symbols to keep (reduces runtime and file size)",
    )
    ap.add_argument("--chunksize", type=int, default=400_000)
    args = ap.parse_args()

    data_root = default_data_root()
    bbo_path = args.csv or default_bbo_csv_path(data_root)
    log.info("Reading %s", bbo_path)

    daily = aggregate_bbo_csv_to_daily(
        bbo_path,
        symbols=list(args.symbols) if args.symbols else None,
        chunksize=args.chunksize,
    )
    out = save_bbo_daily(daily)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
