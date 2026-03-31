#!/usr/bin/env python3
"""Sample LLM governance calls (see README)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.llm_explainer import explain_execution
from src.utils import regime_display_name


def main() -> None:
    for regime in (0, 1):
        print(
            explain_execution(
                regime=regime,
                regime_name=regime_display_name(regime, 2),
                inventory_remaining=0.6,
                action_taken=0.2,
                execution_cost_bps=4.0,
                twap_cost_bps=5.0,
                ac_cost_bps=4.7,
                sigma_t=0.018,
                liquidity_t=1e-5,
                use_cache=True,
            )
        )


if __name__ == "__main__":
    main()
