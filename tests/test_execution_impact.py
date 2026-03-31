from __future__ import annotations

import numpy as np
import pandas as pd

from src.execution_impact import ImpactHyper, sell_effective_close


def test_sell_effective_below_base():
    row = pd.Series(
        {
            "Close": 100.0,
            "Volume": 1e6,
            "sigma_daily": 0.02,
            "amihud_illiquidity": 1e-4,
        }
    )
    px = sell_effective_close(100.0, 5e6, row, ImpactHyper())
    assert px < 100.0
    assert px > 50.0
