from __future__ import annotations

import numpy as np
import pandas as pd

from src.regime_detector import RegimeDetector, VolatilityThresholdClassifier


def _feat(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    vol = np.concatenate(
        [rng.normal(0.15, 0.02, n // 2), rng.normal(0.35, 0.03, n - n // 2)]
    )
    return pd.DataFrame(
        {
            "realised_vol_20": vol,
            "volume_to_spread": rng.lognormal(10, 1, size=n),
            "order_imbalance_daily": rng.uniform(-0.2, 0.2, size=n),
        },
        index=idx,
    )


def test_hmm_with_obi():
    df = _feat(300, 42)
    det = RegimeDetector(n_components=2, fallback_threshold=0.24)
    det.fit(df)
    lab = det.predict(df)
    assert len(lab) == len(df)


def test_fallback_small():
    df = _feat(8, 0)
    det = RegimeDetector(n_components=3, fallback_threshold=0.24)
    det.fit(df)
    assert det.use_fallback


def test_vol_threshold():
    df = pd.DataFrame({"realised_vol_20": [0.1, 0.3]})
    assert list(VolatilityThresholdClassifier(0.24).predict(df)) == [0, 1]
