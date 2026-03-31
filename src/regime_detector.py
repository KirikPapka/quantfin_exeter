"""Gaussian HMM + vol fallback; optional 3rd feature ``order_imbalance_daily``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class VolatilityThresholdClassifier:
    def __init__(self, threshold: float = 0.24) -> None:
        self.threshold = threshold

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        rv = features["realised_vol_20"].to_numpy(dtype=float)
        return (rv >= self.threshold).astype(np.int64)


class RegimeDetector:
    def __init__(self, n_components: int = 2, fallback_threshold: float = 0.24) -> None:
        self.n_components = int(n_components)
        self.fallback_threshold = float(fallback_threshold)
        self._scaler = StandardScaler()
        self._hmm: Optional[GaussianHMM] = None
        self.use_fallback = False
        self._fallback = VolatilityThresholdClassifier(threshold=fallback_threshold)
        self._feature_cols: list[str] = []

    def _feature_columns(self, features: pd.DataFrame) -> list[str]:
        cols = ["realised_vol_20", "volume_to_spread"]
        if "order_imbalance_daily" in features.columns:
            cols.append("order_imbalance_daily")
        return cols

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        self._feature_cols = self._feature_columns(features)
        missing = [c for c in self._feature_cols if c not in features.columns]
        if missing:
            raise ValueError(f"features missing: {missing}")

        Xraw = features[self._feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(Xraw) < self.n_components * 20:
            logger.warning("Too few rows for HMM; using fallback.")
            self.use_fallback = True
            return self

        vts = np.log1p(np.clip(Xraw["volume_to_spread"].to_numpy(), 0, None))
        blocks = [Xraw["realised_vol_20"].to_numpy(dtype=float), vts]
        if "order_imbalance_daily" in self._feature_cols:
            blocks.append(Xraw["order_imbalance_daily"].to_numpy(dtype=float))
        raw = np.column_stack(blocks)
        Z = self._scaler.fit_transform(raw)

        self._hmm = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        try:
            self._hmm.fit(Z)
        except Exception as e:  # noqa: BLE001
            logger.warning("HMM fit failed (%s); fallback.", e)
            self.use_fallback = True
            return self

        states = self._hmm.predict(Z)
        counts = np.bincount(states, minlength=self.n_components)
        frac = counts / max(len(states), 1)
        if (frac < 0.05).any():
            logger.warning("Degenerate HMM (min state share %.3f); fallback.", float(frac.min()))
            self.use_fallback = True
            return self

        means = self._hmm.means_[:, 0]
        order = np.argsort(means)
        self._state_permutation = {int(o): int(i) for i, o in enumerate(order)}
        self.use_fallback = False
        logger.info("HMM OK (%s states, %s features).", self.n_components, len(self._feature_cols))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.use_fallback:
            return self._fallback.predict(features)

        cols = self._feature_cols
        X = features[cols].replace([np.inf, -np.inf], np.nan)
        mask = X.notna().all(axis=1)
        out = np.zeros(len(features), dtype=np.int64)
        if not mask.any():
            return out

        sub = X.loc[mask]
        vts = np.log1p(np.clip(sub["volume_to_spread"].to_numpy(), 0, None))
        parts = [sub["realised_vol_20"].to_numpy(dtype=float), vts]
        if len(cols) == 3:
            parts.append(sub["order_imbalance_daily"].to_numpy(dtype=float))
        raw = np.column_stack(parts)
        Z = self._scaler.transform(raw)
        hid = self._hmm.predict(Z)  # type: ignore[union-attr]
        remapped = np.array([self._state_permutation[int(s)] for s in hid], dtype=np.int64)
        out[np.where(mask.to_numpy())[0]] = remapped
        return out

    def plot_regimes(
        self,
        price: pd.Series,
        regimes: np.ndarray,
        save_path: str,
    ) -> None:
        colors = ("#cfead8", "#ffe9b3", "#f8bcbc")
        fig, ax = plt.subplots(figsize=(12, 4))
        t = pd.to_datetime(price.index, utc=False)
        x = np.arange(len(t))
        ax.plot(x, price.to_numpy(), color="#1f3a5f", linewidth=1.2, label="Price")
        rmax = int(regimes.max()) if len(regimes) else 0
        for r in range(rmax + 1):
            idx = np.where(regimes == r)[0]
            if len(idx) == 0:
                continue
            for block in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
                ax.axvspan(
                    block[0] - 0.5,
                    block[-1] + 0.5,
                    color=colors[min(r, len(colors) - 1)],
                    alpha=0.35,
                )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylabel("Price")
        ax.set_title("Price with regimes")
        fig.tight_layout()
        Path(save_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def regime_sanity_summary(features: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
    s = pd.Series(regimes, index=features.index, name="regime")
    df = features[["realised_vol_20"]].copy()
    df["regime"] = s
    g = df.dropna().groupby("regime")["realised_vol_20"]
    return pd.DataFrame(
        {"count": g.size(), "mean_vol": g.mean(), "freq": g.size() / max(len(df.dropna()), 1)}
    )
