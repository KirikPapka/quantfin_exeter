"""Gymnasium optimal execution environment."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class OptimalExecutionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_data: pd.DataFrame,
        T: int = 10,
        X_0: float = 1.0,
        eta: float = 0.01,
        gamma: float = 0.001,
        lam: float = 0.5,
        resample: bool = True,
        terminal_inventory_penalty: float = 12.0,
        completion_bonus: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.T = int(T)
        self.X_0 = float(X_0)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.resample = bool(resample)
        self.terminal_inventory_penalty = float(terminal_inventory_penalty)
        self.completion_bonus = float(completion_bonus)

        required = {"Close", "amihud_illiquidity", "regime"}
        missing = required - set(price_data.columns)
        if missing:
            raise ValueError(f"price_data missing: {sorted(missing)}")

        self.price_data = price_data.dropna(subset=["Close"]).copy()
        if "sigma_daily" not in self.price_data.columns:
            self.price_data["sigma_daily"] = (
                self.price_data["realised_vol_20"].astype(float) / np.sqrt(252.0)
            )

        self._liq_mean = float(self.price_data["amihud_illiquidity"].mean())
        self._liq_std = float(self.price_data["amihud_illiquidity"].std() or 1.0)
        self._sig_mean = float(self.price_data["sigma_daily"].mean())
        self._sig_std = float(self.price_data["sigma_daily"].std() or 1.0)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._row_start = 0
        self._t = 0
        self._X = self.X_0
        self._S0 = 1.0
        self._arrival = 1.0
        if seed is not None:
            super().reset(seed=seed)

    def _obs(self) -> np.ndarray:
        rel = min(self._t, self.T - 1)
        row = self.price_data.iloc[self._row_start + rel]
        close = float(row["Close"])
        S_ratio = close / self._S0 if self._S0 else 1.0
        liq = (float(row["amihud_illiquidity"]) - self._liq_mean) / (self._liq_std + 1e-12)
        sig = (float(row["sigma_daily"]) - self._sig_mean) / (self._sig_std + 1e-12)
        rem = (self.T - self._t) / max(self.T, 1)
        r = float(row["regime"])
        return np.array(
            [self._X / self.X_0, rem, S_ratio, liq, sig, r],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        n = len(self.price_data)
        max_start = n - self.T - 1
        if max_start < 0:
            raise ValueError("price_data too short.")
        self._row_start = int(self.np_random.integers(0, max_start + 1)) if self.resample else 0
        self._t = 0
        self._X = self.X_0
        self._S0 = float(self.price_data.iloc[self._row_start]["Close"])
        self._arrival = self._S0
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = float(np.clip(action[0], 0.0, 1.0))
        row = self.price_data.iloc[self._row_start + self._t]
        base_S = float(row["Close"])
        sigma_d = float(row["sigma_daily"])
        regime = int(row["regime"])
        X_before = self._X
        v_t = min(a * X_before, X_before)
        inv_risk = (X_before / self.X_0) ** 2 * (sigma_d**2)
        exec_cost = v_t * (self.eta * v_t + self.gamma * (self.X_0 - X_before))
        reward = -(exec_cost + self.lam * inv_risk)
        impact = self.eta * v_t + self.gamma * (self.X_0 - X_before)
        exec_price = max(base_S - impact, 1e-12)
        self._X = X_before - v_t
        self._t += 1
        terminated = bool(self._X <= 1e-6 or self._t >= self.T)
        completed = bool(self._X <= 1e-6)
        if terminated:
            if not completed:
                reward -= self.terminal_inventory_penalty * (self._X / self.X_0) ** 2
            else:
                reward += self.completion_bonus
        info = {
            "execution_cost": float(exec_cost),
            "inventory_risk": float(inv_risk),
            "regime": regime,
            "v_t": float(v_t),
            "exec_price": float(exec_price),
            "completed": completed,
        }
        return self._obs(), float(reward), terminated, False, info
