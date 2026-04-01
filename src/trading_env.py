"""Gymnasium optimal execution environment."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .execution_impact import (
    arrival_price_full,
    impact_hyper_from_dict,
    sell_effective_close,
)


class OptimalExecutionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_data: pd.DataFrame,
        T: int = 10,
        X_0: float = 1.0,
        eta: float = 0.01,
        gamma: float = 0.001,
        lam: float = 0.22,
        legacy_is_align: float = 0.25,
        resample: bool = True,
        terminal_inventory_penalty: float = 12.0,
        completion_bonus: float = 0.05,
        order_notional_usd: Optional[float] = None,
        order_start_bar: int = 0,
        impact_params: Optional[dict[str, Any]] = None,
        max_inventory_fraction_per_step: Optional[float] = None,
        is_reward_scale: float = 1.0,
        twap_slice_bonus_coef: float = 0.0,
        eval_is_reward_coef: float = 0.0,
        residual_bound: Optional[float] = None,
        relative_is_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.T = int(T)
        self.X_0 = float(X_0)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.legacy_is_align = float(legacy_is_align)
        self.resample = bool(resample)
        self.terminal_inventory_penalty = float(terminal_inventory_penalty)
        self.completion_bonus = float(completion_bonus)
        self.order_notional_usd = float(order_notional_usd) if order_notional_usd is not None else None
        self.order_start_bar = int(order_start_bar)
        self._impact_hyper = impact_hyper_from_dict(impact_params or {})
        self._max_step_frac = (
            float(max_inventory_fraction_per_step)
            if max_inventory_fraction_per_step is not None
            else None
        )
        if self._max_step_frac is not None and not (0.0 < self._max_step_frac <= 1.0):
            raise ValueError("max_inventory_fraction_per_step must be in (0, 1].")
        self.is_reward_scale = float(is_reward_scale)
        if self.is_reward_scale <= 0:
            raise ValueError("is_reward_scale must be positive.")
        self.twap_slice_bonus_coef = float(twap_slice_bonus_coef)
        if self.twap_slice_bonus_coef < 0:
            raise ValueError("twap_slice_bonus_coef must be non-negative.")
        self.eval_is_reward_coef = float(eval_is_reward_coef)
        if self.eval_is_reward_coef < 0:
            raise ValueError("eval_is_reward_coef must be non-negative.")
        self._residual_bound = float(residual_bound) if residual_bound is not None else None
        if self._residual_bound is not None and self._residual_bound <= 0:
            raise ValueError("residual_bound must be positive.")
        self.relative_is_scale = float(relative_is_scale)

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

        if "news_count" not in self.price_data.columns:
            self.price_data["news_count"] = 0.0
        self._news_mean = float(self.price_data["news_count"].astype(float).mean())
        self._news_std = float(self.price_data["news_count"].astype(float).std() or 1.0)

        self._physical = bool(
            self.order_notional_usd is not None and self.order_notional_usd > 0
        )
        t0 = max(0, min(self.order_start_bar, max(self.T - 1, 0)))
        self._t0 = t0

        # [inventory, rem, S_ratio, liq_z, sig_z, regime, pva, twap_gap, news_z]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._row_start = 0
        self._t = 0
        self._X = self.X_0
        self._S0 = 1.0
        self._arrival = 1.0
        self._Q_shares = 1.0
        self._notional_scale = 1.0
        self._ep_is_num = 0.0
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
        pva = (close / max(self._arrival, 1e-12)) - 1.0
        pva = float(np.clip(pva, -0.5, 0.5))
        if self._t < self._t0:
            twap_gap = 0.0
        else:
            n_tr = max(self.T - self._t0, 1)
            i = self._t - self._t0
            twap_target = (n_tr - i) / float(n_tr)
            twap_gap = float(
                np.clip(self._X / self.X_0 - twap_target, -1.0, 1.0)
            )
        nc = float(row["news_count"])
        news_z = (nc - self._news_mean) / (self._news_std + 1e-12)
        news_z = float(np.clip(news_z, -4.0, 4.0))
        return np.array(
            [self._X / self.X_0, rem, S_ratio, liq, sig, r, pva, twap_gap, news_z],
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
        if self._physical:
            self._arrival = float(
                arrival_price_full(self.price_data, self._row_start, self._t0)
            )
            self._Q_shares = float(self.order_notional_usd) / max(self._arrival, 1e-12)
            self._notional_scale = max(self._Q_shares * self._arrival, 1e-12)
        else:
            self._arrival = self._S0
            self._Q_shares = 1.0
            self._notional_scale = 1.0
        self._ep_is_num = 0.0
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

        if self._physical and self._t < self._t0:
            inv_risk = (X_before / self.X_0) ** 2 * (sigma_d**2)
            rem0 = (self.T - self._t) / max(self.T, 1)
            self._t += 1
            terminated = bool(self._X <= 1e-6 or self._t >= self.T)
            completed = bool(self._X <= 1e-6)
            reward = -self.lam * inv_risk * (rem0**1.5)
            if terminated and not completed:
                reward -= self.terminal_inventory_penalty * (self._X / self.X_0) ** 2
            elif completed:
                reward += self.completion_bonus
            reward = self._maybe_eval_is_terminal_bonus(reward, terminated, completed)
            info = {
                "execution_cost": 0.0,
                "inventory_risk": float(inv_risk),
                "regime": regime,
                "v_t": 0.0,
                "v_shares": 0.0,
                "exec_price": float(base_S),
                "completed": completed,
                "twap_slice_bonus": 0.0,
            }
            return self._obs(), float(reward), terminated, False, info

        residual_delta = 0.0
        if self._residual_bound is not None:
            T_eff_r = max(self.T - self._t0, 1)
            bars_left = max(T_eff_r - (self._t - self._t0), 1)
            twap_frac_of_x0 = (X_before / max(self.X_0, 1e-12)) / bars_left
            residual_delta = (a - 0.5) * 2.0 * self._residual_bound
            target_frac = np.clip(
                twap_frac_of_x0 + residual_delta * (X_before / max(self.X_0, 1e-12)),
                0.0,
                X_before / max(self.X_0, 1e-12),
            )
            raw_v = float(target_frac) * self.X_0
        else:
            raw_v = min(a * X_before, X_before)
        if self._max_step_frac is not None:
            v_t = min(raw_v, self._max_step_frac * self.X_0)
        else:
            v_t = raw_v
        inv_risk = (X_before / self.X_0) ** 2 * (sigma_d**2)
        rem = (self.T - self._t) / max(self.T, 1)
        twap_delta = 0.0

        if self._physical:
            v_shares = (v_t / max(self.X_0, 1e-12)) * self._Q_shares
            exec_price = sell_effective_close(base_S, v_shares, row, self._impact_hyper)
            self._ep_is_num += float(v_shares * (exec_price - self._arrival))
            dollar_cost = v_shares * (self._arrival - exec_price)
            inv_w = rem**1.5
            exec_cost = float(dollar_cost / self._notional_scale)
            impact = float(self._arrival - exec_price)
            act_leg = (v_shares * (exec_price - self._arrival)) / self._notional_scale

            T_eff = max(self.T - self._t0, 1)
            q_twap = self._Q_shares / float(T_eff)
            exec_ref = sell_effective_close(base_S, q_twap, row, self._impact_hyper)
            ref_leg = (q_twap * (exec_ref - self._arrival)) / self._notional_scale

            if self.relative_is_scale > 0 and v_shares > 1e-12:
                abs_weight = 0.2
                reward = (
                    self.relative_is_scale * float(act_leg - ref_leg)
                    - abs_weight * ((dollar_cost / self._notional_scale) * self.is_reward_scale)
                    - self.lam * inv_risk * inv_w
                )
            else:
                reward = (
                    -((dollar_cost / self._notional_scale) * self.is_reward_scale)
                    - self.lam * inv_risk * inv_w
                )

            if self.twap_slice_bonus_coef > 0 and v_shares > 1e-12:
                twap_delta = float(self.twap_slice_bonus_coef * (act_leg - ref_leg))
                reward += twap_delta
        else:
            exec_cost = v_t * (self.eta * v_t + self.gamma * (self.X_0 - X_before))
            inv_w = rem**1.5
            impact = self.eta * v_t + self.gamma * (self.X_0 - X_before)
            exec_price = max(base_S - impact, 1e-12)
            v_shares = v_t
            self._ep_is_num += float(v_t * (exec_price - self._arrival))
            is_edge = (exec_price - self._arrival) / max(self._arrival, 1e-12)
            align = self.legacy_is_align * (v_t / max(self.X_0, 1e-12)) * is_edge
            reward = -(exec_cost + self.lam * inv_risk * inv_w) + align

        self._X = X_before - v_t
        self._t += 1
        terminated = bool(self._X <= 1e-6 or self._t >= self.T)
        completed = bool(self._X <= 1e-6)
        if terminated:
            if not completed:
                reward -= self.terminal_inventory_penalty * (self._X / self.X_0) ** 2
            else:
                reward += self.completion_bonus
        reward = self._maybe_eval_is_terminal_bonus(reward, terminated, completed)
        info = {
            "execution_cost": float(exec_cost),
            "inventory_risk": float(inv_risk),
            "regime": regime,
            "v_t": float(v_t),
            "v_shares": float(v_shares),
            "exec_price": float(exec_price),
            "completed": completed,
            "twap_slice_bonus": float(twap_delta),
            "residual_delta": float(residual_delta),
        }
        return self._obs(), float(reward), terminated, False, info

    def _maybe_eval_is_terminal_bonus(
        self, reward: float, terminated: bool, completed: bool
    ) -> float:
        """Terminal bonus aligned with ``evaluate_agent`` IS numerator (optional)."""
        if (
            not terminated
            or self.eval_is_reward_coef <= 0.0
        ):
            return float(reward)
        tot = float(self._ep_is_num)
        if not completed:
            rel = max(min(self._t, self.T) - 1, 0)
            last_px = float(self.price_data.iloc[self._row_start + rel]["Close"])
            x_final = self._X
            if self._physical:
                x_out = (x_final / max(self.X_0, 1e-12)) * float(self._Q_shares)
            else:
                x_out = x_final
            tot += float(x_out * (last_px - self._arrival))
        if self._physical:
            frac = tot / max(float(self._notional_scale), 1e-12)
        else:
            frac = (tot / max(self.X_0, 1e-12)) / max(self._arrival, 1e-12)
        bonus = self.eval_is_reward_coef * float(np.clip(frac, -0.15, 0.15))
        return float(reward) + bonus


def physical_institutional_kwargs(
    order_notional_usd: Optional[float],
    *,
    no_per_step_cap: bool = False,
    max_inventory_fraction_per_step: Optional[float] = None,
    is_reward_scale: Optional[float] = None,
    twap_slice_bonus_coef: Optional[float] = None,
    terminal_inventory_penalty: Optional[float] = None,
    lam: Optional[float] = None,
    residual_bound: Optional[float] = None,
    relative_is_scale: Optional[float] = None,
) -> dict[str, Any]:
    """Default knobs when training/evaluating with USD notional (participation model)."""
    on = order_notional_usd is not None and float(order_notional_usd) > 0
    if not on:
        return {}
    out: dict[str, Any] = {}
    if no_per_step_cap:
        out["max_inventory_fraction_per_step"] = None
    else:
        out["max_inventory_fraction_per_step"] = (
            float(max_inventory_fraction_per_step)
            if max_inventory_fraction_per_step is not None
            else 0.25
        )
    out["is_reward_scale"] = (
        float(is_reward_scale) if is_reward_scale is not None else 1.28
    )
    out["twap_slice_bonus_coef"] = (
        float(twap_slice_bonus_coef) if twap_slice_bonus_coef is not None else 0.60
    )
    out["terminal_inventory_penalty"] = (
        float(terminal_inventory_penalty)
        if terminal_inventory_penalty is not None
        else 5.0
    )
    if lam is not None:
        out["lam"] = float(lam)
    if residual_bound is not None:
        out["residual_bound"] = float(residual_bound)
    if relative_is_scale is not None:
        out["relative_is_scale"] = float(relative_is_scale)
    return out
