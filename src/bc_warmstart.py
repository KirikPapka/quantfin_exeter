"""Behavioral cloning from TWAP demonstrations for policy warmstart."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .trading_env import OptimalExecutionEnv

logger = logging.getLogger(__name__)


def collect_twap_demos(
    env: OptimalExecutionEnv,
    n_episodes: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out TWAP on ``env`` and record (obs, action) pairs.

    The action value depends on whether the env uses residual mode:
    - Residual mode: action = 0.5 (center = TWAP)
    - Raw mode: action = fraction of current inventory that TWAP would sell
    """
    rng = np.random.default_rng(seed)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []

    residual_mode = env._residual_bound is not None
    t0 = env._t0
    T = env.T

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        while not terminated:
            if residual_mode:
                twap_action = np.array([0.5], dtype=np.float32)
            else:
                T_eff = max(T - t0, 1)
                trading_step = env._t - t0
                if env._t < t0:
                    twap_action = np.array([0.0], dtype=np.float32)
                else:
                    bars_left = max(T_eff - trading_step, 1)
                    frac = 1.0 / bars_left
                    twap_action = np.array([float(np.clip(frac, 0.0, 1.0))], dtype=np.float32)

            obs_list.append(obs.copy())
            act_list.append(twap_action.copy())
            obs, _, term, trunc, _ = env.step(twap_action)
            terminated = bool(term or trunc)

    return np.array(obs_list), np.array(act_list)


def train_bc(
    obs: np.ndarray,
    actions: np.ndarray,
    obs_dim: int = 9,
    act_dim: int = 1,
    hidden: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Train a small MLP to imitate TWAP actions via MSE, return state_dict compatible with SB3.

    Returns the ``policy.state_dict()`` that can be loaded with
    ``model.policy.load_state_dict(result, strict=False)``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device)
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, act_dim),
        nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_obs, batch_act in loader:
            pred = net(batch_obs)
            loss = loss_fn(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("BC epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg)

    bc_state = net.state_dict()

    sb3_state: dict[str, Any] = {}
    layer_map = {
        "0.weight": "mlp_extractor.policy_net.0.weight",
        "0.bias": "mlp_extractor.policy_net.0.bias",
        "2.weight": "mlp_extractor.policy_net.2.weight",
        "2.bias": "mlp_extractor.policy_net.2.bias",
    }
    for bc_key, sb3_key in layer_map.items():
        if bc_key in bc_state:
            sb3_state[sb3_key] = bc_state[bc_key].cpu()

    logger.info(
        "BC training done: %d samples, %d epochs, mapped %d layers to SB3 format",
        len(obs), epochs, len(sb3_state),
    )
    return sb3_state


def bc_warmstart_state_dict(
    env: OptimalExecutionEnv,
    n_episodes: int = 500,
    seed: int = 42,
    epochs: int = 20,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """End-to-end: collect TWAP demos and train BC, return SB3-compatible state_dict."""
    logger.info("Collecting %d TWAP demo episodes...", n_episodes)
    obs, actions = collect_twap_demos(env, n_episodes=n_episodes, seed=seed)
    logger.info("Collected %d (obs, action) pairs", len(obs))
    return train_bc(obs, actions, epochs=epochs, lr=lr)
