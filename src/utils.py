"""Paths, seeds, logging."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

SEED = 42


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    import os

    env = os.environ.get("CFA_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root().parent.parent / "CFADATA"


def setup_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def regime_display_name(regime: int, n_regimes: int) -> str:
    if n_regimes <= 2:
        return ("calm", "elevated volatility")[min(regime, 1)]
    return ("calm", "elevated volatility", "stressed")[min(regime, 2)]
