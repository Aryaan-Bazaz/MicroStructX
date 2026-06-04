from __future__ import annotations

import numpy as np
import polars as pl


REGIME_VOL = {
    "calm": 0.00035,
    "trend": 0.00055,
    "volatile": 0.00125,
    "panic": 0.00250,
}

REGIME_SPREAD_BPS = {
    "calm": 1.2,
    "trend": 1.8,
    "volatile": 4.0,
    "panic": 9.0,
}


def _sample_regimes(n_events: int, rng: np.random.Generator) -> np.ndarray:
    transition = {
        "calm": ("calm", "trend", "volatile"),
        "trend": ("trend", "calm", "volatile"),
        "volatile": ("volatile", "calm", "panic"),
        "panic": ("panic", "volatile", "calm"),
    }
    probs = {
        "calm": (0.92, 0.06, 0.02),
        "trend": (0.86, 0.10, 0.04),
        "volatile": (0.82, 0.14, 0.04),
        "panic": (0.72, 0.24, 0.04),
    }

    regimes = np.empty(n_events, dtype=object)
    state = "calm"
    for idx in range(n_events):
        regimes[idx] = state
        state = rng.choice(transition[state], p=probs[state])
    return regimes


def generate_synthetic_lob(
    n_events: int = 10_000,
    seed: int = 7,
    start_price: float = 100.0,
) -> pl.DataFrame:
    """Generate synthetic level-1 order book events with regime shifts."""
    if n_events <= 2:
        raise ValueError("n_events must be greater than 2")

    rng = np.random.default_rng(seed)
    regimes = _sample_regimes(n_events, rng)
    vol = np.array([REGIME_VOL[state] for state in regimes])
    spread_bps = np.array([REGIME_SPREAD_BPS[state] for state in regimes])

    trend_drift = np.where(regimes == "trend", 0.00008, 0.0)
    panic_drift = np.where(regimes == "panic", -0.00015, 0.0)
    returns = trend_drift + panic_drift + rng.normal(0.0, vol)
    mid_price = start_price * np.exp(np.cumsum(returns))

    liquidity_scale = np.where(regimes == "panic", 0.35, np.where(regimes == "volatile", 0.65, 1.0))
    base_size = rng.lognormal(mean=5.1, sigma=0.45, size=n_events)
    bid_size = np.maximum(10.0, base_size * liquidity_scale * rng.uniform(0.7, 1.3, n_events))
    ask_size = np.maximum(10.0, base_size * liquidity_scale * rng.uniform(0.7, 1.3, n_events))

    half_spread = mid_price * spread_bps / 20_000.0
    bid_price = mid_price - half_spread
    ask_price = mid_price + half_spread
    volume = rng.poisson(lam=np.maximum(5.0, (bid_size + ask_size) / 18.0)).astype(float)

    return pl.DataFrame(
        {
            "timestamp": np.arange(n_events, dtype=np.int64),
            "mid_price": mid_price,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread_bps": spread_bps,
            "volume": volume,
            "volatility": vol,
            "regime": regimes,
        }
    )
