from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from time import sleep

import polars as pl

from microstructx.data import generate_synthetic_lob
from microstructx.simulator import SimulationConfig, run_multi_agent_simulation


@dataclass(frozen=True)
class StreamingConfig:
    n_events: int = 5_000
    batch_size: int = 250
    seed: int = 7
    window_events: int = 1_500
    sleep_seconds: float = 0.0
    simulation: SimulationConfig | None = None


def stream_synthetic_lob(config: StreamingConfig) -> Iterator[pl.DataFrame]:
    """Yield synthetic LOB events in timestamp-ordered batches."""
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.n_events <= 0:
        raise ValueError("n_events must be positive")

    book = generate_synthetic_lob(n_events=max(config.n_events, 3), seed=config.seed)
    for start in range(0, config.n_events, config.batch_size):
        batch = book.slice(start, config.batch_size)
        if batch.is_empty():
            break
        yield batch
        if config.sleep_seconds > 0:
            sleep(config.sleep_seconds)


def run_streaming_simulation(config: StreamingConfig) -> list[dict[str, float]]:
    """Run rolling-window simulations over streamed LOB batches."""
    if config.window_events < config.batch_size:
        raise ValueError("window_events must be at least batch_size")

    history = pl.DataFrame()
    snapshots: list[dict[str, float]] = []
    sim_config = config.simulation or SimulationConfig(warmup_events=min(100, max(5, config.batch_size // 2)))

    for batch_id, batch in enumerate(stream_synthetic_lob(config), start=1):
        history = pl.concat([history, batch]) if not history.is_empty() else batch
        history = history.tail(config.window_events)

        if history.height <= sim_config.warmup_events + 2:
            snapshots.append(
                {
                    "batch_id": float(batch_id),
                    "last_timestamp": float(batch["timestamp"][-1]),
                    "events_seen": float(history.height),
                    "orders": 0.0,
                    "realized_pnl": 0.0,
                    "fill_rate": 0.0,
                    "avg_spread_bps": float(batch["spread_bps"].mean()),
                }
            )
            continue

        result = run_multi_agent_simulation(history, config=sim_config)
        executions = result["executions"]
        current_start = int(batch["timestamp"][0])
        current_end = int(batch["timestamp"][-1])
        current_executions = executions.filter(
            (pl.col("timestamp") >= current_start) & (pl.col("timestamp") <= current_end)
        )

        snapshots.append(
            {
                "batch_id": float(batch_id),
                "last_timestamp": float(current_end),
                "events_seen": float(history.height),
                "orders": float(current_executions.height),
                "realized_pnl": float(current_executions["realized_pnl"].sum()) if not current_executions.is_empty() else 0.0,
                "fill_rate": float((current_executions["filled_qty"] > 0).mean())
                if not current_executions.is_empty()
                else 0.0,
                "avg_spread_bps": float(batch["spread_bps"].mean()),
            }
        )

    return snapshots


def snapshots_to_frame(snapshots: list[dict[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(snapshots) if snapshots else pl.DataFrame()
