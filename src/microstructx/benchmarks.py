from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np
import polars as pl
from numba import njit

from microstructx.data import generate_synthetic_lob
from microstructx.execution import ExecutionConfig, execute_orders
from microstructx.signals import add_market_features
from microstructx.simulator import build_agent_orders


def naive_execute_orders(book: pl.DataFrame, orders: pl.DataFrame, config: ExecutionConfig | None = None) -> pl.DataFrame:
    """Reference Python-loop executor used for correctness checks and benchmarking."""
    config = config or ExecutionConfig()
    if orders.is_empty():
        return pl.DataFrame()

    book_by_timestamp = {row["timestamp"]: row for row in book.iter_rows(named=True)}
    rows: list[dict[str, float | str | int]] = []

    for order in orders.iter_rows(named=True):
        event = book_by_timestamp.get(order["timestamp"] + config.latency_events)
        next_event = book_by_timestamp.get(order["timestamp"] + config.latency_events + 1, event)
        if event is None:
            continue

        side_sign = 1.0 if order["side"] == "buy" else -1.0
        requested_qty = abs(float(order["quantity"]))
        order_type = "limit" if order["order_type"] == "limit" else "market"
        opposite_depth = float(event["ask_size"] if order["side"] == "buy" else event["bid_size"])
        same_side_depth = float(event["bid_size"] if order["side"] == "buy" else event["ask_size"])
        touch_price = float(event["ask_price"] if order["side"] == "buy" else event["bid_price"])
        exec_mid_price = float(event["mid_price"])
        exec_volume = float(event["volume"])

        queue_pressure = min(max(requested_qty / (opposite_depth + 1.0), 0.0), 10.0)
        queue_position_ahead = same_side_depth * config.queue_ahead_fraction * (
            1.0 + config.latency_events * config.latency_queue_penalty
        )
        queue_cancellations = same_side_depth * config.cancellation_rate
        imbalance_term = (
            (-float(event["bid_size"]) + float(event["ask_size"]))
            if order["side"] == "buy"
            else (float(event["bid_size"]) - float(event["ask_size"]))
        ) / (float(event["bid_size"]) + float(event["ask_size"]) + 1.0)
        queue_depletion = max(0.0, exec_volume * (1.0 + imbalance_term * 0.35) + queue_cancellations)
        if queue_depletion > queue_position_ahead:
            limit_fill_probability = min(
                max(
                    (queue_depletion - queue_position_ahead) / (requested_qty + 1.0),
                    config.min_limit_fill_probability,
                ),
                1.0,
            )
        else:
            limit_fill_probability = 0.0
        max_fill_qty = opposite_depth * config.max_participation_rate
        if order_type == "market":
            filled_qty = min(requested_qty, max_fill_qty)
        else:
            fillable_after_queue = max(0.0, queue_depletion - queue_position_ahead)
            filled_qty = min(requested_qty, max_fill_qty, fillable_after_queue) * limit_fill_probability

        temporary_impact = (
            side_sign * exec_mid_price * (config.temporary_impact_bps / 10_000.0) * queue_pressure
        )
        permanent_impact = (
            side_sign * exec_mid_price * (config.permanent_impact_bps / 10_000.0) * queue_pressure
        )
        fill_price = touch_price + temporary_impact
        notional = filled_qty * fill_price
        fees = notional * (config.fee_bps / 10_000.0)
        slippage = side_sign * filled_qty * (fill_price - exec_mid_price)
        next_mid_price = float(next_event["mid_price"]) if next_event is not None else exec_mid_price
        realized_pnl = side_sign * filled_qty * (next_mid_price - fill_price) - fees

        rows.append(
            {
                "timestamp": order["timestamp"],
                "agent": order["agent"],
                "side": order["side"],
                "order_type": order_type,
                "requested_qty": requested_qty,
                "filled_qty": filled_qty,
                "fill_price": fill_price,
                "exec_mid_price": exec_mid_price,
                "next_mid_price": next_mid_price,
                "notional": notional,
                "fees": fees,
                "slippage": slippage,
                "temporary_impact": temporary_impact,
                "permanent_impact": permanent_impact,
                "queue_pressure": queue_pressure,
                "same_side_depth": same_side_depth,
                "opposite_depth": opposite_depth,
                "queue_position_ahead": queue_position_ahead,
                "queue_depletion": queue_depletion,
                "limit_fill_probability": limit_fill_probability,
                "realized_pnl": realized_pnl,
                "regime": event["regime"],
            }
        )

    return pl.DataFrame(rows)


@njit(cache=True)
def _numba_fill_kernel(
    side_sign: np.ndarray,
    order_type_code: np.ndarray,
    requested_qty: np.ndarray,
    bid_size: np.ndarray,
    ask_size: np.ndarray,
    volume: np.ndarray,
    max_participation_rate: float,
    min_limit_fill_probability: float,
    queue_ahead_fraction: float,
    latency_events: int,
    latency_queue_penalty: float,
    cancellation_rate: float,
) -> np.ndarray:
    filled = np.empty(requested_qty.shape[0], dtype=np.float64)
    for idx in range(requested_qty.shape[0]):
        is_buy = side_sign[idx] > 0.0
        opposite_depth = ask_size[idx] if is_buy else bid_size[idx]
        same_side_depth = bid_size[idx] if is_buy else ask_size[idx]
        queue_position_ahead = same_side_depth * queue_ahead_fraction * (
            1.0 + latency_events * latency_queue_penalty
        )
        queue_cancellations = same_side_depth * cancellation_rate
        imbalance_term = ((-bid_size[idx] + ask_size[idx]) if is_buy else (bid_size[idx] - ask_size[idx])) / (
            bid_size[idx] + ask_size[idx] + 1.0
        )
        queue_depletion = max(0.0, volume[idx] * (1.0 + imbalance_term * 0.35) + queue_cancellations)
        max_fill_qty = opposite_depth * max_participation_rate

        if order_type_code[idx] == 0:
            filled[idx] = min(requested_qty[idx], max_fill_qty)
        else:
            fillable_after_queue = max(0.0, queue_depletion - queue_position_ahead)
            if queue_depletion > queue_position_ahead:
                limit_fill_probability = min(
                    max((queue_depletion - queue_position_ahead) / (requested_qty[idx] + 1.0), min_limit_fill_probability),
                    1.0,
                )
            else:
                limit_fill_probability = 0.0
            filled[idx] = min(requested_qty[idx], max_fill_qty, fillable_after_queue) * limit_fill_probability
    return filled


def numba_estimate_fills(book: pl.DataFrame, orders: pl.DataFrame, config: ExecutionConfig | None = None) -> np.ndarray:
    """Benchmarkable Numba kernel for queue-aware fill quantities."""
    config = config or ExecutionConfig()
    if orders.is_empty():
        return np.array([], dtype=np.float64)

    joined = (
        orders.with_columns(
            [
                pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0).alias("side_sign"),
                pl.when(pl.col("order_type") == "limit").then(1).otherwise(0).alias("order_type_code"),
                pl.col("quantity").abs().cast(pl.Float64).alias("requested_qty"),
            ]
        )
        .join(
            book.select(
                [
                    (pl.col("timestamp") - config.latency_events).alias("timestamp"),
                    "bid_size",
                    "ask_size",
                    "volume",
                ]
            ),
            on="timestamp",
            how="inner",
        )
        .select(["side_sign", "order_type_code", "requested_qty", "bid_size", "ask_size", "volume"])
    )

    return _numba_fill_kernel(
        joined["side_sign"].to_numpy(),
        joined["order_type_code"].to_numpy(),
        joined["requested_qty"].to_numpy(),
        joined["bid_size"].to_numpy(),
        joined["ask_size"].to_numpy(),
        joined["volume"].to_numpy(),
        config.max_participation_rate,
        config.min_limit_fill_probability,
        config.queue_ahead_fraction,
        config.latency_events,
        config.latency_queue_penalty,
        config.cancellation_rate,
    )


def benchmark_execution(
    book: pl.DataFrame,
    orders: pl.DataFrame,
    config: ExecutionConfig | None = None,
    repeat: int = 3,
) -> dict[str, float | dict[str, float]]:
    """Measure naive Python, vectorized Polars, and Numba fill-kernel runtimes."""
    config = config or ExecutionConfig()
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    numba_estimate_fills(book, orders, config)

    timings: dict[str, float] = {}
    for name, func in {
        "naive_seconds": lambda: naive_execute_orders(book, orders, config),
        "vectorized_seconds": lambda: execute_orders(book, orders, config),
        "numba_fill_seconds": lambda: numba_estimate_fills(book, orders, config),
    }.items():
        best = float("inf")
        for _ in range(repeat):
            started = perf_counter()
            func()
            best = min(best, perf_counter() - started)
        timings[name] = best

    return {
        **timings,
        "vectorized_speedup_vs_naive": timings["naive_seconds"] / timings["vectorized_seconds"],
        "numba_fill_speedup_vs_naive": timings["naive_seconds"] / timings["numba_fill_seconds"],
        "orders": float(orders.height),
        "config": asdict(config),
    }


def benchmark_scaling_report(
    event_sizes: tuple[int, ...] = (1_000, 5_000, 10_000),
    seed: int = 101,
    repeat: int = 3,
) -> pl.DataFrame:
    """Benchmark execution speed across event sizes for README/report use."""
    rows = []
    for n_events in event_sizes:
        book = generate_synthetic_lob(n_events=n_events, seed=seed)
        market = add_market_features(book).filter(pl.col("timestamp") >= 100)
        orders = build_agent_orders(market)
        result = benchmark_execution(market, orders, repeat=repeat)
        rows.append(
            {
                "events": n_events,
                "orders": result["orders"],
                "naive_seconds": result["naive_seconds"],
                "vectorized_seconds": result["vectorized_seconds"],
                "numba_fill_seconds": result["numba_fill_seconds"],
                "vectorized_speedup_vs_naive": result["vectorized_speedup_vs_naive"],
                "numba_fill_speedup_vs_naive": result["numba_fill_speedup_vs_naive"],
            }
        )
    return pl.DataFrame(rows)
