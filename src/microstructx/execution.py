from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class ExecutionConfig:
    fee_bps: float = 0.5
    temporary_impact_bps: float = 2.0
    permanent_impact_bps: float = 0.2
    max_participation_rate: float = 0.25
    latency_events: int = 1
    min_limit_fill_probability: float = 0.02
    queue_ahead_fraction: float = 0.28
    latency_queue_penalty: float = 0.12
    cancellation_rate: float = 0.12


def _validate_orders(orders: pl.DataFrame) -> None:
    required = {"timestamp", "agent", "side", "quantity", "order_type"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError(f"orders missing columns: {sorted(missing)}")


def execute_orders(
    book: pl.DataFrame,
    orders: pl.DataFrame,
    config: ExecutionConfig | None = None,
) -> pl.DataFrame:
    """Execute agent orders with explicit queue-ahead and impact approximations."""
    config = config or ExecutionConfig()
    _validate_orders(orders)
    if orders.is_empty():
        return pl.DataFrame()

    book_at_execution = book.select(
        [
            (pl.col("timestamp") - config.latency_events).alias("timestamp"),
            pl.col("mid_price").alias("exec_mid_price"),
            pl.col("bid_price").alias("exec_bid_price"),
            pl.col("ask_price").alias("exec_ask_price"),
            pl.col("bid_size").alias("exec_bid_size"),
            pl.col("ask_size").alias("exec_ask_size"),
            pl.col("spread_bps").alias("exec_spread_bps"),
            pl.col("volume").alias("exec_volume"),
            pl.col("regime").alias("exec_regime"),
        ]
    )

    enriched = (
        orders.with_columns(
            [
                pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0).alias("side_sign"),
                pl.col("quantity").cast(pl.Float64).abs().alias("requested_qty"),
                pl.when(pl.col("order_type") == "limit").then(pl.lit("limit")).otherwise(pl.lit("market")).alias(
                    "normalized_order_type"
                ),
            ]
        )
        .join(book_at_execution, on="timestamp", how="inner")
        .with_columns(
            [
                pl.when(pl.col("side") == "buy")
                .then(pl.col("exec_ask_size"))
                .otherwise(pl.col("exec_bid_size"))
                .alias("opposite_depth"),
                pl.when(pl.col("side") == "buy")
                .then(pl.col("exec_bid_size"))
                .otherwise(pl.col("exec_ask_size"))
                .alias("same_side_depth"),
                pl.when(pl.col("side") == "buy")
                .then(pl.col("exec_ask_price"))
                .otherwise(pl.col("exec_bid_price"))
                .alias("touch_price"),
            ]
        )
        .with_columns(
            (pl.col("requested_qty") / (pl.col("opposite_depth") + 1.0)).clip(0.0, 10.0).alias("queue_pressure")
        )
        .with_columns(
            [
                (
                    pl.col("same_side_depth")
                    * config.queue_ahead_fraction
                    * (1.0 + config.latency_events * config.latency_queue_penalty)
                ).alias("queue_position_ahead"),
                (pl.col("same_side_depth") * config.cancellation_rate).alias("queue_cancellations"),
                (pl.col("opposite_depth") * config.max_participation_rate).alias("max_fill_qty"),
            ]
        )
        .with_columns(
            (
                pl.col("exec_volume")
                * (
                    1.0
                    + pl.when(pl.col("side") == "buy")
                    .then(-pl.col("exec_bid_size") + pl.col("exec_ask_size"))
                    .otherwise(pl.col("exec_bid_size") - pl.col("exec_ask_size"))
                    / (pl.col("exec_bid_size") + pl.col("exec_ask_size") + 1.0)
                    * 0.35
                )
                + pl.col("queue_cancellations")
            )
            .clip(0.0, None)
            .alias("queue_depletion")
        )
        .with_columns(
            pl.when(pl.col("queue_depletion") > pl.col("queue_position_ahead"))
            .then(
                (
                    (pl.col("queue_depletion") - pl.col("queue_position_ahead"))
                    / (pl.col("requested_qty") + 1.0)
                ).clip(config.min_limit_fill_probability, 1.0)
            )
            .otherwise(0.0)
            .alias("limit_fill_probability")
        )
        .with_columns(
            pl.when(pl.col("normalized_order_type") == "market")
            .then(pl.min_horizontal("requested_qty", "max_fill_qty"))
            .otherwise(
                pl.min_horizontal(
                    "requested_qty",
                    "max_fill_qty",
                    (pl.col("queue_depletion") - pl.col("queue_position_ahead")).clip(0.0, None),
                )
                * pl.col("limit_fill_probability")
            )
            .alias("filled_qty")
        )
        .with_columns(
            [
                (
                    pl.col("side_sign")
                    * pl.col("exec_mid_price")
                    * (config.temporary_impact_bps / 10_000.0)
                    * pl.col("queue_pressure")
                ).alias("temporary_impact"),
                (
                    pl.col("side_sign")
                    * pl.col("exec_mid_price")
                    * (config.permanent_impact_bps / 10_000.0)
                    * pl.col("queue_pressure")
                ).alias("permanent_impact"),
            ]
        )
        .with_columns((pl.col("touch_price") + pl.col("temporary_impact")).alias("fill_price"))
        .with_columns(
            [
                (pl.col("filled_qty") * pl.col("fill_price")).alias("notional"),
                (pl.col("filled_qty") * pl.col("fill_price") * (config.fee_bps / 10_000.0)).alias("fees"),
                (
                    pl.col("side_sign")
                    * pl.col("filled_qty")
                    * (pl.col("fill_price") - pl.col("exec_mid_price"))
                ).alias("slippage"),
                (-pl.col("side_sign") * pl.col("filled_qty") * pl.col("fill_price")).alias("cash_flow"),
            ]
        )
    )

    mark_prices = book.select(
        [
            (pl.col("timestamp") - config.latency_events - 1).alias("timestamp"),
            pl.col("mid_price").alias("next_mid_price"),
        ]
    )

    return (
        enriched.join(mark_prices, on="timestamp", how="left")
        .with_columns(pl.col("next_mid_price").fill_null(pl.col("exec_mid_price")))
        .with_columns(
            (
                pl.col("side_sign")
                * pl.col("filled_qty")
                * (pl.col("next_mid_price") - pl.col("fill_price"))
                - pl.col("fees")
            ).alias("realized_pnl")
        )
        .select(
            [
                "timestamp",
                "agent",
                "side",
                "normalized_order_type",
                "requested_qty",
                "filled_qty",
                "fill_price",
                "exec_mid_price",
                "next_mid_price",
                "notional",
                "fees",
                "slippage",
                "temporary_impact",
                "permanent_impact",
                "queue_pressure",
                "same_side_depth",
                "opposite_depth",
                "queue_position_ahead",
                "queue_depletion",
                "limit_fill_probability",
                "realized_pnl",
                "exec_regime",
            ]
        )
        .rename({"normalized_order_type": "order_type", "exec_regime": "regime"})
    )
