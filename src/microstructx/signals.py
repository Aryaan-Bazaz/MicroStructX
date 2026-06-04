from __future__ import annotations

import polars as pl


def add_market_features(book: pl.DataFrame, momentum_window: int = 20, mean_window: int = 80) -> pl.DataFrame:
    """Add vectorized returns, momentum, imbalance, and z-score features."""
    return (
        book.sort("timestamp")
        .with_columns(
            [
                pl.col("mid_price").pct_change().fill_null(0.0).alias("return_1"),
                ((pl.col("bid_size") - pl.col("ask_size")) / (pl.col("bid_size") + pl.col("ask_size")))
                .fill_nan(0.0)
                .alias("book_imbalance"),
            ]
        )
        .with_columns(
            [
                pl.col("return_1").rolling_sum(momentum_window).fill_null(0.0).alias("momentum"),
                pl.col("mid_price").rolling_mean(mean_window).alias("rolling_mean"),
                pl.col("mid_price").rolling_std(mean_window).alias("rolling_std"),
            ]
        )
        .with_columns(
            ((pl.col("mid_price") - pl.col("rolling_mean")) / pl.col("rolling_std"))
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("zscore")
        )
    )


def vectorized_backtest(
    featured_book: pl.DataFrame,
    signal_column: str,
    capital: float = 1_000_000.0,
    max_position_notional: float = 100_000.0,
    fee_bps: float = 0.5,
) -> pl.DataFrame:
    """Backtest a continuous signal using vectorized close-to-close mark-to-market PnL."""
    if signal_column not in featured_book.columns:
        raise ValueError(f"missing signal column: {signal_column}")

    max_shares = max_position_notional / featured_book.select(pl.col("mid_price").mean()).item()
    return (
        featured_book.sort("timestamp")
        .with_columns(pl.col(signal_column).clip(-1.0, 1.0).alias("target_signal"))
        .with_columns((pl.col("target_signal") * max_shares).alias("position"))
        .with_columns((pl.col("position") - pl.col("position").shift(1).fill_null(0.0)).alias("trade_qty"))
        .with_columns(
            [
                (pl.col("position").shift(1).fill_null(0.0) * pl.col("mid_price").diff().fill_null(0.0)).alias(
                    "gross_pnl"
                ),
                (
                    pl.col("trade_qty").abs()
                    * pl.col("mid_price")
                    * (fee_bps / 10_000.0)
                ).alias("fees"),
            ]
        )
        .with_columns((pl.col("gross_pnl") - pl.col("fees")).alias("net_pnl"))
        .with_columns(
            [
                pl.col("net_pnl").cum_sum().alias("cum_pnl"),
                (capital + pl.col("net_pnl").cum_sum()).alias("equity"),
            ]
        )
    )
