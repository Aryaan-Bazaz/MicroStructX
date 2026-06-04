from __future__ import annotations

import math

import polars as pl


def summarize_pnl(results: pl.DataFrame, pnl_column: str = "net_pnl") -> dict[str, float]:
    """Return compact PnL and risk statistics for simulation outputs."""
    if pnl_column not in results.columns:
        raise ValueError(f"missing pnl column: {pnl_column}")

    pnl = results.select(pl.col(pnl_column)).to_series()
    total_pnl = float(pnl.sum())
    mean_pnl = float(pnl.mean() or 0.0)
    std_pnl = float(pnl.std() or 0.0)
    sharpe_like = mean_pnl / std_pnl * math.sqrt(len(pnl)) if std_pnl > 0 else 0.0

    equity = results.select(pl.col(pnl_column).cum_sum()).to_series()
    high_watermark = equity.cum_max()
    drawdown = equity - high_watermark

    return {
        "events": float(len(results)),
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "pnl_std": std_pnl,
        "sharpe_like": float(sharpe_like),
        "max_drawdown": float(drawdown.min() or 0.0),
    }


def summarize_executions(executions: pl.DataFrame) -> dict[str, float]:
    """Summarize fills, costs, slippage, and realized PnL from executed orders."""
    if executions.is_empty():
        return {
            "orders": 0.0,
            "order_fill_rate": 0.0,
            "full_fill_rate": 0.0,
            "quantity_fill_ratio": 0.0,
            "notional": 0.0,
            "fees": 0.0,
            "slippage": 0.0,
            "realized_pnl": 0.0,
            "turnover": 0.0,
            "avg_queue_ahead": 0.0,
        }

    requested_qty = float(executions["requested_qty"].sum())
    filled_qty = float(executions["filled_qty"].sum())
    return {
        "orders": float(executions.height),
        "order_fill_rate": float((executions["filled_qty"] > 0).mean()),
        "full_fill_rate": float((executions["filled_qty"] >= executions["requested_qty"]).mean()),
        "quantity_fill_ratio": filled_qty / requested_qty if requested_qty > 0 else 0.0,
        "notional": float(executions["notional"].sum()),
        "fees": float(executions["fees"].sum()),
        "slippage": float(executions["slippage"].sum()),
        "realized_pnl": float(executions["realized_pnl"].sum()),
        "turnover": float(executions["notional"].sum()),
        "avg_queue_ahead": float(executions["queue_position_ahead"].mean() or 0.0),
    }


def summarize_execution_risk(executions: pl.DataFrame) -> dict[str, float]:
    """Risk metrics for executed-order PnL."""
    if executions.is_empty():
        return {
            "total_pnl": 0.0,
            "mean_pnl": 0.0,
            "pnl_std": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_slippage_bps": 0.0,
        }

    pnl = executions["realized_pnl"]
    total_pnl = float(pnl.sum())
    mean_pnl = float(pnl.mean() or 0.0)
    std_pnl = float(pnl.std() or 0.0)
    sharpe_like = mean_pnl / std_pnl * math.sqrt(len(pnl)) if std_pnl > 0 else 0.0
    equity = pnl.cum_sum()
    drawdown = equity - equity.cum_max()
    wins = float((pnl > 0).mean())
    gross_profit = float(executions.filter(pl.col("realized_pnl") > 0)["realized_pnl"].sum())
    gross_loss = abs(float(executions.filter(pl.col("realized_pnl") < 0)["realized_pnl"].sum()))
    notional = float(executions["notional"].sum())
    avg_slippage_bps = float(executions["slippage"].sum()) / notional * 10_000.0 if notional > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "pnl_std": std_pnl,
        "sharpe_like": float(sharpe_like),
        "max_drawdown": float(drawdown.min() or 0.0),
        "win_rate": wins,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
        "avg_slippage_bps": avg_slippage_bps,
    }


def regime_execution_report(executions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate execution quality by market regime."""
    if executions.is_empty():
        return pl.DataFrame()

    return (
        executions.group_by("regime")
        .agg(
            [
                pl.len().alias("orders"),
                pl.col("requested_qty").sum().alias("requested_qty"),
                pl.col("filled_qty").sum().alias("filled_qty"),
                pl.col("notional").sum().alias("notional"),
                pl.col("slippage").sum().alias("slippage"),
                pl.col("fees").sum().alias("fees"),
                pl.col("realized_pnl").sum().alias("realized_pnl"),
                pl.col("queue_position_ahead").mean().alias("avg_queue_ahead"),
            ]
        )
        .with_columns(
            [
                (pl.col("filled_qty") / pl.col("requested_qty")).fill_nan(0.0).alias("quantity_fill_ratio"),
                (pl.col("slippage") / pl.col("notional") * 10_000.0).fill_nan(0.0).alias("slippage_bps"),
            ]
        )
        .sort("slippage_bps", descending=True)
    )
