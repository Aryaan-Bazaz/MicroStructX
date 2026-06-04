from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from microstructx.agents import (
    ArbitrageurAgent,
    MarketMakerAgent,
    MeanReversionAgent,
    MomentumAgent,
    NoiseAgent,
    TradingAgent,
)
from microstructx.analytics import regime_execution_report, summarize_execution_risk, summarize_executions
from microstructx.execution import ExecutionConfig, execute_orders
from microstructx.signals import add_market_features


@dataclass(frozen=True)
class SimulationConfig:
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    warmup_events: int = 100


def default_agents() -> list[TradingAgent]:
    return [
        MomentumAgent(),
        MeanReversionAgent(),
        MarketMakerAgent(),
        ArbitrageurAgent(),
        NoiseAgent(),
    ]


def build_agent_orders(market: pl.DataFrame, agents: list[TradingAgent] | None = None) -> pl.DataFrame:
    agents = agents or default_agents()
    order_frames = [agent.generate_orders(market) for agent in agents]
    if not order_frames:
        return pl.DataFrame()
    return pl.concat(order_frames).sort(["timestamp", "agent"])


def run_multi_agent_simulation(
    book: pl.DataFrame,
    agents: list[TradingAgent] | None = None,
    config: SimulationConfig | None = None,
) -> dict[str, pl.DataFrame | dict[str, float]]:
    """Run feature generation, agent order creation, and microstructure execution."""
    config = config or SimulationConfig()
    featured = add_market_features(book).filter(pl.col("timestamp") >= config.warmup_events)
    orders = build_agent_orders(featured, agents)
    executions = execute_orders(featured, orders, config.execution)

    agent_summary = (
        executions.group_by("agent")
        .agg(
            [
                pl.len().alias("orders"),
                pl.col("filled_qty").sum().alias("filled_qty"),
                pl.col("notional").sum().alias("notional"),
                pl.col("fees").sum().alias("fees"),
                pl.col("slippage").sum().alias("slippage"),
                pl.col("realized_pnl").sum().alias("realized_pnl"),
            ]
        )
        .sort("realized_pnl", descending=True)
        if not executions.is_empty()
        else pl.DataFrame()
    )

    regime_summary = (
        executions.group_by("regime")
        .agg(
            [
                pl.len().alias("orders"),
                pl.col("filled_qty").sum().alias("filled_qty"),
                pl.col("slippage").sum().alias("slippage"),
                pl.col("realized_pnl").sum().alias("realized_pnl"),
            ]
        )
        .sort("orders", descending=True)
        if not executions.is_empty()
        else pl.DataFrame()
    )

    return {
        "market": featured,
        "orders": orders,
        "executions": executions,
        "agent_summary": agent_summary,
        "regime_summary": regime_summary,
        "execution_summary": summarize_executions(executions),
        "risk_summary": summarize_execution_risk(executions),
        "regime_execution_report": regime_execution_report(executions),
    }
