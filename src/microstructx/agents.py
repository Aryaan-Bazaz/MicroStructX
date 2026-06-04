from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class AgentConfig:
    name: str
    max_order_size: float = 100.0
    activity_rate: float = 1.0


class TradingAgent:
    def __init__(self, config: AgentConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    def generate_orders(self, market: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def _base_orders(self, market: pl.DataFrame, signal: pl.Expr, order_type: str = "market") -> pl.DataFrame:
        return (
            market.with_columns(signal.clip(-1.0, 1.0).alias("_signal"))
            .with_columns((pl.col("_signal").abs() * self.config.max_order_size * self.config.activity_rate).alias("quantity"))
            .filter(pl.col("quantity") > 1e-9)
            .with_columns(
                [
                    pl.lit(self.name).alias("agent"),
                    pl.when(pl.col("_signal") > 0).then(pl.lit("buy")).otherwise(pl.lit("sell")).alias("side"),
                    pl.lit(order_type).alias("order_type"),
                ]
            )
            .select(["timestamp", "agent", "side", "quantity", "order_type"])
        )


class MomentumAgent(TradingAgent):
    def __init__(self, max_order_size: float = 120.0):
        super().__init__(AgentConfig(name="momentum", max_order_size=max_order_size))

    def generate_orders(self, market: pl.DataFrame) -> pl.DataFrame:
        return self._base_orders(market, pl.col("momentum") * 150.0, "market")


class MeanReversionAgent(TradingAgent):
    def __init__(self, max_order_size: float = 90.0):
        super().__init__(AgentConfig(name="mean_reversion", max_order_size=max_order_size))

    def generate_orders(self, market: pl.DataFrame) -> pl.DataFrame:
        return self._base_orders(market, -pl.col("zscore") / 3.0, "limit")


class NoiseAgent(TradingAgent):
    def __init__(self, max_order_size: float = 35.0, seed: int = 11):
        super().__init__(AgentConfig(name="noise", max_order_size=max_order_size, activity_rate=0.25))
        self.seed = seed

    def generate_orders(self, market: pl.DataFrame) -> pl.DataFrame:
        rng = np.random.default_rng(self.seed)
        raw_signal = rng.normal(0.0, 1.0, market.height)
        active = rng.uniform(0.0, 1.0, market.height) < self.config.activity_rate
        signal = np.where(active, raw_signal, 0.0)
        return self._base_orders(market.with_columns(pl.Series("_noise_signal", signal)), pl.col("_noise_signal"), "market")


class MarketMakerAgent(TradingAgent):
    def __init__(self, max_order_size: float = 70.0):
        super().__init__(AgentConfig(name="market_maker", max_order_size=max_order_size))

    def generate_orders(self, market: pl.DataFrame) -> pl.DataFrame:
        inventory_lean = -pl.col("book_imbalance") * 0.65
        base = self._base_orders(market, inventory_lean, "limit")
        passive_buy = market.select(
            [
                "timestamp",
                pl.lit(self.name).alias("agent"),
                pl.lit("buy").alias("side"),
                (pl.col("ask_size") / (pl.col("bid_size") + pl.col("ask_size")) * self.config.max_order_size * 0.15).alias(
                    "quantity"
                ),
                pl.lit("limit").alias("order_type"),
            ]
        )
        passive_sell = market.select(
            [
                "timestamp",
                pl.lit(self.name).alias("agent"),
                pl.lit("sell").alias("side"),
                (pl.col("bid_size") / (pl.col("bid_size") + pl.col("ask_size")) * self.config.max_order_size * 0.15).alias(
                    "quantity"
                ),
                pl.lit("limit").alias("order_type"),
            ]
        )
        return pl.concat([base, passive_buy, passive_sell]).filter(pl.col("quantity") > 1e-9)


class ArbitrageurAgent(TradingAgent):
    def __init__(self, max_order_size: float = 80.0):
        super().__init__(AgentConfig(name="arbitrageur", max_order_size=max_order_size))

    def generate_orders(self, market: pl.DataFrame) -> pl.DataFrame:
        fair_value = (
            pl.col("mid_price").rolling_mean(12).fill_null(pl.col("mid_price"))
            + pl.col("book_imbalance") * pl.col("mid_price") * 0.0005
        )
        market_with_fair = market.with_columns(fair_value.alias("_fair_value")).with_columns(
            ((pl.col("_fair_value") - pl.col("mid_price")) / (pl.col("mid_price") * 0.001)).alias("_arb_signal")
        )
        return self._base_orders(market_with_fair, pl.col("_arb_signal"), "market")
