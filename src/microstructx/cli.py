from __future__ import annotations

import argparse

import polars as pl

from microstructx.data import generate_synthetic_lob
from microstructx.signals import add_market_features, vectorized_backtest
from microstructx.simulator import run_multi_agent_simulation


def print_ascii_table(frame: pl.DataFrame, columns: list[str]) -> None:
    for row in frame.select(columns).iter_rows(named=True):
        print("  " + " | ".join(f"{key}={value}" for key, value in row.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MicroStructX simulation demo.")
    parser.add_argument("--events", type=int, default=10_000, help="Number of synthetic LOB events.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic market generation.")
    args = parser.parse_args()

    book = generate_synthetic_lob(n_events=args.events, seed=args.seed)
    featured = add_market_features(book)
    baseline = vectorized_backtest(featured, "momentum")
    simulation = run_multi_agent_simulation(book)

    print("MicroStructX demo")
    print(f"events: {args.events}")
    print(f"baseline final pnl: {baseline['cum_pnl'][-1]:,.2f}")
    print("execution summary:")
    for key, value in simulation["execution_summary"].items():
        print(f"  {key}: {value:,.4f}")
    print("risk summary:")
    for key, value in simulation["risk_summary"].items():
        print(f"  {key}: {value:,.4f}")
    print("agent summary:")
    print_ascii_table(
        simulation["agent_summary"],
        ["agent", "orders", "filled_qty", "notional", "fees", "slippage", "realized_pnl"],
    )


if __name__ == "__main__":
    main()
