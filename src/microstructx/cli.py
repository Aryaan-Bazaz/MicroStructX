from __future__ import annotations

import argparse

import polars as pl

from microstructx.data import generate_synthetic_lob
from microstructx.loaders import download_lob_dataset, load_lob_file
from microstructx.ml import run_ml_research_pipeline
from microstructx.signals import add_market_features, vectorized_backtest
from microstructx.simulator import run_multi_agent_simulation


def print_ascii_table(frame: pl.DataFrame, columns: list[str]) -> None:
    if frame.is_empty():
        print("  <empty>")
        return
    for row in frame.select(columns).iter_rows(named=True):
        print("  " + " | ".join(f"{key}={value}" for key, value in row.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MicroStructX from synthetic, local, or online LOB data.")
    parser.add_argument("--events", type=int, default=10_000, help="Number of synthetic LOB events.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic market generation.")
    parser.add_argument("--path", type=str, default=None, help="Local CSV/Parquet/ZIP LOB dataset.")
    parser.add_argument("--url", type=str, default=None, help="Online CSV/ZIP LOB dataset URL to download and run.")
    parser.add_argument("--output", type=str, default="datasets/online_lob.zip", help="Download path for --url.")
    parser.add_argument("--max-rows", type=int, default=50_000, help="Maximum rows to load from a real dataset.")
    parser.add_argument("--ml", action="store_true", help="Train ML models after the simulation.")
    args = parser.parse_args()

    source = "synthetic"
    dataset_path = args.path
    if args.url:
        dataset_path = str(download_lob_dataset(args.url, args.output))
        source = f"downloaded:{dataset_path}"
    elif dataset_path:
        source = dataset_path

    if dataset_path:
        book = load_lob_file(dataset_path, max_rows=args.max_rows)
    else:
        book = generate_synthetic_lob(n_events=args.events, seed=args.seed)

    featured = add_market_features(book)
    baseline = vectorized_backtest(featured, "momentum")
    simulation = run_multi_agent_simulation(book)

    print("MicroStructX CLI simulation")
    print(f"source: {source}")
    print(f"rows: {book.height:,}")
    print(f"timestamp range: {book['timestamp'][0]} -> {book['timestamp'][-1]}")
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
    print("regime execution report:")
    print_ascii_table(
        simulation["regime_execution_report"],
        ["regime", "orders", "quantity_fill_ratio", "slippage_bps", "realized_pnl"],
    )

    if args.ml:
        print("ml summary:")
        ml_results = run_ml_research_pipeline(book)
        for model_name, result in ml_results.items():
            metrics = " | ".join(f"{key}={value:.4f}" for key, value in result.metrics.items())
            print(f"  {model_name}: {metrics}")


if __name__ == "__main__":
    main()
