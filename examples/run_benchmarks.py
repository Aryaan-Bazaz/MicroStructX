import polars as pl

from microstructx.benchmarks import benchmark_scaling_report
from microstructx.data import generate_synthetic_lob
from microstructx.simulator import build_agent_orders
from microstructx.signals import add_market_features


def main() -> None:
    book = generate_synthetic_lob(n_events=5_000, seed=101)
    market = add_market_features(book).filter(pl.col("timestamp") >= 100)
    orders = build_agent_orders(market)
    report = benchmark_scaling_report(event_sizes=(1_000, 5_000, 10_000), repeat=3)

    print("MicroStructX execution benchmark")
    print(f"single-run order sample: {orders.height:,} orders")
    for row in report.iter_rows(named=True):
        print(
            "  "
            + " | ".join(
                [
                    f"events={row['events']:,.0f}",
                    f"orders={row['orders']:,.0f}",
                    f"naive={row['naive_seconds']:.6f}s",
                    f"vectorized={row['vectorized_seconds']:.6f}s",
                    f"numba_fill={row['numba_fill_seconds']:.6f}s",
                    f"vectorized_speedup={row['vectorized_speedup_vs_naive']:.2f}x",
                    f"numba_fill_speedup={row['numba_fill_speedup_vs_naive']:.2f}x",
                ]
            )
        )


if __name__ == "__main__":
    main()
