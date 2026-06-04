from microstructx.analytics import regime_execution_report
from microstructx.data import generate_synthetic_lob
from microstructx.simulator import run_multi_agent_simulation


def main() -> None:
    book = generate_synthetic_lob(n_events=25_000, seed=202)
    result = run_multi_agent_simulation(book)
    report = regime_execution_report(result["executions"])
    risk = result["risk_summary"]

    print("MicroStructX research note: slippage by regime")
    print(
        "summary: "
        f"pnl={risk['total_pnl']:,.2f}, "
        f"sharpe_like={risk['sharpe_like']:.2f}, "
        f"max_drawdown={risk['max_drawdown']:,.2f}, "
        f"avg_slippage_bps={risk['avg_slippage_bps']:.4f}"
    )
    for row in report.iter_rows(named=True):
        print(
            "  "
            + " | ".join(
                [
                    f"regime={row['regime']}",
                    f"orders={row['orders']:,.0f}",
                    f"qty_fill={row['quantity_fill_ratio']:.2%}",
                    f"slippage_bps={row['slippage_bps']:.4f}",
                    f"realized_pnl={row['realized_pnl']:,.2f}",
                    f"avg_queue_ahead={row['avg_queue_ahead']:.2f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
