from microstructx.data import generate_synthetic_lob
from microstructx.signals import add_market_features, vectorized_backtest
from microstructx.simulator import run_multi_agent_simulation


def print_ascii_agent_summary(agent_summary):
    for row in agent_summary.iter_rows(named=True):
        print("  " + " | ".join(f"{key}={value}" for key, value in row.items()))


def main() -> None:
    book = generate_synthetic_lob(n_events=25_000, seed=42)

    featured = add_market_features(book)
    phase_1 = vectorized_backtest(featured, signal_column="momentum")
    print(f"Phase 1 vectorized backtest final PnL: {phase_1['cum_pnl'][-1]:,.2f}")

    phase_2_3 = run_multi_agent_simulation(book)
    print("Phase 2-3 execution summary:")
    for key, value in phase_2_3["execution_summary"].items():
        print(f"  {key}: {value:,.4f}")

    print("Agent PnL summary:")
    print_ascii_agent_summary(phase_2_3["agent_summary"])


if __name__ == "__main__":
    main()
