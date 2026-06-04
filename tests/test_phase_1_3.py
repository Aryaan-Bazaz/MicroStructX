import polars as pl

from microstructx.benchmarks import benchmark_execution, benchmark_scaling_report, naive_execute_orders, numba_estimate_fills
from microstructx.data import generate_synthetic_lob
from microstructx.dashboard import prepare_dashboard_data
from microstructx.execution import execute_orders
from microstructx.loaders import load_lob_file, normalize_lob_frame
from microstructx.ml import build_ml_market_frame, run_ml_research_pipeline
from microstructx.signals import add_market_features, vectorized_backtest
from microstructx.simulator import run_multi_agent_simulation
from microstructx.streaming import StreamingConfig, run_streaming_simulation, snapshots_to_frame, stream_synthetic_lob


def test_vectorized_backtest_produces_equity_curve() -> None:
    book = generate_synthetic_lob(n_events=500, seed=1)
    featured = add_market_features(book)
    result = vectorized_backtest(featured, "momentum")

    assert result.height == book.height
    assert "equity" in result.columns
    assert result["equity"].null_count() == 0


def test_execution_fills_orders_with_costs() -> None:
    book = generate_synthetic_lob(n_events=250, seed=2)
    orders = pl.DataFrame(
        {
            "timestamp": [10, 11, 12],
            "agent": ["test", "test", "test"],
            "side": ["buy", "sell", "buy"],
            "quantity": [20.0, 30.0, 25.0],
            "order_type": ["market", "limit", "market"],
        }
    )

    executions = execute_orders(book, orders)

    assert executions.height == 3
    assert executions["filled_qty"].sum() > 0
    assert executions["fees"].sum() > 0
    assert "queue_pressure" in executions.columns
    assert "queue_position_ahead" in executions.columns
    assert "queue_depletion" in executions.columns
    assert "limit_fill_probability" in executions.columns


def test_multi_agent_simulation_returns_summaries() -> None:
    book = generate_synthetic_lob(n_events=800, seed=3)
    result = run_multi_agent_simulation(book)

    assert result["orders"].height > 0
    assert result["executions"].height > 0
    assert result["agent_summary"].height >= 3
    assert result["execution_summary"]["orders"] > 0
    assert "quantity_fill_ratio" in result["execution_summary"]
    assert "risk_summary" in result
    assert result["regime_execution_report"].height > 0


def test_naive_and_vectorized_execution_have_matching_fills() -> None:
    book = generate_synthetic_lob(n_events=300, seed=4)
    orders = pl.DataFrame(
        {
            "timestamp": [20, 21, 22, 23],
            "agent": ["test", "test", "test", "test"],
            "side": ["buy", "sell", "buy", "sell"],
            "quantity": [15.0, 30.0, 50.0, 12.0],
            "order_type": ["market", "limit", "limit", "market"],
        }
    )

    vectorized = execute_orders(book, orders)
    naive = naive_execute_orders(book, orders)

    assert vectorized.height == naive.height
    assert abs(vectorized["filled_qty"].sum() - naive["filled_qty"].sum()) < 1e-9


def test_numba_fill_kernel_and_benchmark_run() -> None:
    book = generate_synthetic_lob(n_events=350, seed=5)
    orders = pl.DataFrame(
        {
            "timestamp": [30, 31, 32],
            "agent": ["test", "test", "test"],
            "side": ["buy", "sell", "buy"],
            "quantity": [10.0, 20.0, 15.0],
            "order_type": ["market", "limit", "market"],
        }
    )

    fills = numba_estimate_fills(book, orders)
    benchmark = benchmark_execution(book, orders, repeat=1)

    assert fills.shape[0] == 3
    assert fills.sum() > 0
    assert benchmark["orders"] == 3.0
    assert benchmark["vectorized_speedup_vs_naive"] > 0


def test_benchmark_scaling_report_runs() -> None:
    report = benchmark_scaling_report(event_sizes=(300,), repeat=1)

    assert report.height == 1
    assert report["orders"][0] > 0
    assert report["vectorized_speedup_vs_naive"][0] > 0


def test_dashboard_data_preparation_runs_without_ui_dependencies() -> None:
    data = prepare_dashboard_data(n_events=400, seed=6)

    assert data["book"].height == 400
    assert data["baseline"].height == 400
    assert data["simulation"]["orders"].height > 0
    assert "fill_quality" in data
    assert "risk_summary" in data


def test_streaming_batches_and_snapshots() -> None:
    config = StreamingConfig(n_events=600, batch_size=150, window_events=400, seed=8)
    batches = list(stream_synthetic_lob(config))
    snapshots = run_streaming_simulation(config)
    frame = snapshots_to_frame(snapshots)

    assert len(batches) == 4
    assert frame.height == 4
    assert frame["events_seen"].max() <= 400
    assert frame["orders"].sum() > 0


def test_load_lob_file_normalizes_csv(tmp_path) -> None:
    raw = pl.DataFrame(
        {
            "timestamp": [2, 1, 3],
            "mid_price": [100.1, 100.0, 100.2],
            "bid_price": [100.0, 99.9, 100.1],
            "ask_price": [100.2, 100.1, 100.3],
            "bid_size": [100.0, 120.0, 110.0],
            "ask_size": [90.0, 95.0, 105.0],
        }
    )
    path = tmp_path / "lob.csv"
    raw.write_csv(path)

    loaded = load_lob_file(path)
    normalized = normalize_lob_frame(raw)

    assert loaded["timestamp"].to_list() == [1, 2, 3]
    assert set(["spread_bps", "volume", "volatility", "regime"]).issubset(loaded.columns)
    assert normalized.height == 3


def test_loader_accepts_binance_bookticker_aliases() -> None:
    raw = pl.DataFrame(
        {
            "event_time": [1, 2, 3],
            "best_bid_price": [99.9, 100.0, 100.1],
            "best_bid_qty": [10.0, 12.0, 11.0],
            "best_ask_price": [100.1, 100.2, 100.3],
            "best_ask_qty": [8.0, 7.0, 9.0],
        }
    )

    normalized = normalize_lob_frame(raw)

    assert normalized["timestamp"].to_list() == [1, 2, 3]
    assert [round(value, 4) for value in normalized["mid_price"].to_list()] == [100.0, 100.1, 100.2]
    assert "bid_size" in normalized.columns
    assert "ask_size" in normalized.columns


def test_ml_pipeline_trains_on_synthetic_data() -> None:
    book = generate_synthetic_lob(n_events=900, seed=9)
    market_frame = build_ml_market_frame(book, horizon=5)
    results = run_ml_research_pipeline(book, horizon=5)

    assert market_frame.height > 0
    assert set(results) == {"regime_detector", "price_direction", "fill_probability", "slippage"}
    assert results["regime_detector"].metrics["rows"] > 0
    assert results["fill_probability"].metrics["accuracy"] >= 0.0
