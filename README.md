# MicroStructX

Adaptive multi-agent market microstructure backtesting engine.

This repository currently implements phases 1-7:

- Phase 1: vectorized portfolio backtester on synthetic level-1 order book data.
- Phase 2: order-book-aware execution with spread crossing, queue pressure, partial fills, transaction costs, and impact.
- Phase 3: multi-agent simulation with momentum, mean reversion, market maker, arbitrageur, and noise agents.
- Phase 4: explicit queue-position modeling with latency-aware queue-ahead, queue depletion, and fill probability.
- Phase 5: performance benchmarking against a naive Python loop and a Numba fill kernel.
- Phase 6: Streamlit/Plotly dashboard for PnL, regimes, agents, fills, and execution quality.
- Phase 7: realtime-style streaming mode using chunked synthetic tick ingestion and rolling-window simulation.
- ML layer: regime clustering, price direction classification, fill probability prediction, and slippage regression.
- Low-level systems layer: standalone C++17 execution core with CMake and microbenchmarking.

The default demo still uses synthetic data, but the engine now also accepts historical level-1 LOB/tick data from CSV or Parquet.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python -m microstructx.cli --events 10000
python -m microstructx.cli --url "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-05-16.zip" --output datasets\BTCUSDT-bookTicker-2023-05-16.zip --max-rows 50000 --ml
python examples\run_benchmarks.py
python examples\run_streaming.py
python examples\research_slippage_by_regime.py
python examples\run_ml_pipeline.py
```

Build and run the optional C++ execution core:

```powershell
cmake -S cpp -B cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp\build --config Release
.\cpp\build\Release\microstructx_bench.exe --events 1000000 --orders 1000000
```

Or run the example:

```powershell
python examples\run_phase_1_3.py
```

The dashboard is optional. For CLI-only usage, use `python -m microstructx.cli`.

For the optional dashboard:

```powershell
python -m pip install -e ".[dashboard]"
streamlit run src\microstructx\dashboard.py
```

## CLI On Online Dataset

Run directly on an online Binance public dataset:

```powershell
python -m microstructx.cli --url "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-05-16.zip" --output datasets\BTCUSDT-bookTicker-2023-05-16.zip --max-rows 50000
```

Add ML model training:

```powershell
python -m microstructx.cli --url "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-05-16.zip" --output datasets\BTCUSDT-bookTicker-2023-05-16.zip --max-rows 50000 --ml
```

Run again from the downloaded local file:

```powershell
python -m microstructx.cli --path datasets\BTCUSDT-bookTicker-2023-05-16.zip --max-rows 50000 --ml
```

The CLI prints baseline PnL, execution metrics, risk metrics, per-agent results, regime execution quality, and optional ML metrics.

## Architecture

```text
Synthetic/Historical LOB Data
          ↓
Feature Engine
          ↓
Vectorized Backtester
          ↓
Agent-Based Strategy Layer
          ↓
Order-Book Execution Simulator
          ↓
PnL + Risk Analytics
          ↓
Dashboard / Streaming
          ↓
C++ Execution Core
```

## Implemented Components

### Phase 1: Vectorized Backtester

`microstructx.data.generate_synthetic_lob` creates synthetic tick/order-book events with:

- mid price
- bid/ask prices
- bid/ask depth
- spread
- volume
- volatility
- market regime

`microstructx.loaders.load_lob_file` loads real/historical CSV, Parquet, or ZIP files with these canonical columns:

- `timestamp`
- `mid_price`
- `bid_price`
- `ask_price`
- `bid_size`
- `ask_size`

Optional columns are `spread_bps`, `volume`, `volatility`, and `regime`. Missing optional columns are derived automatically.

The loader also accepts common online best-bid/ask schemas such as Binance/Tardis-style columns:

- `event_time`, `transaction_time`, `local_timestamp`, or `time` for timestamp
- `best_bid_price`, `bidPrice`, `bids[0].price` for bid price
- `best_ask_price`, `askPrice`, `asks[0].price` for ask price
- `best_bid_qty`, `bidQty`, `bids[0].amount` for bid size
- `best_ask_qty`, `askQty`, `asks[0].amount` for ask size

`microstructx.signals.add_market_features` adds:

- one-step returns
- rolling momentum
- rolling z-score
- order book imbalance

`microstructx.signals.vectorized_backtest` converts a continuous signal into positions and produces gross PnL, fees, net PnL, cumulative PnL, and equity.

### Phase 2: Order-Book-Aware Execution

`microstructx.execution.execute_orders` models:

- market and limit orders
- latency in event steps
- spread crossing
- partial fills
- missed passive limit orders
- queue pressure
- same-side queue depth
- queue position ahead
- queue depletion
- limit fill probability
- temporary impact
- permanent impact
- fees
- next-tick realized PnL

### Phase 3: Multi-Agent Simulation

`microstructx.simulator.run_multi_agent_simulation` runs these agents:

- `MomentumAgent`
- `MeanReversionAgent`
- `MarketMakerAgent`
- `ArbitrageurAgent`
- `NoiseAgent`

It returns market features, generated orders, executions, per-agent summary, per-regime summary, and global execution metrics.

### Phase 4: Queue Position Modeling

`ExecutionConfig` controls the queue model:

- `queue_ahead_fraction`: estimated fraction of same-side depth ahead of the order.
- `latency_queue_penalty`: extra queue-ahead penalty per latency event.
- `cancellation_rate`: depth removed from cancellations before trade-through.
- `min_limit_fill_probability`: lower bound for probabilistic partial fills.

The execution output includes `same_side_depth`, `opposite_depth`, `queue_position_ahead`, `queue_depletion`, and `limit_fill_probability`.

Important dashboard metrics:

- `order_fill_rate`: fraction of orders with any fill.
- `full_fill_rate`: fraction of orders filled completely.
- `quantity_fill_ratio`: total filled quantity divided by requested quantity.
- `avg_queue_ahead`: average simulated queue size ahead of each order.

### Phase 5: Performance Benchmarking

`microstructx.benchmarks` provides:

- `naive_execute_orders`: readable Python-loop reference implementation.
- `execute_orders`: vectorized Polars implementation used by the simulator.
- `numba_estimate_fills`: Numba-accelerated queue-aware fill kernel.
- `benchmark_execution`: timing harness with speedup metrics.
- `benchmark_scaling_report`: benchmark table across multiple event sizes.

Run:

```powershell
python examples\run_benchmarks.py
```

### Phase 6: Dashboard

`microstructx.dashboard.prepare_dashboard_data` builds dashboard-ready datasets without importing UI libraries. `microstructx.dashboard.main` launches a Streamlit dashboard with:

- market regime counts
- agent realized PnL
- cumulative realized PnL
- execution quality and fill ratio
- raw execution inspection
- CSV/Parquet/ZIP upload for historical LOB data
- order fill, full-fill, and quantity fill metrics
- slippage by regime

Run:

```powershell
streamlit run src\microstructx\dashboard.py
```

### Phase 7: Streaming Mode

`microstructx.streaming` provides:

- `StreamingConfig`: controls event count, batch size, rolling window, seed, and pacing.
- `stream_synthetic_lob`: yields timestamp-ordered LOB batches.
- `run_streaming_simulation`: runs rolling-window multi-agent simulations over incoming batches.
- `snapshots_to_frame`: converts live snapshots to a Polars DataFrame.

Run:

```powershell
python examples\run_streaming.py
```

## Research Experiment

Run a compact slippage-by-regime experiment:

```powershell
python examples\research_slippage_by_regime.py
```

This prints realized PnL, Sharpe-like score, max drawdown, average slippage bps, quantity fill ratio, and per-regime execution quality.

## Machine Learning Layer

`microstructx.ml` adds classical ML models on top of the simulator:

- `train_regime_detector`: lightweight KMeans-style clustering for learned market regimes.
- `train_price_direction_model`: random forest classification for short-horizon mid-price direction.
- `train_fill_probability_model`: random forest classification for whether an order receives any fill.
- `train_slippage_model`: random forest regression for expected absolute slippage bps.
- `run_ml_research_pipeline`: trains all models from one synthetic or historical LOB dataset.

Run on synthetic data:

```powershell
python examples\run_ml_pipeline.py --events 5000
```

Run on a local online dataset file after downloading:

```powershell
python examples\run_ml_pipeline.py --path datasets\BTCUSDT-bookTicker-2023-05-16.zip
```

Or download and train in one command:

```powershell
python examples\run_ml_pipeline.py --url "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2023-05-16.zip" --output datasets\BTCUSDT-bookTicker-2023-05-16.zip
```

## Low-Level C++ Execution Core

The `cpp/` directory adds a standalone C++17 systems layer:

- cache-friendly `BboEvent`, `Order`, and `FillResult` structs
- queue-aware market/limit order execution logic
- partial fills, queue-ahead, queue depletion, fees, slippage, and impact
- lightweight CSV reader for canonical BBO CSVs or headerless Binance `bookTicker` CSVs
- CMake build with `microstructx_core` static library
- `microstructx_bench` executable for throughput benchmarking

Build:

```powershell
cmake -S cpp -B cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp\build --config Release
```

Run synthetic C++ benchmark:

```powershell
.\cpp\build\Release\microstructx_bench.exe --events 1000000 --orders 1000000
```

If CMake is not installed but `g++` is available:

```powershell
New-Item -ItemType Directory -Force cpp\build
g++ -std=c++17 -O2 -Icpp\include cpp\src\execution_core.cpp cpp\src\csv_loader.cpp cpp\benchmarks\execution_benchmark.cpp -o cpp\build\microstructx_bench.exe
.\cpp\build\microstructx_bench.exe --events 100000 --orders 100000
```

Run on a CSV file:

```powershell
.\cpp\build\Release\microstructx_bench.exe --csv path\to\bbo.csv --events 1000000 --orders 1000000
```

This makes the project suitable to discuss as quant research infrastructure plus low-level execution systems work.

## Current Assumptions

- The bundled demo data is synthetic and is meant for engine validation, not trading conclusions.
- Real-market use requires loading historical tick/LOB data through `load_lob_file`.
- Queue position is an approximation from level-1 depth, not full market-by-order queue reconstruction.
- Transaction cost, impact, and latency parameters are configurable and should be calibrated per venue/instrument.

## Future Extensions

- Kafka or Redpanda adapter for real tick ingestion.
- ITCH/LOBSTER-specific historical data adapters.
- Optional Rust execution core for a lower-level systems extension.
