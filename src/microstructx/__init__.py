"""MicroStructX: multi-agent market microstructure backtesting."""

from microstructx.agents import (
    ArbitrageurAgent,
    MarketMakerAgent,
    MeanReversionAgent,
    MomentumAgent,
    NoiseAgent,
)
from microstructx.analytics import regime_execution_report, summarize_execution_risk, summarize_executions
from microstructx.benchmarks import benchmark_execution, benchmark_scaling_report, naive_execute_orders, numba_estimate_fills
from microstructx.data import generate_synthetic_lob
from microstructx.dashboard import prepare_dashboard_data
from microstructx.execution import ExecutionConfig, execute_orders
from microstructx.loaders import download_lob_dataset, load_lob_file, normalize_lob_frame
from microstructx.ml import (
    ModelResult,
    build_ml_market_frame,
    run_ml_research_pipeline,
    train_fill_probability_model,
    train_price_direction_model,
    train_regime_detector,
    train_slippage_model,
)
from microstructx.simulator import SimulationConfig, run_multi_agent_simulation
from microstructx.streaming import StreamingConfig, run_streaming_simulation, snapshots_to_frame, stream_synthetic_lob

__all__ = [
    "ArbitrageurAgent",
    "ExecutionConfig",
    "MarketMakerAgent",
    "MeanReversionAgent",
    "MomentumAgent",
    "NoiseAgent",
    "SimulationConfig",
    "StreamingConfig",
    "ModelResult",
    "benchmark_execution",
    "benchmark_scaling_report",
    "build_ml_market_frame",
    "download_lob_dataset",
    "execute_orders",
    "generate_synthetic_lob",
    "load_lob_file",
    "naive_execute_orders",
    "normalize_lob_frame",
    "numba_estimate_fills",
    "prepare_dashboard_data",
    "regime_execution_report",
    "run_ml_research_pipeline",
    "run_multi_agent_simulation",
    "run_streaming_simulation",
    "snapshots_to_frame",
    "stream_synthetic_lob",
    "summarize_execution_risk",
    "summarize_executions",
    "train_fill_probability_model",
    "train_price_direction_model",
    "train_regime_detector",
    "train_slippage_model",
]
