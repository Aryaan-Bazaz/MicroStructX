from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import Any

import polars as pl

from microstructx.data import generate_synthetic_lob
from microstructx.loaders import load_lob_file, normalize_lob_frame
from microstructx.ml import run_ml_research_pipeline
from microstructx.signals import add_market_features, vectorized_backtest
from microstructx.simulator import run_multi_agent_simulation


def prepare_dashboard_data(
    n_events: int = 10_000,
    seed: int = 7,
    book: pl.DataFrame | None = None,
    data_path: str | None = None,
) -> dict[str, Any]:
    """Build all datasets needed by the dashboard without importing UI libraries."""
    if book is None and data_path is not None:
        book = load_lob_file(data_path)
    if book is None:
        book = generate_synthetic_lob(n_events=n_events, seed=seed)
    else:
        book = normalize_lob_frame(book)

    featured = add_market_features(book)
    baseline = vectorized_backtest(featured, signal_column="momentum")
    simulation = run_multi_agent_simulation(book)
    executions = simulation["executions"]

    pnl_curve = (
        executions.sort("timestamp")
        .with_columns(pl.col("realized_pnl").cum_sum().alias("cumulative_realized_pnl"))
        .select(["timestamp", "agent", "realized_pnl", "cumulative_realized_pnl", "regime"])
        if not executions.is_empty()
        else pl.DataFrame()
    )

    fill_quality = (
        executions.group_by("agent")
        .agg(
            [
                pl.col("filled_qty").sum().alias("filled_qty"),
                pl.col("requested_qty").sum().alias("requested_qty"),
                pl.col("slippage").sum().alias("slippage"),
                pl.col("fees").sum().alias("fees"),
                pl.col("limit_fill_probability").mean().alias("avg_limit_fill_probability"),
                pl.col("queue_position_ahead").mean().alias("avg_queue_ahead"),
            ]
        )
        .with_columns((pl.col("filled_qty") / pl.col("requested_qty")).fill_nan(0.0).alias("fill_ratio"))
        .sort("fill_ratio", descending=True)
        if not executions.is_empty()
        else pl.DataFrame()
    )

    return {
        "book": book,
        "featured": featured,
        "baseline": baseline,
        "simulation": simulation,
        "executions": executions,
        "pnl_curve": pnl_curve,
        "fill_quality": fill_quality,
        "regime_execution_report": simulation["regime_execution_report"],
        "risk_summary": simulation["risk_summary"],
    }


def main() -> None:
    try:
        import plotly.express as px
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            'Dashboard dependencies are missing. Install them with: python -m pip install -e ".[dashboard]"'
        ) from exc

    st.set_page_config(page_title="MicroStructX Dashboard", layout="wide")
    st.title("MicroStructX Market Microstructure Dashboard")

    with st.sidebar:
        st.header("Simulation Controls")
        source = st.radio("Data source", ["Synthetic", "Upload CSV/Parquet/ZIP"])
        n_events = st.slider("Events", min_value=1_000, max_value=100_000, value=10_000, step=1_000)
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=7, step=1)
        uploaded = st.file_uploader("Historical LOB file", type=["csv", "parquet", "pq", "zip"]) if source.startswith("Upload") else None
        train_ml = st.checkbox("Train ML models", value=False)

    if uploaded is not None:
        if uploaded.name.endswith(".zip"):
            with NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
                temp_file.write(uploaded.getvalue())
                temp_path = temp_file.name
            data = prepare_dashboard_data(data_path=temp_path)
        elif uploaded.name.endswith(".csv"):
            uploaded_book = pl.read_csv(uploaded)
            data = prepare_dashboard_data(book=uploaded_book)
        else:
            uploaded_book = pl.read_parquet(uploaded)
            data = prepare_dashboard_data(book=uploaded_book)
    else:
        data = prepare_dashboard_data(n_events=n_events, seed=seed)
    simulation = data["simulation"]
    executions = data["executions"]
    baseline = data["baseline"]

    summary = simulation["execution_summary"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orders", f"{summary['orders']:,.0f}")
    col2.metric("Qty Fill Ratio", f"{summary['quantity_fill_ratio']:.2%}")
    col3.metric("Realized PnL", f"{summary['realized_pnl']:,.2f}")
    col4.metric("Baseline PnL", f"{baseline['cum_pnl'][-1]:,.2f}")

    st.caption(
        f"Order fill rate: {summary['order_fill_rate']:.2%} | "
        f"Full-fill rate: {summary['full_fill_rate']:.2%} | "
        f"Sharpe-like: {data['risk_summary']['sharpe_like']:.2f} | "
        f"Max drawdown: {data['risk_summary']['max_drawdown']:,.2f}"
    )

    st.subheader("Market Regimes")
    regime_counts = data["book"].group_by("regime").len().sort("len", descending=True)
    st.plotly_chart(px.bar(regime_counts.to_pandas(), x="regime", y="len"), use_container_width=True)

    st.subheader("Agent Realized PnL")
    agent_summary = simulation["agent_summary"]
    if not agent_summary.is_empty():
        st.plotly_chart(
            px.bar(agent_summary.to_pandas(), x="agent", y="realized_pnl", color="agent"),
            use_container_width=True,
        )
        st.dataframe(agent_summary.to_pandas(), use_container_width=True)

    st.subheader("Cumulative Realized PnL")
    pnl_curve = data["pnl_curve"]
    if not pnl_curve.is_empty():
        st.plotly_chart(
            px.line(pnl_curve.to_pandas(), x="timestamp", y="cumulative_realized_pnl", color="regime"),
            use_container_width=True,
        )

    st.subheader("Execution Quality")
    fill_quality = data["fill_quality"]
    if not fill_quality.is_empty():
        st.plotly_chart(
            px.bar(fill_quality.to_pandas(), x="agent", y="fill_ratio", color="agent"),
            use_container_width=True,
        )
        st.dataframe(fill_quality.to_pandas(), use_container_width=True)

    st.subheader("Slippage by Regime")
    regime_report = data["regime_execution_report"]
    if not regime_report.is_empty():
        st.plotly_chart(
            px.bar(regime_report.to_pandas(), x="regime", y="slippage_bps", color="regime"),
            use_container_width=True,
        )
        st.dataframe(regime_report.to_pandas(), use_container_width=True)

    if train_ml:
        st.subheader("ML Model Metrics")
        with st.spinner("Training ML models..."):
            try:
                ml_results = run_ml_research_pipeline(data["book"])
                rows = [
                    {"model": name, **result.metrics}
                    for name, result in ml_results.items()
                ]
                st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True)
            except ValueError as exc:
                st.warning(f"ML training skipped: {exc}")

    with st.expander("Raw Executions"):
        st.dataframe(executions.head(1_000).to_pandas(), use_container_width=True)


if __name__ == "__main__":
    main()
