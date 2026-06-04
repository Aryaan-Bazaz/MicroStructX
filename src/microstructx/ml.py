from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from microstructx.signals import add_market_features
from microstructx.simulator import run_multi_agent_simulation


MARKET_FEATURES = [
    "return_1",
    "momentum",
    "book_imbalance",
    "zscore",
    "spread_bps",
    "volume",
    "volatility",
]

EXECUTION_FEATURES = [
    "requested_qty",
    "exec_mid_price",
    "queue_pressure",
    "same_side_depth",
    "opposite_depth",
    "queue_position_ahead",
    "queue_depletion",
    "limit_fill_probability",
]


@dataclass(frozen=True)
class ModelResult:
    model: Any
    features: list[str]
    metrics: dict[str, float]
    predictions: pl.DataFrame


def _feature_matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    return frame.select(features).fill_nan(0.0).fill_null(0.0).to_numpy()


def _can_stratify(labels: np.ndarray) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return len(counts) > 1 and counts.min() >= 2


def _standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std


def _simple_kmeans(x: np.ndarray, n_clusters: int, random_state: int, max_iter: int = 50) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if x.shape[0] < n_clusters:
        raise ValueError("not enough rows for requested regimes")

    centroids = x[rng.choice(x.shape[0], size=n_clusters, replace=False)]
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster in range(n_clusters):
            members = x[labels == cluster]
            if len(members) > 0:
                centroids[cluster] = members.mean(axis=0)
    return labels


def _cluster_separation_score(x: np.ndarray, labels: np.ndarray) -> float:
    overall = x.mean(axis=0)
    within = 0.0
    between = 0.0
    for label in np.unique(labels):
        members = x[labels == label]
        centroid = members.mean(axis=0)
        within += float(((members - centroid) ** 2).sum())
        between += float(len(members) * ((centroid - overall) ** 2).sum())
    return between / (within + 1e-12)


def build_ml_market_frame(book: pl.DataFrame, horizon: int = 10) -> pl.DataFrame:
    """Create ML-ready market features and a short-horizon direction label."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    return (
        add_market_features(book)
        .with_columns(
            [
                ((pl.col("mid_price").shift(-horizon) / pl.col("mid_price")) - 1.0).alias("future_return"),
            ]
        )
        .with_columns(
            pl.when(pl.col("future_return") > 0)
            .then(1)
            .when(pl.col("future_return") < 0)
            .then(-1)
            .otherwise(0)
            .alias("future_direction")
        )
        .drop_nulls(MARKET_FEATURES + ["future_return", "future_direction"])
    )


def train_regime_detector(
    market_frame: pl.DataFrame,
    n_regimes: int = 4,
    random_state: int = 7,
) -> ModelResult:
    """Cluster market states into learned regimes using KMeans."""
    if market_frame.height < n_regimes + 2:
        raise ValueError("not enough rows to train regime detector")

    features = [feature for feature in MARKET_FEATURES if feature in market_frame.columns]
    x = _standardize(_feature_matrix(market_frame, features))
    labels = _simple_kmeans(x, n_clusters=n_regimes, random_state=random_state)
    score = _cluster_separation_score(x, labels)
    predictions = market_frame.select(["timestamp", "mid_price", "regime"]).with_columns(
        pl.Series("ml_regime", labels.astype(int))
    )

    return ModelResult(
        model={"type": "simple_kmeans", "n_regimes": n_regimes, "random_state": random_state},
        features=features,
        metrics={"cluster_separation": score, "rows": float(market_frame.height), "n_regimes": float(n_regimes)},
        predictions=predictions,
    )


def train_price_direction_model(
    market_frame: pl.DataFrame,
    random_state: int = 7,
) -> ModelResult:
    """Train a short-horizon up/down classifier from market microstructure features."""
    features = [feature for feature in MARKET_FEATURES if feature in market_frame.columns]
    model_frame = market_frame.filter(pl.col("future_direction") != 0)
    if model_frame.height < 20:
        raise ValueError("not enough non-flat labels to train price direction model")

    x = _feature_matrix(model_frame, features)
    y = model_frame["future_direction"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("price direction model needs both up and down examples")
    stratify = y if _can_stratify(y) else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=random_state, stratify=stratify)
    model = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    predictions = model_frame.select(["timestamp", "future_direction"]).with_columns(
        pl.Series("predicted_direction", model.predict(x).astype(int))
    )
    return ModelResult(
        model=model,
        features=features,
        metrics={"accuracy": float(accuracy_score(y_test, y_pred)), "rows": float(model_frame.height)},
        predictions=predictions,
    )


def train_fill_probability_model(
    executions: pl.DataFrame,
    random_state: int = 7,
) -> ModelResult:
    """Train a classifier that predicts whether an order receives any fill."""
    if executions.height < 20:
        raise ValueError("not enough executions to train fill probability model")

    features = [feature for feature in EXECUTION_FEATURES if feature in executions.columns]
    model_frame = executions.with_columns((pl.col("filled_qty") > 0).cast(pl.Int64).alias("filled_label"))
    x = _feature_matrix(model_frame, features)
    y = model_frame["filled_label"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("fill probability model needs both filled and unfilled examples")
    stratify = y if _can_stratify(y) else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=random_state, stratify=stratify)
    model = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    probability = model.predict_proba(x)[:, 1]

    predictions = model_frame.select(["timestamp", "agent", "filled_label"]).with_columns(
        pl.Series("predicted_fill_probability", probability)
    )
    return ModelResult(
        model=model,
        features=features,
        metrics={"accuracy": float(accuracy_score(y_test, y_pred)), "rows": float(executions.height)},
        predictions=predictions,
    )


def train_slippage_model(
    executions: pl.DataFrame,
    random_state: int = 7,
) -> ModelResult:
    """Train a regression model for absolute slippage in basis points."""
    if executions.height < 20:
        raise ValueError("not enough executions to train slippage model")

    features = [feature for feature in EXECUTION_FEATURES if feature in executions.columns]
    model_frame = executions.with_columns(
        (pl.col("slippage").abs() / pl.col("notional").clip(1e-12, None) * 10_000.0).alias("slippage_bps_abs")
    )
    x = _feature_matrix(model_frame, features)
    y = model_frame["slippage_bps_abs"].fill_nan(0.0).fill_null(0.0).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=random_state)
    model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    predictions = model_frame.select(["timestamp", "agent", "slippage_bps_abs"]).with_columns(
        pl.Series("predicted_slippage_bps_abs", model.predict(x))
    )
    return ModelResult(
        model=model,
        features=features,
        metrics={
            "mae_bps": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "rows": float(executions.height),
        },
        predictions=predictions,
    )


def run_ml_research_pipeline(book: pl.DataFrame, horizon: int = 10) -> dict[str, ModelResult]:
    """Train all ML components against one historical or synthetic LOB dataset."""
    market_frame = build_ml_market_frame(book, horizon=horizon)
    simulation = run_multi_agent_simulation(book)
    executions = simulation["executions"]
    return {
        "regime_detector": train_regime_detector(market_frame),
        "price_direction": train_price_direction_model(market_frame),
        "fill_probability": train_fill_probability_model(executions),
        "slippage": train_slippage_model(executions),
    }
