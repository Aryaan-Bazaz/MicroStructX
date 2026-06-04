from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import polars as pl


REQUIRED_LOB_COLUMNS = {
    "timestamp",
    "mid_price",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
}

BINANCE_BOOKTICKER_COLUMNS = [
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
]

COLUMN_ALIASES = {
    "timestamp": [
        "timestamp",
        "event_time",
        "transaction_time",
        "local_timestamp",
        "time",
        "ts",
        "E",
        "T",
    ],
    "bid_price": [
        "bid_price",
        "best_bid_price",
        "bidPrice",
        "bid_px",
        "b",
        "bids[0].price",
    ],
    "ask_price": [
        "ask_price",
        "best_ask_price",
        "askPrice",
        "ask_px",
        "a",
        "asks[0].price",
    ],
    "bid_size": [
        "bid_size",
        "best_bid_qty",
        "best_bid_quantity",
        "bid_qty",
        "bidQty",
        "bid_amount",
        "B",
        "bids[0].amount",
    ],
    "ask_size": [
        "ask_size",
        "best_ask_qty",
        "best_ask_quantity",
        "ask_qty",
        "askQty",
        "ask_amount",
        "A",
        "asks[0].amount",
    ],
    "mid_price": ["mid_price", "mid", "midpoint"],
}


def _can_normalize(frame: pl.DataFrame) -> bool:
    return REQUIRED_LOB_COLUMNS.issubset(set(_with_aliases(frame).columns))


def _read_csv_bytes(data: bytes, max_rows: int | None = None) -> pl.DataFrame:
    with_header = pl.read_csv(BytesIO(data), n_rows=max_rows)
    if _can_normalize(with_header):
        return with_header

    headerless = pl.read_csv(BytesIO(data), has_header=False, n_rows=max_rows)
    if headerless.width == len(BINANCE_BOOKTICKER_COLUMNS):
        return headerless.rename(
            {old: new for old, new in zip(headerless.columns, BINANCE_BOOKTICKER_COLUMNS, strict=True)}
        )
    return with_header


def _with_aliases(frame: pl.DataFrame) -> pl.DataFrame:
    existing = set(frame.columns)
    expressions = []
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in existing:
            continue
        source = next((alias for alias in aliases if alias in existing), None)
        if source is not None:
            expressions.append(pl.col(source).alias(canonical))
    if expressions:
        frame = frame.with_columns(expressions)
    if "mid_price" not in frame.columns and {"bid_price", "ask_price"}.issubset(frame.columns):
        frame = frame.with_columns(((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid_price"))
    return frame


def normalize_lob_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalize historical level-1 LOB/tick data."""
    frame = _with_aliases(frame)
    missing = REQUIRED_LOB_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"LOB data missing columns: {sorted(missing)}")

    normalized = frame.with_columns(
        [
            pl.col("timestamp").cast(pl.Int64),
            pl.col("mid_price").cast(pl.Float64),
            pl.col("bid_price").cast(pl.Float64),
            pl.col("ask_price").cast(pl.Float64),
            pl.col("bid_size").cast(pl.Float64),
            pl.col("ask_size").cast(pl.Float64),
        ]
    ).filter(
        (pl.col("mid_price").is_finite())
        & (pl.col("bid_price").is_finite())
        & (pl.col("ask_price").is_finite())
        & (pl.col("bid_size").is_finite())
        & (pl.col("ask_size").is_finite())
        & (pl.col("mid_price") > 0)
        & (pl.col("bid_price") > 0)
        & (pl.col("ask_price") > 0)
        & (pl.col("bid_size") >= 0)
        & (pl.col("ask_size") >= 0)
    ).sort("timestamp")

    if "spread_bps" not in normalized.columns:
        normalized = normalized.with_columns(
            ((pl.col("ask_price") - pl.col("bid_price")) / pl.col("mid_price") * 10_000.0).alias("spread_bps")
        )
    if "volume" not in normalized.columns:
        normalized = normalized.with_columns(((pl.col("bid_size") + pl.col("ask_size")) / 20.0).alias("volume"))
    if "volatility" not in normalized.columns:
        normalized = normalized.with_columns(pl.col("mid_price").pct_change().rolling_std(50).fill_null(0.0).alias("volatility"))
    if "regime" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.when(pl.col("spread_bps") > 6.0)
            .then(pl.lit("stress"))
            .when(pl.col("volatility") > pl.col("volatility").quantile(0.75))
            .then(pl.lit("volatile"))
            .otherwise(pl.lit("normal"))
            .alias("regime")
        )

    return normalized.select(
        [
            "timestamp",
            "mid_price",
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "spread_bps",
            "volume",
            "volatility",
            "regime",
        ]
    )


def load_lob_file(path: str | Path, max_rows: int | None = None) -> pl.DataFrame:
    """Load historical LOB/tick data from CSV, Parquet, or a ZIP containing CSV."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = _read_csv_bytes(file_path.read_bytes(), max_rows=max_rows)
    elif suffix in {".parquet", ".pq"}:
        frame = pl.read_parquet(file_path)
        if max_rows is not None:
            frame = frame.head(max_rows)
    elif suffix == ".zip":
        with ZipFile(file_path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("ZIP file does not contain a CSV file")
            with archive.open(csv_names[0]) as csv_file:
                frame = _read_csv_bytes(csv_file.read(), max_rows=max_rows)
    else:
        raise ValueError("supported LOB file formats: .csv, .parquet, .pq, .zip")
    return normalize_lob_frame(frame)


def download_lob_dataset(url: str, output_path: str | Path) -> Path:
    """Download an online dataset file so it can be loaded with load_lob_file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path
