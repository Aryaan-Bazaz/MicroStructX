from __future__ import annotations

import argparse
from pathlib import Path

from microstructx.data import generate_synthetic_lob
from microstructx.loaders import download_lob_dataset, load_lob_file
from microstructx.ml import run_ml_research_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MicroStructX ML models.")
    parser.add_argument("--path", type=str, default=None, help="Local CSV/Parquet/ZIP LOB dataset.")
    parser.add_argument("--url", type=str, default=None, help="Online CSV/ZIP dataset URL to download first.")
    parser.add_argument("--output", type=str, default="datasets/downloaded_lob.zip", help="Download path for --url.")
    parser.add_argument("--events", type=int, default=5_000, help="Synthetic events when no dataset is provided.")
    parser.add_argument("--seed", type=int, default=13, help="Synthetic data seed.")
    parser.add_argument("--horizon", type=int, default=10, help="Forward ticks for price direction label.")
    args = parser.parse_args()

    dataset_path = args.path
    if args.url:
        dataset_path = str(download_lob_dataset(args.url, Path(args.output)))

    if dataset_path:
        book = load_lob_file(dataset_path)
        source = dataset_path
    else:
        book = generate_synthetic_lob(n_events=args.events, seed=args.seed)
        source = "synthetic"

    results = run_ml_research_pipeline(book, horizon=args.horizon)

    print(f"MicroStructX ML research pipeline source={source}")
    for name, result in results.items():
        print(f"{name}:")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
