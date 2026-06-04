from microstructx.streaming import StreamingConfig, run_streaming_simulation, snapshots_to_frame


def main() -> None:
    config = StreamingConfig(n_events=2_000, batch_size=250, window_events=1_000, seed=77)
    snapshots = run_streaming_simulation(config)
    frame = snapshots_to_frame(snapshots)

    print("MicroStructX streaming simulation")
    for row in frame.iter_rows(named=True):
        print(
            "  "
            + " | ".join(
                [
                    f"batch={row['batch_id']:.0f}",
                    f"last_ts={row['last_timestamp']:.0f}",
                    f"events_seen={row['events_seen']:.0f}",
                    f"orders={row['orders']:.0f}",
                    f"fill_rate={row['fill_rate']:.2%}",
                    f"realized_pnl={row['realized_pnl']:.2f}",
                    f"avg_spread_bps={row['avg_spread_bps']:.2f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
