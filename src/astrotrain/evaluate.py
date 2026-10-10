"""
Evaluate a saved countdown clock model on LAMP files, reporting error in seconds
for the model and for the published schedule, by how many stops ahead the
prediction is and by route.

Predictions are made as a train leaves a stop: how long until it arrives at each
of the next stops. The schedule's estimate is the sum of the scheduled
arrival-to-arrival times along the way, minus the dwell the train just finished.

Use days the model wasn't trained on. train() validates on the last 20% of its
files, so later days are the cleanest test.

usage: uv run python -m astrotrain.evaluate data/2026-10-0*.parquet
"""

import argparse

import numpy as np
import pandas as pd

from astrotrain.train_lstm import CountdownClockLSTM


def evaluate(lstm: CountdownClockLSTM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict every window in raw LAMP data, returning one row per known target:
    its route, how many stops ahead it is, and the actual, predicted and
    scheduled seconds from leaving the stop to arriving there.
    """
    df = lstm.preprocess_data(df)
    X, y, end_rows = lstm.create_sequences(df, return_index=True)
    horizon = lstm.prediction_horizon
    results = pd.DataFrame(
        {
            "route_id": np.repeat(df.loc[end_rows, "route_id"].to_numpy(), horizon),
            "stops_ahead": np.tile(np.arange(1, horizon + 1), len(X)),
            "actual": y.ravel(),
            "predicted": lstm.predict(X).ravel(),
            "scheduled": df.loc[end_rows, lstm.scheduled_target_columns]
            .to_numpy()
            .ravel(),
        }
    )
    return results.dropna(subset=["actual"]).reset_index(drop=True)


def summarize(results: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    Absolute error in seconds for the model and the schedule, overall ("all")
    and for each value of `by`. Both are measured on the same targets: the ones
    that have a scheduled time.
    """
    compared = results.dropna(subset=["scheduled"])
    errors = pd.DataFrame(
        {
            by: compared[by],
            "actual": compared["actual"],
            "model": (compared["predicted"] - compared["actual"]).abs(),
            "schedule": (compared["scheduled"] - compared["actual"]).abs(),
        }
    )
    with_overall = pd.concat([errors.assign(**{by: "all"}), errors.sort_values(by)])
    return (
        with_overall.groupby(by, sort=False)
        .agg(
            targets=("model", "size"),
            actual_median=("actual", "median"),
            model_mae=("model", "mean"),
            model_median=("model", "median"),
            model_p90=("model", lambda e: e.quantile(0.9)),
            schedule_mae=("schedule", "mean"),
            schedule_median=("schedule", "median"),
        )
        .round(1)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Report a saved model's error in seconds against the schedule"
    )
    parser.add_argument("files", nargs="+", help="LAMP parquet files to evaluate on")
    parser.add_argument(
        "--model", default="models/countdown_lstm", help="Saved model directory"
    )
    args = parser.parse_args()

    lstm = CountdownClockLSTM()
    lstm.load(args.model)
    results = evaluate(lstm, lstm.load_data(sorted(args.files)))

    by_stops = summarize(results, "stops_ahead")
    compared = by_stops.loc["all", "targets"]
    print(
        f"\nCompared {compared} of {len(results)} targets (the rest have no "
        "scheduled time). Times and errors in seconds.\n"
    )
    print(by_stops.to_string())
    print()
    print(summarize(results, "route_id").to_string())


if __name__ == "__main__":
    main()
