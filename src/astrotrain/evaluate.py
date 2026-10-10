"""
Evaluate a saved countdown clock model on LAMP files, reporting error in seconds
for the model and for the published schedule, overall and by route.

The schedule's estimate for a stop-to-stop travel time is its arrival-to-arrival
time minus the dwell the train just finished, which is known when the train
leaves the stop (the moment the model predicts from).

Use days the model wasn't trained on. train() validates on the last 20% of its
files, so later days are the cleanest test.

usage: uv run python -m astrotrain.evaluate data/2026-10-0*.parquet
"""

import argparse

import pandas as pd

from astrotrain.train_lstm import CountdownClockLSTM


def evaluate(lstm: CountdownClockLSTM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict every window in raw LAMP data, returning one row per prediction with
    its route and the actual, predicted and scheduled travel time in seconds.
    """
    df = lstm.preprocess_data(df)
    X, y, end_rows = lstm.create_sequences(df, return_index=True)
    window_ends = df.loc[end_rows]
    return pd.DataFrame(
        {
            "route_id": window_ends["route_id"].to_numpy(),
            "actual": y.ravel(),
            "predicted": lstm.predict(X).ravel(),
            "scheduled": window_ends["target_scheduled_travel_time"].to_numpy(),
        }
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """
    Absolute error in seconds for the model and the schedule, for all routes
    ("all") and each route. Both are measured on the same predictions: the ones
    that have a scheduled time.
    """
    compared = results.dropna(subset=["scheduled"])
    errors = pd.DataFrame(
        {
            "route_id": compared["route_id"],
            "model": (compared["predicted"] - compared["actual"]).abs(),
            "schedule": (compared["scheduled"] - compared["actual"]).abs(),
        }
    )
    with_overall = pd.concat(
        [errors.assign(route_id="all"), errors.sort_values("route_id")]
    )
    return (
        with_overall.groupby("route_id", sort=False)
        .agg(
            predictions=("model", "size"),
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
    summary = summarize(results)

    compared = summary.loc["all", "predictions"]
    print(
        f"\nCompared {compared} of {len(results)} predictions "
        "(the rest have no scheduled time). Errors in seconds:\n"
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
