"""
Evaluate a saved countdown clock model on LAMP files, reporting error in seconds
for the model and for the published schedule, by how many stops ahead the
prediction is and by route.

Predictions are made as a train leaves a stop: how long until it arrives at each
of the next stops. The schedule's estimate is the sum of the scheduled
arrival-to-arrival times along the way, minus the dwell the train just finished.

It can also score the model the way the MBTA scores its own published
predictions (--mbta-accuracy, see mbta_comparison) for the MBTA weeks the
evaluated days fully cover.

Use days the model wasn't trained on. train() validates on the last 20% of its
files, so later days are the cleanest test.

--model can be a saved LightGBM model (the default, models/countdown_gbm) or
LSTM model.

usage: uv run python -m astrotrain.evaluate data/2026-10-0*.parquet
       uv run python -m astrotrain.evaluate --mbta-accuracy \
           data/mbta_prediction_accuracy.csv data/2026-08-2*.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from astrotrain.gbm import CountdownClockGBM
from astrotrain.train_lstm import CountdownClockLSTM


def evaluate(lstm: CountdownClockLSTM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict every window in raw LAMP data, returning one row per known target:
    its trip, route, the stop it's for and how many stops ahead that is, when the
    prediction is made (departure_time, as the train leaves its current stop),
    and the actual, predicted and scheduled seconds from leaving to arriving.
    """
    df = lstm.preprocess_data(df)
    X, y, end_rows = lstm.create_sequences(df, return_index=True)
    horizon = lstm.prediction_horizon
    window_ends = df.loc[end_rows]
    stops_ahead = np.tile(np.arange(1, horizon + 1), len(X))

    # A known target k stops ahead is k rows further down the same trip, since
    # preprocess_data sorts each trip's stops together (RangeIndex)
    target_rows = np.minimum(np.repeat(end_rows, horizon) + stops_ahead, len(df) - 1)

    results = pd.DataFrame(
        {
            "service_date": np.repeat(window_ends["service_date"].to_numpy(), horizon),
            "trip_id": np.repeat(window_ends["trip_id"].to_numpy(), horizon),
            "route_id": np.repeat(window_ends["route_id"].to_numpy(), horizon),
            "stop_id": df["stop_id"].to_numpy()[target_rows],
            "stops_ahead": stops_ahead,
            "departure_time": np.repeat(
                window_ends["departure_time"].to_numpy(), horizon
            ),
            "actual": y.ravel(),
            "predicted": lstm.predict(X).ravel(),
            "scheduled": window_ends[lstm.scheduled_target_columns].to_numpy().ravel(),
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


# How the MBTA scores its own published predictions. Predictions are grouped by
# how far ahead they were made, and count as accurate if the train arrives within
# the group's window: (group, up to this many seconds ahead, seconds early
# allowed, seconds late allowed).
MBTA_ACCURACY_BINS = [
    ("0-3 min", 3 * 60, 60, 60),
    ("3-6 min", 6 * 60, 90, 120),
    ("6-12 min", 12 * 60, 150, 210),
    ("12-30 min", 30 * 60, 240, 360),
]


def score_like_mbta(results: pd.DataFrame, prediction: str) -> pd.DataFrame:
    """
    Put each prediction in the MBTA's group for how far ahead it is (its predicted
    seconds to arrival) and mark it accurate if the train arrived within that
    group's window. Predictions more than 30 minutes ahead are in no group and
    are dropped, as the MBTA's figures stop at 30 minutes.
    """
    labels = np.array([b[0] for b in MBTA_ACCURACY_BINS])
    upper, early, late = (
        np.array([b[i] for b in MBTA_ACCURACY_BINS]) for i in (1, 2, 3)
    )
    ahead = results[prediction].to_numpy()
    group = np.searchsorted(upper, ahead, side="right")
    in_group = (ahead >= 0) & (group < len(labels))
    group = group[in_group]
    later_than_predicted = results["actual"].to_numpy()[in_group] - ahead[in_group]
    accurate = (later_than_predicted >= -early[group]) & (
        later_than_predicted <= late[group]
    )
    return pd.DataFrame(
        {
            "service_date": results["service_date"].to_numpy()[in_group],
            "route_id": results["route_id"].to_numpy()[in_group],
            "bin": labels[group],
            "accurate": accurate,
        }
    )


def mbta_weeks(service_dates) -> list:
    """
    The MBTA weeks completely covered by these service dates (YYYYMMDD). Its
    weeks run Friday through Thursday and are labeled with the following Friday.
    """
    dates = pd.Series(
        pd.to_datetime(pd.Series(service_dates).astype(str), format="%Y%m%d").unique()
    )
    week_start = dates - pd.to_timedelta((dates.dt.weekday - 4) % 7, unit="D")
    label = week_start + pd.Timedelta(days=7)
    days_seen = dates.groupby(label).nunique()
    return sorted(days_seen[days_seen == 7].index.strftime("%Y-%m-%d"))


def mbta_comparison(results: pd.DataFrame, mbta: pd.DataFrame) -> pd.DataFrame:
    """
    Percent of predictions accurate by the MBTA's rules, ours and the MBTA's own,
    by line and group, for the MBTA weeks the results fully cover. `mbta` is the
    MBTA's Rapid Transit and Bus Prediction Accuracy Data CSV.

    Each line also gets an "all" row. The MBTA makes far more long-range
    predictions than we do (it keeps re-predicting every arrival), so our "all"
    weights each group by the MBTA's own count of predictions in it.

    This overstates our advantage, so treat it as context, not a result: the
    MBTA scores predictions made at any moment, including trains waiting at
    terminals, held or stuck between stations; ours are made only as a train
    leaves a stop, the easiest moment. On 2026-08-21..09-03 even the plain
    schedule, scored this way on our moments, "beat" the MBTA by 10-20 points.
    For a fair comparison use --mbta-predictions (mbta_predictions.py), which
    compares with the MBTA's own predictions made at the same moments.
    """
    weeks = mbta_weeks(results["service_date"])
    if not weeks:
        raise ValueError(
            "No complete MBTA week (Friday-Thursday) in the evaluated days"
        )
    week_dates = [date for week in weeks for date in _service_dates_in(week)]
    in_weeks = results[results["service_date"].isin(week_dates)]

    keys = ["route_id", "bin"]
    ours = score_like_mbta(in_weeks, "predicted").groupby(keys)["accurate"]
    schedule = score_like_mbta(in_weeks.dropna(subset=["scheduled"]), "scheduled")
    theirs = (
        mbta[(mbta["mode"] == "subway") & mbta["weekly"].isin(weeks)]
        .groupby(keys)[["num_predictions", "num_accurate_predictions"]]
        .sum()
    )
    table = pd.DataFrame(
        {
            "our_predictions": ours.size(),
            "ours": 100 * ours.mean(),
            "mbta": 100
            * theirs["num_accurate_predictions"]
            / theirs["num_predictions"],
            "schedule": 100 * schedule.groupby(keys)["accurate"].mean(),
            "mbta_predictions": theirs["num_predictions"],
        }
    ).dropna(subset=["ours", "mbta"])

    def overall(group):
        weights = group["mbta_predictions"]
        row = {
            column: (group[column] * weights).sum() / weights.sum()
            for column in ("ours", "mbta", "schedule")
        }
        row["our_predictions"] = group["our_predictions"].sum()
        row["mbta_predictions"] = weights.sum()
        return pd.Series(row)

    totals = table.groupby("route_id")[table.columns.tolist()].apply(overall)
    totals["bin"] = "all"
    table = pd.concat([table.reset_index(), totals.reset_index()])
    table["bin"] = pd.Categorical(
        table["bin"], [*(b[0] for b in MBTA_ACCURACY_BINS), "all"], ordered=True
    )
    table["ours_minus_mbta"] = table["ours"] - table["mbta"]
    columns = ["ours", "mbta", "ours_minus_mbta", "schedule", "our_predictions"]
    table = table.sort_values(["route_id", "bin"]).set_index(["route_id", "bin"])
    return table[columns].round(1)


def _service_dates_in(week: str) -> list:
    """The service dates (YYYYMMDD) in the MBTA week labeled with this Friday"""
    friday = pd.Timestamp(week)
    days = pd.date_range(friday - pd.Timedelta(days=7), periods=7)
    return [int(day.strftime("%Y%m%d")) for day in days]


def print_head_to_head(results: pd.DataFrame, feed_dir: str):
    # Imported here: mbta_predictions imports from this module
    from astrotrain.mbta_predictions import (
        archive_files,
        attach_mbta_predictions,
        head_to_head,
        load_mbta_predictions,
    )

    files = archive_files(Path(feed_dir), results["service_date"].unique())
    if not files:
        print(f"\nNo archived MBTA predictions in {feed_dir} for these days")
        return
    predictions = load_mbta_predictions(files)
    report = head_to_head(attach_mbta_predictions(results, predictions))
    print(
        f"\nHead-to-head with the MBTA's own predictions, made at the same moments "
        f"for the same trips and stops ({report['coverage']:.0%} of our targets "
        "had one). Mean absolute error in seconds:\n"
    )
    print(report["by_stops_ahead"].to_string())
    print()
    print(report["by_route"].to_string())
    print("\nPercent accurate by the MBTA's rules, on the same targets:\n")
    print(report["accuracy"].to_string())

    # In the MBTA's favour: its first prediction after the train left, when it
    # knows the stop is over too
    after = head_to_head(attach_mbta_predictions(results, predictions, after=True))
    overall = after["by_stops_ahead"].loc["all"]
    print(
        f"\nWith the MBTA's first prediction made after the train left instead "
        f"(in its favour; {after['coverage']:.0%} coverage): ours {overall['ours']}s, "
        f"MBTA {overall['mbta']}s ({overall['ours_vs_mbta']})\n"
    )
    print(after["by_route"].to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Report a saved model's error in seconds against the schedule"
    )
    parser.add_argument("files", nargs="+", help="LAMP parquet files to evaluate on")
    parser.add_argument(
        "--model",
        default="models/countdown_gbm",
        help="Saved model directory, LightGBM or LSTM",
    )
    parser.add_argument(
        "--mbta-predictions",
        help="Feed folder of the MBTA's archived TripUpdates (trip_updates/feed=... "
        "from us-rail-peformance-archiver), for a head-to-head with the MBTA's own "
        "predictions made at the same moments",
    )
    parser.add_argument(
        "--mbta-accuracy",
        help="MBTA prediction accuracy CSV, to compare against the MBTA's own "
        "predictions (prep_data.py downloads it)",
    )
    args = parser.parse_args()

    if CountdownClockGBM.is_saved_in(args.model):
        gbm = CountdownClockGBM.load(args.model)
        results = gbm.results(gbm.prep.load_data(sorted(args.files)))
    else:
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

    if args.mbta_predictions:
        print_head_to_head(results, args.mbta_predictions)

    if args.mbta_accuracy:
        comparison = mbta_comparison(results, pd.read_csv(args.mbta_accuracy))
        weeks = ", ".join(mbta_weeks(results["service_date"]))
        print(
            f"\nPercent of predictions accurate by the MBTA's rules, for MBTA "
            f"weeks {weeks}. Context only: the MBTA's figures cover harder moments "
            "than ours (see the schedule column); use --mbta-predictions for a "
            "fair comparison.\n"
        )
        print(comparison.to_string())


if __name__ == "__main__":
    main()
