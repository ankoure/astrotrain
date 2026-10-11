import numpy as np
import pandas as pd
import pytest

from astrotrain.mbta_predictions import (
    MAX_PREDICTION_AGE_SECONDS,
    archive_files,
    attach_mbta_predictions,
    head_to_head,
    load_mbta_predictions,
)

T = 1_781_300_000.0  # when our train leaves its stop


def our_targets(**columns):
    base = {
        "service_date": 20260613,
        "trip_id": "t1",
        "route_id": "Red",
        "stop_id": "70061",
        "stops_ahead": 1,
        "departure_time": T,
        "actual": 100.0,
        "predicted": 100.0,
    }
    return pd.DataFrame([{**base, **row} for row in columns.get("rows", [{}])])


def predictions(*rows):
    return pd.DataFrame(
        rows, columns=["made", "trip_id", "stop_id", "predicted_at"]
    ).sort_values("made")


def test_uses_the_latest_prediction_made_at_or_before_our_departure():
    preds = predictions(
        (T - 60, "t1", "70061", T + 150),
        (T - 30, "t1", "70061", T + 120),  # the latest before we leave
        (T + 10, "t1", "70061", T + 999),  # made after: must not be used
        (T - 5, "t1", "other-stop", T + 5),  # another stop
        (T - 5, "t2", "70061", T + 5),  # another trip
    )

    out = attach_mbta_predictions(our_targets(), preds)

    assert out["mbta"].tolist() == [120.0]


def test_a_prediction_older_than_the_limit_is_not_used():
    stale = T - MAX_PREDICTION_AGE_SECONDS - 1
    out = attach_mbta_predictions(our_targets(), predictions((stale, "t1", "70061", T)))

    assert np.isnan(out["mbta"].iloc[0])


def test_rows_keep_their_order_and_targets_without_a_prediction_stay_missing():
    targets = our_targets(
        rows=[
            {"departure_time": T + 500, "stop_id": "70063"},
            {"departure_time": T},
            {"departure_time": T, "trip_id": "never-predicted"},
        ]
    )
    preds = predictions(
        (T - 10, "t1", "70061", T + 90),
        (T + 490, "t1", "70063", T + 600),
    )

    out = attach_mbta_predictions(targets, preds)

    assert out["stop_id"].tolist() == ["70063", "70061", "70061"]
    assert out["mbta"].tolist()[:2] == [100.0, 90.0]
    assert np.isnan(out["mbta"].iloc[2])


def test_load_keeps_subway_and_falls_back_to_departure_times(tmp_path):
    archived = pd.DataFrame(
        {
            "feed_timestamp": [3, 1, 2, 4],
            "trip_update.trip.trip_id": ["a", "b", "c", "d"],
            "trip_update.trip.route_id": ["Red", "Orange", "Red", "1"],  # 1 is a bus
            "trip_update.stop_time_update.stop_id": ["s1", "s2", "s3", "s4"],
            "trip_update.stop_time_update.arrival.time": [103.0, np.nan, np.nan, 104.0],
            "trip_update.stop_time_update.departure.time": [
                np.nan,
                202.0,
                np.nan,
                204.0,
            ],
        }
    )
    archived.to_parquet(tmp_path / "data.parquet")

    loaded = load_mbta_predictions([tmp_path / "data.parquet"])

    # Sorted by when made; "c" has neither time and the bus row is dropped
    assert loaded["trip_id"].tolist() == ["b", "a"]
    assert loaded["predicted_at"].tolist() == [202.0, 103.0]


def test_archive_files_include_the_next_utc_day(tmp_path):
    for day in (12, 13, 14, 15):
        folder = tmp_path / f"year=2026/month=6/day={day}"
        folder.mkdir(parents=True)
        (folder / "data.parquet").touch()

    files = archive_files(tmp_path, [20260613])

    assert [f.parent.name for f in files] == ["day=13", "day=14"]


def test_head_to_head_compares_on_targets_the_mbta_predicted():
    results = pd.DataFrame(
        {
            "route_id": ["Red", "Red", "Red"],
            "service_date": 20260613,
            "stops_ahead": [1, 1, 2],
            "actual": [100.0, 100.0, 400.0],
            "predicted": [110.0, 90.0, 300.0],
            "mbta": [140.0, 100.0, np.nan],  # no MBTA prediction for the last
        }
    )

    report = head_to_head(results)

    assert report["coverage"] == pytest.approx(2 / 3)
    by_stops = report["by_stops_ahead"]
    # Ours missed by 10 and 10; the MBTA by 40 and 0
    assert by_stops.loc["all", ["targets", "ours", "mbta"]].tolist() == [2, 10.0, 20.0]
    assert by_stops.loc["all", "ours_vs_mbta"] == "-50%"
    accuracy = report["accuracy"].loc[("Red", "0-3 min")]
    assert accuracy[["ours", "mbta"]].tolist() == [100.0, 100.0]
