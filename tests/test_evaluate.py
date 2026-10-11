import sys

import numpy as np
import pandas as pd
import pytest

from astrotrain.evaluate import (
    evaluate,
    main,
    mbta_comparison,
    mbta_weeks,
    score_like_mbta,
    summarize,
)
from tests.lamp_data import lamp_frame, stations, with_timestamps


def test_evaluate_gives_one_row_per_known_target(built_lstm):
    df = lamp_frame(20250901, "Red", stations(0, 6), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    df["dwell_time_seconds"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df["scheduled_travel_time"] = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]

    results = evaluate(built_lstm, with_timestamps(df))

    # Windows end on stops 2, 3 and 4, with 3, 2 and 1 stops left in the trip
    assert results["stops_ahead"].tolist() == [1, 2, 3, 1, 2, 1]
    assert results["actual"].tolist() == [
        40.0,
        40 + 4 + 50.0,
        40 + 4 + 50 + 5 + 60.0,
        50.0,
        50 + 5 + 60.0,
        60.0,
    ]
    # Scheduled arrival-to-arrival legs, minus the dwell at the stop being left
    assert results["scheduled"].tolist() == [
        130 - 3.0,
        130 + 140 - 3.0,
        130 + 140 + 150 - 3.0,
        140 - 4.0,
        140 + 150 - 4.0,
        150 - 5.0,
    ]
    assert results["route_id"].tolist() == ["Red"] * 6
    assert results["predicted"].notna().all()
    # Each target's stop, and the moment the prediction is made (leaving stop 2,
    # 3 or 4)
    assert results["stop_id"].tolist() == [
        "place-s3-01",
        "place-s4-01",
        "place-s5-01",
        "place-s4-01",
        "place-s5-01",
        "place-s5-01",
    ]
    out = built_lstm.preprocess_data(with_timestamps(df))
    leaving = out.set_index("stop_sequence")["departure_time"]
    assert results["departure_time"].tolist() == [leaving[2]] * 3 + [leaving[3]] * 2 + [
        leaving[4]
    ]


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "route_id": ["Red", "Red", "Blue", "Blue"],
            "stops_ahead": [1, 2, 1, 2],
            "actual": [100.0, 200.0, 60.0, 60.0],
            "predicted": [110.0, 180.0, 65.0, 0.0],
            # The last target has no schedule to compare, so it's left out
            "scheduled": [130.0, 220.0, 50.0, np.nan],
        }
    )


def test_summarize_by_route_compares_on_the_same_targets(results):
    summary = summarize(results, "route_id")

    assert summary.index.tolist() == ["all", "Blue", "Red"]
    assert summary["targets"].tolist() == [3, 1, 2]
    # Model errors: Red 10, 20; Blue 5. Schedule errors: Red 30, 20; Blue 10
    assert summary["model_mae"].tolist() == [11.7, 5.0, 15.0]
    assert summary["schedule_mae"].tolist() == [20.0, 10.0, 25.0]
    assert summary.loc["Red", "model_p90"] == 19.0
    assert summary.loc["Red", "actual_median"] == 150.0


def test_summarize_by_stops_ahead(results):
    summary = summarize(results, "stops_ahead")

    assert summary.index.tolist() == ["all", 1, 2]
    assert summary["targets"].tolist() == [3, 2, 1]
    assert summary.loc[1, "model_mae"] == 7.5  # errors 10 and 5
    assert summary.loc[2, "schedule_mae"] == 20.0


# The model is saved untrained, so there's no optimizer state to restore
@pytest.mark.filterwarnings("ignore:Skipping variable loading for optimizer")
def test_main_prints_summaries_for_a_saved_model(
    built_lstm, parquet_files, tmp_path, monkeypatch, capsys
):
    built_lstm.save(str(tmp_path / "model"))
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--model", str(tmp_path / "model"), parquet_files[-1]]
    )

    main()

    # 4 trips of 8 stops, windows of 3 ending on stops 2-6, which have 3, 3, 3,
    # 2 and 1 stops ahead: 12 targets a trip, all with a scheduled time
    out = capsys.readouterr().out
    assert "Compared 48 of 48 targets" in out
    assert "stops_ahead" in out
    assert "route_id" in out


# --- scoring like the MBTA ---


@pytest.mark.parametrize(
    ("predicted", "actual", "group", "accurate"),
    [
        (120, 180, "0-3 min", True),  # 60s late is the limit
        (120, 181, "0-3 min", False),
        (120, 60, "0-3 min", True),  # 60s early is the limit
        (120, 59, "0-3 min", False),
        (180, 300, "3-6 min", True),  # 3 minutes ahead starts the next group
        (300, 421, "3-6 min", False),  # 121s late
        (300, 210, "3-6 min", True),  # 90s early
        (600, 810, "6-12 min", True),  # 210s late
        (600, 449, "6-12 min", False),  # 151s early
        (1000, 1360, "12-30 min", True),  # 360s late
        (1000, 760, "12-30 min", True),  # 240s early
        (1000, 759, "12-30 min", False),
    ],
)
def test_score_like_mbta_uses_each_groups_window(predicted, actual, group, accurate):
    results = pd.DataFrame(
        {
            "service_date": [20260828],
            "route_id": ["Red"],
            "predicted": [float(predicted)],
            "actual": [float(actual)],
        }
    )

    scored = score_like_mbta(results, "predicted")

    assert scored["bin"].tolist() == [group]
    assert scored["accurate"].tolist() == [accurate]


def test_predictions_30_minutes_or_more_ahead_are_not_scored():
    results = pd.DataFrame(
        {
            "service_date": [20260828, 20260828],
            "route_id": ["Red", "Red"],
            "predicted": [1800.0, 2400.0],
            "actual": [1800.0, 2400.0],
        }
    )

    assert score_like_mbta(results, "predicted").empty


def days(start, n):
    first = pd.Timestamp(str(start))
    return [int((first + pd.Timedelta(days=i)).strftime("%Y%m%d")) for i in range(n)]


def test_mbta_weeks_are_complete_friday_to_thursday_weeks():
    # Friday 2026-08-21 to Thursday 2026-09-03, labeled the following Fridays
    two_weeks = days(20260821, 14)
    assert mbta_weeks(two_weeks) == ["2026-08-28", "2026-09-04"]

    missing_a_day = [d for d in two_weeks if d != 20260825]
    assert mbta_weeks(missing_a_day) == ["2026-09-04"]


def test_mbta_comparison_lines_up_our_accuracy_with_theirs():
    week = days(20260828, 7)  # the week labeled 2026-09-04
    rows = [
        # 0-3 min: 3 of 4 accurate for us, 2 of 4 for the schedule. Each is
        # grouped by its own prediction, so the schedule's misses are kept under
        # 3 minutes ahead too.
        (week[0], 120.0, 130.0, 130.0),
        (week[1], 120.0, 130.0, 10.0),
        (week[2], 120.0, 130.0, 130.0),
        (week[3], 120.0, 300.0, 10.0),
        # 12-30 min: 1 of 2 accurate
        (week[4], 1000.0, 1000.0, 1000.0),
        (week[5], 1000.0, 2000.0, 2000.0),
        # Only there so every day of the week is covered; over 30 min, not scored
        (week[6], 3000.0, 3000.0, 3000.0),
        # A day outside the week, which shouldn't count
        (20260827, 120.0, 900.0, 900.0),
    ]
    results = pd.DataFrame(
        rows, columns=["service_date", "predicted", "actual", "scheduled"]
    ).assign(route_id="Red")
    mbta = pd.DataFrame(
        {
            "weekly": ["2026-09-04", "2026-09-04", "2026-08-28", "2026-09-04"],
            "mode": ["subway", "subway", "subway", "bus"],
            "route_id": ["Red", "Red", "Red", np.nan],
            "bin": ["0-3 min", "12-30 min", "0-3 min", "0-3 min"],
            "arrival_departure": ["blended"] * 3 + ["departure"],
            "num_predictions": [100, 100, 100, 100],
            "num_accurate_predictions": [80, 70, 10, 10],
        }
    )

    table = mbta_comparison(results, mbta)

    assert table.loc[("Red", "0-3 min"), ["ours", "mbta", "schedule"]].tolist() == [
        75.0,
        80.0,
        50.0,
    ]
    assert table.loc[("Red", "12-30 min"), ["ours", "mbta"]].tolist() == [50.0, 70.0]
    # "all" weights each group by the MBTA's count of predictions in it
    assert table.loc[("Red", "all"), ["ours", "mbta"]].tolist() == [62.5, 75.0]
    assert table.loc[("Red", "all"), "ours_minus_mbta"] == -12.5
    assert table.loc[("Red", "all"), "our_predictions"] == 6


def test_mbta_comparison_needs_a_complete_week():
    results = pd.DataFrame(
        {
            "service_date": days(20260828, 3),
            "route_id": "Red",
            "predicted": 120.0,
            "actual": 120.0,
            "scheduled": 120.0,
        }
    )

    with pytest.raises(ValueError, match="No complete MBTA week"):
        mbta_comparison(results, pd.DataFrame())
