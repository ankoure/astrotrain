import sys

import numpy as np
import pandas as pd
import pytest

from astrotrain.evaluate import evaluate, main, summarize
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
