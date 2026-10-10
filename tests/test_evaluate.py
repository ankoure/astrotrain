import sys

import numpy as np
import pandas as pd
import pytest

from astrotrain.evaluate import evaluate, main, summarize
from tests.lamp_data import lamp_frame, stations


def test_evaluate_lines_up_actual_and_scheduled_times(built_lstm):
    df = lamp_frame(20250901, "Red", stations(0, 6), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    df["scheduled_travel_time"] = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    df["dwell_time_seconds"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    results = evaluate(built_lstm, df)

    # Windows end on stops 2-4; each target is the next stop's travel time.
    # Scheduled times run arrival to arrival, so the schedule's estimate for the
    # same segment is that stop's scheduled time minus the dwell before it.
    assert results["actual"].tolist() == [40.0, 50.0, 60.0]
    assert results["scheduled"].tolist() == [130.0 - 3, 140.0 - 4, 150.0 - 5]
    assert results["route_id"].tolist() == ["Red"] * 3
    assert results["predicted"].notna().all()


def test_summarize_compares_model_and_schedule_on_the_same_predictions():
    results = pd.DataFrame(
        {
            "route_id": ["Red", "Red", "Blue", "Blue"],
            "actual": [100.0, 100.0, 60.0, 60.0],
            "predicted": [110.0, 80.0, 65.0, 0.0],
            # The last prediction has no schedule to compare, so it's left out
            "scheduled": [130.0, 120.0, 50.0, np.nan],
        }
    )

    summary = summarize(results)

    assert summary.index.tolist() == ["all", "Blue", "Red"]
    assert summary["predictions"].tolist() == [3, 1, 2]
    # Model errors: Red 10, 20; Blue 5. Schedule errors: Red 30, 20; Blue 10
    assert summary["model_mae"].tolist() == [11.7, 5.0, 15.0]
    assert summary["schedule_mae"].tolist() == [20.0, 10.0, 25.0]
    assert summary.loc["Red", "model_p90"] == 19.0


# The model is saved untrained, so there's no optimizer state to restore
@pytest.mark.filterwarnings("ignore:Skipping variable loading for optimizer")
def test_main_prints_summary_for_a_saved_model(
    built_lstm, parquet_files, tmp_path, monkeypatch, capsys
):
    built_lstm.save(str(tmp_path / "model"))
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--model", str(tmp_path / "model"), parquet_files[-1]]
    )

    main()

    # 4 trips x 8 stops, windows of 3 -> 5 per trip, all with a scheduled time
    out = capsys.readouterr().out
    assert "Compared 20 of 20 predictions" in out
    assert "schedule_mae" in out
