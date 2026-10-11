import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from astrotrain import tracking
from astrotrain.evaluate import evaluate
from astrotrain.gbm import CountdownClockGBM
from astrotrain.network import AHEAD_COLUMNS, LINE_AHEAD_COLUMNS
from tests.lamp_data import lamp_frame, stations, with_timestamps

KEY = ["trip_id", "departure_time", "stops_ahead"]


@pytest.fixture
def gbm(lstm3):
    """A GBM sharing lstm3's fitted preprocessing (windows of 3, 3 stops ahead)"""
    model = CountdownClockGBM(use_train_ahead=True)
    model.prep = lstm3
    return model


def one_trip(n_stops=6):
    df = lamp_frame(20250901, "Red", stations(0, n_stops), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0][:n_stops]
    return with_timestamps(df)


def test_table_covers_exactly_the_lstms_targets(gbm, built_lstm, parquet_files):
    raw = built_lstm.load_data(parquet_files)

    table = gbm.table(built_lstm.preprocess_data(raw))
    lstm_results = evaluate(built_lstm, raw)

    ours = table[KEY].sort_values(KEY).reset_index(drop=True)
    theirs = lstm_results[KEY].sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(ours, theirs)
    merged = table.merge(lstm_results, on=KEY, suffixes=("", "_lstm"))
    assert (merged["actual"] == merged["actual_lstm"]).all()
    assert (merged["stop_id"] == merged["stop_id_lstm"]).all()


def test_rows_targets_and_lags(gbm):
    out = gbm.prep.preprocess_data(one_trip())

    table = gbm.table(out).sort_values(["departure_time", "stops_ahead"])

    # Departures from stop 2 on (windows of 3), each with its known targets
    first = table[table["position"] == 2]
    assert first["stops_ahead"].tolist() == [1, 2, 3]
    assert first["actual"].tolist() == out.loc[2, gbm.prep.target_columns].tolist()
    assert first["stop_id"].tolist() == ["place-s3-01", "place-s4-01", "place-s5-01"]
    # Lags reach back within the trip only: from stop 2 there's no 3rd leg back
    row = first.iloc[0]
    assert row["travel_time_seconds_lag1"] == 20.0
    assert row["travel_time_seconds_lag2"] == 10.0
    assert np.isnan(row["travel_time_seconds_lag3"])
    # How much slower than typical the leg into stop 2 was
    assert row["leg_excess"] == pytest.approx(30.0 - out.loc[1, "typical_time_1"])


def test_train_ahead_columns_only_in_variant_b(lstm3):
    with_ahead = CountdownClockGBM(use_train_ahead=True).feature_columns
    without = CountdownClockGBM(use_train_ahead=False).feature_columns

    assert set(AHEAD_COLUMNS) <= set(with_ahead)
    assert not set(AHEAD_COLUMNS) & set(without)


def test_train_and_results_match_evaluate_format(parquet_files, built_lstm):
    def trained(seed):
        model = CountdownClockGBM(
            use_train_ahead=True,
            sequence_length=3,
            prediction_horizon=3,
            seed=seed,
            params={"num_leaves": 4, "min_data_in_leaf": 5},
        )
        model.train(parquet_files, num_boost_round=20, file_batch_size=2)
        return model

    model = trained(seed=0)
    raw = model.prep.load_data(parquet_files[-1:])
    results = model.results(raw)

    assert list(results.columns) == list(evaluate(built_lstm, raw).columns)
    assert results["predicted"].notna().all()
    np.testing.assert_array_equal(
        results["predicted"], trained(seed=0).results(raw)["predicted"]
    )


def train_small(parquet_files):
    model = CountdownClockGBM(
        use_train_ahead=True,
        sequence_length=3,
        prediction_horizon=3,
        params={"num_leaves": 4, "min_data_in_leaf": 5},
    )
    model.train(parquet_files, num_boost_round=20, file_batch_size=2)
    return model


@pytest.fixture
def trained_gbm(parquet_files):
    return train_small(parquet_files)


def test_save_and_load_round_trip(trained_gbm, parquet_files, tmp_path):
    raw = trained_gbm.prep.load_data(parquet_files[-1:])
    expected = trained_gbm.results(raw)

    trained_gbm.save(tmp_path / "model")
    loaded = CountdownClockGBM.load(tmp_path / "model")

    pd.testing.assert_frame_equal(loaded.results(raw), expected)
    assert loaded.use_train_ahead
    assert loaded.prep.prediction_horizon == 3
    assert CountdownClockGBM.is_saved_in(tmp_path / "model")


def test_evaluate_command_runs_a_saved_gbm(
    trained_gbm, parquet_files, tmp_path, monkeypatch, capsys
):
    import sys

    from astrotrain.evaluate import main

    trained_gbm.save(tmp_path / "model")
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--model", str(tmp_path / "model"), parquet_files[-1]]
    )

    main()

    # Same targets as the LSTM's evaluate test: 12 per trip, 4 trips
    assert "Compared 48 of 48 targets" in capsys.readouterr().out


def test_line_ahead_columns_only_in_variant_c():
    c = CountdownClockGBM(use_train_ahead=True, use_line_ahead=True).feature_columns
    b = CountdownClockGBM(use_train_ahead=True).feature_columns

    assert set(LINE_AHEAD_COLUMNS) <= set(c)
    assert not set(LINE_AHEAD_COLUMNS) & set(b)


def test_saved_metadata_says_how_the_model_was_made(trained_gbm, tmp_path):
    trained_gbm.save(tmp_path / "model")

    metadata = json.loads((tmp_path / "model" / "metadata.json").read_text())
    assert metadata["model"] == "lightgbm"
    assert metadata["training"]["days"] == 5
    assert metadata["training"]["first_day"] < metadata["training"]["last_day"]
    assert metadata["training"]["trees"] > 0
    assert metadata["training"]["max_trees"] == 20
    assert metadata["settings"]["use_train_ahead"] is True
    assert metadata["features"] == trained_gbm.feature_columns
    assert len(metadata["commit"]) == 40  # the tests run from a git checkout
    assert metadata["mlflow_run_id"] is None  # trained outside a run


def test_training_and_saving_log_to_the_active_run(parquet_files, tmp_path):
    with tracking.start_run("test", "training") as run:
        model = train_small(parquet_files)
        model.save(tmp_path / "model")

    logged = mlflow.get_run(run.info.run_id)
    assert logged.data.tags["kind"] == "training"
    assert logged.data.tags["uncommitted_changes"] in {"True", "False"}
    assert logged.data.params["num_leaves"] == "4"
    assert logged.data.params["training_days"] == "5"
    assert logged.data.params["training_trees"] == str(model.training["trees"])
    assert logged.data.metrics["best_val_l1"] == pytest.approx(
        model.booster.best_score["valid_0"]["l1"]
    )
    client = mlflow.MlflowClient()
    curve = client.get_metric_history(run.info.run_id, "val_l1")
    assert len(curve) == model.booster.current_iteration()  # every round
    files = {item.path for item in client.list_artifacts(run.info.run_id, "model")}
    assert files == {"model/gbm.txt", "model/preprocessing.pkl", "model/metadata.json"}
    metadata = json.loads((tmp_path / "model" / "metadata.json").read_text())
    assert metadata["mlflow_run_id"] == run.info.run_id


def test_evaluating_a_tracked_model_nests_under_its_training_run(
    parquet_files, tmp_path, monkeypatch, capsys
):
    import sys

    from astrotrain.evaluate import main

    with tracking.start_run("test", "training") as training:
        train_small(parquet_files).save(tmp_path / "model")
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--model", str(tmp_path / "model"), parquet_files[-1]]
    )

    main()

    evaluation = mlflow.last_active_run()
    tags, params = evaluation.data.tags, evaluation.data.params
    assert tags["kind"] == "evaluation"
    assert tags["mlflow.parentRunId"] == training.info.run_id
    assert evaluation.info.experiment_id == training.info.experiment_id
    day = Path(parquet_files[-1]).name[:10]
    assert evaluation.info.run_name == f"{day}..{day}"
    assert params["eval_days"] == "1"
    assert params["num_leaves"] == "4"  # the model's settings, as in training
    metrics = evaluation.data.metrics
    assert metrics["targets"] == 48
    assert {
        "model_mae",
        "schedule_mae",
        "model_mae_by_stops_ahead/1",
        "model_mae_by_stops_ahead/3",
        "model_mae_by_route/Green-B",
    } <= set(metrics)
    client = mlflow.MlflowClient()
    files = {item.path for item in client.list_artifacts(evaluation.info.run_id)}
    assert files == {"by_stops_ahead.csv", "by_route.csv"}
