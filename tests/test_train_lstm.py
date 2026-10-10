import numpy as np
import pandas as pd
import pytest

from astrotrain.train_lstm import CountdownClockLSTM, keras
from tests.lamp_data import lamp_frame, stations


def fit_scalers(lstm, parquet_files):
    """Fit the feature/target scalers the way train() does, returning the data"""
    X, y = lstm.create_sequences(lstm.preprocess_data(lstm.load_data(parquet_files)))
    lstm.feature_scaler.fit(X.reshape(-1, X.shape[-1]))
    lstm.target_scaler.fit(y)
    return X, y


# --- fit_encoders / preprocess_data ---


def test_fit_encoders_includes_values_only_in_later_files(lstm, parquet_files):
    first_day = pd.read_parquet(parquet_files[0])
    last_day = pd.read_parquet(parquet_files[-1])

    assert "Green-B" not in set(first_day["route_id"])
    assert "Green-B" in lstm.route_encoder.classes_
    assert set(last_day["parent_station"]) <= set(lstm.stop_encoder.classes_)


def test_station_codes_match_across_file_batches(lstm, parquet_files):
    # Regression: encoders used to be refit on each file batch, so the same
    # station got a different code depending on which batch it was in
    first = lstm.preprocess_data(lstm.load_data(parquet_files[:2]))
    second = lstm.preprocess_data(lstm.load_data(parquet_files[2:4]))

    first_codes = dict(zip(first["parent_station"], first["stop_encoded"]))
    second_codes = dict(zip(second["parent_station"], second["stop_encoded"]))
    shared = first_codes.keys() & second_codes.keys()

    assert shared
    assert all(first_codes[s] == second_codes[s] for s in shared)


def known_targets(out, by):
    """Targets per group, leaving out the stops whose next travel time is unknown"""
    known = out.dropna(subset=["target_travel_time"])
    return known.groupby(by)["target_travel_time"].apply(list).to_dict()


def test_target_is_next_stops_travel_time_within_each_trip(lstm):
    df = lamp_frame(20250901, "Red", stations(0, 3), n_trips=2)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, 11.0, 21.0, 31.0]
    df = df.sample(frac=1, random_state=0)  # preprocess_data should restore order

    out = lstm.preprocess_data(df)

    assert known_targets(out, "trip_id") == {
        "20250901-Red-0": [20.0, 30.0],
        "20250901-Red-1": [21.0, 31.0],
    }
    # Each trip's last stop has no next stop, rather than borrowing the
    # following trip's first travel time
    assert out.groupby("trip_id").tail(1)["target_travel_time"].isna().all()


@pytest.mark.parametrize(
    ("column", "values"),
    [("service_date", [20250901, 20250902]), ("route_id", ["Red", "Green-B"])],
    ids=["different-days", "different-routes"],
)
def test_trips_sharing_an_id_stay_separate(lstm, column, values):
    # Regression: real trip IDs repeat across days with the same timetable, and
    # some Green Line trips share an ID across branches. Grouping by trip_id
    # alone interleaved their stops.
    first = lamp_frame(20250901, "Red", stations(0, 5), n_trips=1)
    second = first.copy()
    first[column], second[column] = values
    first["travel_time_seconds"] = [10.0, 20.0, 30.0, 40.0, 50.0]
    second["travel_time_seconds"] = [11.0, 21.0, 31.0, 41.0, 51.0]

    out = lstm.preprocess_data(pd.concat([first, second]))
    X, _ = lstm.create_sequences(out)

    assert known_targets(out, column) == {
        values[0]: [20.0, 30.0, 40.0, 50.0],
        values[1]: [21.0, 31.0, 41.0, 51.0],
    }
    # The first trip's times end in 0 and the second's in 1, so a window
    # drawn from a single trip has only one remainder
    travel_time = X[:, :, lstm.feature_columns.index("travel_time_seconds")]
    assert len(X) > 0
    assert all(len(set(window % 10)) == 1 for window in travel_time)


def test_missing_values_filled_with_median(lstm):
    df = lamp_frame(20250901, "Red", stations(0, 4), n_trips=1)
    df["dwell_time_seconds"] = [10.0, np.nan, 30.0, 50.0]

    out = lstm.preprocess_data(df)

    assert out["dwell_time_seconds"].tolist() == [10.0, 30.0, 30.0, 50.0]


def test_impossible_travel_times_are_treated_as_missing(lstm):
    df = lamp_frame(20250901, "Red", stations(0, 5), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 64486.0, 40.0, 50.0]

    out = lstm.preprocess_data(df)

    # Filled from the schedule (120s) and flagged, and never used as a target
    assert out["travel_time_seconds"].tolist() == [10.0, 20.0, 120.0, 40.0, 50.0]
    assert out["travel_time_missing"].tolist() == [0, 0, 1, 0, 0]
    unknown = [False, True, False, False, True]
    assert out["target_travel_time"].isna().tolist() == unknown


@pytest.mark.parametrize(
    ("scheduled", "filled"), [(90.0, 90.0), (np.nan, 0.0)], ids=["scheduled", "none"]
)
def test_missing_first_stop_travel_time_uses_schedule_or_zero(lstm, scheduled, filled):
    df = lamp_frame(20250901, "Red", stations(0, 3), n_trips=1)
    df["travel_time_seconds"] = [np.nan, 20.0, 30.0]
    df.loc[0, "scheduled_travel_time"] = scheduled

    out = lstm.preprocess_data(df)

    assert out["travel_time_seconds"].tolist() == [filled, 20.0, 30.0]
    assert out["travel_time_missing"].tolist() == [1, 0, 0]


@pytest.mark.parametrize(
    ("start_time", "hour"),
    [(7 * 3600 + 15 * 60, 7), (25 * 3600 + 30 * 60, 1)],
    ids=["morning", "past-midnight"],  # service-day times can run past 24:00
)
def test_hour_of_day(lstm, start_time, hour):
    df = lamp_frame(20250901, "Red", stations(0, 2), n_trips=1)
    df["start_time"] = start_time

    out = lstm.preprocess_data(df)

    assert set(out["hour_of_day"]) == {hour}


def test_day_of_week(lstm):
    df = lamp_frame(20250906, "Red", stations(0, 2), n_trips=1)

    out = lstm.preprocess_data(df)

    assert set(out["day_of_week"]) == {5}  # 2025-09-06 is a Saturday


# --- create_sequences ---


def test_sequence_window_and_target(lstm):
    df = lamp_frame(20250901, "Red", stations(0, 6), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    X, y = lstm.create_sequences(lstm.preprocess_data(df))

    # Each window is 3 consecutive stops; the target is the travel time to the
    # stop right after the window
    travel_time = X[:, :, lstm.feature_columns.index("travel_time_seconds")]
    assert travel_time.tolist() == [
        [10.0, 20.0, 30.0],
        [20.0, 30.0, 40.0],
        [30.0, 40.0, 50.0],
    ]
    assert y.tolist() == [[40.0], [50.0], [60.0]]


def test_sequence_shapes(lstm):
    df = lstm.preprocess_data(lamp_frame(20250901, "Red", stations(0, 8), n_trips=2))

    X, y = lstm.create_sequences(df)

    # 8 stops -> 7 with a known next travel time -> 7 - 3 + 1 = 5 windows per trip
    assert X.shape == (10, 3, len(lstm.feature_columns))
    assert y.shape == (10, 1)


@pytest.mark.parametrize(("n_stops", "n_windows"), [(3, 0), (4, 1), (5, 2)])
def test_window_count_per_trip(lstm, n_stops, n_windows):
    # Regression: a trip's last usable window used to be skipped, so a trip with
    # exactly sequence_length (3) stops that have a known next travel time gave none
    df = lamp_frame(20250901, "Red", stations(0, n_stops), n_trips=1)

    X, y = lstm.create_sequences(lstm.preprocess_data(df))

    assert len(X) == len(y) == n_windows


def test_windows_never_span_two_trips(lstm):
    df = lstm.preprocess_data(lamp_frame(20250901, "Red", stations(0, 8), n_trips=3))

    X, _ = lstm.create_sequences(df)

    stop_sequence = X[:, :, lstm.feature_columns.index("stop_sequence")]
    assert (np.diff(stop_sequence, axis=1) == 1).all()


def test_gap_mid_trip_skips_only_the_window_with_no_target(lstm):
    # Regression: the stop before a missing travel time used to be dropped, so
    # later windows silently jumped over it
    df = lamp_frame(20250901, "Red", stations(0, 7), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0]

    X, y = lstm.create_sequences(lstm.preprocess_data(df))

    # The window ending on stop 2 has no target; the rest cover consecutive
    # stops, with the gap filled from the schedule (120s)
    stop_sequence = X[:, :, lstm.feature_columns.index("stop_sequence")]
    travel_time = X[:, :, lstm.feature_columns.index("travel_time_seconds")]
    assert stop_sequence.tolist() == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert travel_time.tolist() == [
        [20.0, 30.0, 120.0],
        [30.0, 120.0, 50.0],
        [120.0, 50.0, 60.0],
    ]
    assert y.tolist() == [[50.0], [60.0], [70.0]]


# --- sequence_generator / train ---


def test_sequence_generator_yields_float32_batches_covering_all_data(
    lstm, parquet_files
):
    X, _ = fit_scalers(lstm, parquet_files)

    batches = list(
        lstm.sequence_generator(parquet_files, file_batch_size=2, mini_batch_size=8)
    )

    assert all(xb.dtype == yb.dtype == np.float32 for xb, yb in batches)
    assert all(len(xb) <= 8 for xb, _ in batches)
    assert sum(len(xb) for xb, _ in batches) == len(X)


def windows(batches):
    """Every (sequence, target) pair from the batches in order, as hashable bytes"""
    return [(x.tobytes(), y.tobytes()) for xb, yb in batches for x, y in zip(xb, yb)]


def test_shuffle_changes_order_each_pass_but_not_contents(lstm, parquet_files):
    fit_scalers(lstm, parquet_files)

    def one_pass(shuffle):
        return windows(lstm.sequence_generator(parquet_files, 2, 8, shuffle=shuffle))

    in_order = one_pass(shuffle=False)
    first, second = one_pass(shuffle=True), one_pass(shuffle=True)

    assert first != in_order
    assert first != second
    assert sorted(first) == sorted(second) == sorted(in_order)


def test_same_seed_gives_same_shuffle(parquet_files):
    def first_pass(seed):
        lstm = CountdownClockLSTM(sequence_length=3, seed=seed)
        lstm.fit_encoders(parquet_files)
        fit_scalers(lstm, parquet_files)
        return windows(lstm.sequence_generator(parquet_files, 2, 8, shuffle=True))

    assert first_pass(seed=0) == first_pass(seed=0)


# Keras warns once per pass while it learns the dataset's length; it's harmless
@pytest.mark.filterwarnings("ignore:Your input ran out of data")
def test_every_epoch_trains_on_all_files(lstm, parquet_files, tmp_path, monkeypatch):
    # Regression: epochs after the first used to train on almost no data because
    # the generator was exhausted after one pass
    monkeypatch.chdir(tmp_path)  # train() writes its checkpoint to the working dir
    batches_per_epoch = []

    class CountBatches(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            batches_per_epoch.append(0)

        def on_train_batch_end(self, batch, logs=None):
            batches_per_epoch[-1] += 1

    fit = keras.Model.fit

    def fit_with_counter(self, *args, **kwargs):
        kwargs["callbacks"] = [*kwargs["callbacks"], CountBatches()]
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(keras.Model, "fit", fit_with_counter)

    lstm.train(parquet_files, epochs=3, batch_size=8, file_batch_size=2)

    # train() holds out the last 20% for validation. Every file has the same number
    # of sequences, so the batch count doesn't depend on how files get shuffled.
    train_files = parquet_files[:4]
    expected = sum(1 for _ in lstm.sequence_generator(train_files, 2, 8))
    assert batches_per_epoch == [expected] * 3


# --- save / load ---


# The model is saved untrained, so there's no optimizer state to restore
@pytest.mark.filterwarnings("ignore:Skipping variable loading for optimizer")
def test_save_and_load_round_trip(lstm, parquet_files, tmp_path):
    X, _ = fit_scalers(lstm, parquet_files)
    lstm.build_model(input_shape=X.shape[1:])
    expected = lstm.predict(X)

    lstm.save(str(tmp_path / "model"))
    loaded = CountdownClockLSTM()
    loaded.load(str(tmp_path / "model"))

    np.testing.assert_allclose(loaded.predict(X), expected, rtol=1e-5)
    assert loaded.sequence_length == lstm.sequence_length
    assert loaded.feature_columns == lstm.feature_columns
    assert list(loaded.stop_encoder.classes_) == list(lstm.stop_encoder.classes_)
