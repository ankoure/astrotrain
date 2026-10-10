import numpy as np
import pandas as pd
import pytest

from astrotrain.train_lstm import CountdownClockLSTM, keras, masked_huber, masked_mae
from tests.lamp_data import lamp_frame, stations, with_timestamps


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
    """target_1 per group, leaving out the stops where it's unknown"""
    known = out.dropna(subset=["target_1"])
    return known.groupby(by)["target_1"].apply(list).to_dict()


def test_target_is_next_stops_travel_time_within_each_trip(lstm):
    df = lamp_frame(20250901, "Red", stations(0, 3), n_trips=2)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, 11.0, 21.0, 31.0]
    df = with_timestamps(df)
    df = df.sample(frac=1, random_state=0)  # preprocess_data should restore order

    out = lstm.preprocess_data(df)

    assert known_targets(out, "trip_id") == {
        "20250901-Red-0": [20.0, 30.0],
        "20250901-Red-1": [21.0, 31.0],
    }
    # Each trip's last stop has no next stop, rather than borrowing the
    # following trip's first travel time
    assert out.groupby("trip_id").tail(1)["target_1"].isna().all()


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

    out = lstm.preprocess_data(
        pd.concat([with_timestamps(first), with_timestamps(second)])
    )
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


def test_missing_values_filled_with_training_median(lstm):
    training = lamp_frame(20250901, "Red", stations(0, 4), n_trips=1)
    training["dwell_time_seconds"] = [10.0, 20.0, 30.0, 40.0]
    lstm.fit_fill_values(training)
    df = lamp_frame(20250902, "Red", stations(0, 4), n_trips=1)
    df["dwell_time_seconds"] = [100.0, np.nan, 300.0, 500.0]

    out = lstm.preprocess_data(df)

    # The training median (25), not this batch's (300)
    assert out["dwell_time_seconds"].tolist() == [100.0, 25.0, 300.0, 500.0]


def test_column_missing_from_a_whole_batch_is_still_filled(lstm):
    # Regression: filling with the batch median left these as NaN for a batch
    # with no branch headways at all (like a Blue Line-only batch), which made
    # every prediction for it NaN
    df = lamp_frame(20250901, "Blue", stations(0, 5), n_trips=2)
    df["headway_branch_seconds"] = np.nan
    df["scheduled_headway_branch"] = np.nan

    X, _ = lstm.create_sequences(lstm.preprocess_data(df))

    assert len(X) > 0
    assert not np.isnan(X).any()


def test_fit_fill_values_uses_zero_for_a_column_with_no_values(lstm):
    df = lamp_frame(20250901, "Blue", stations(0, 3), n_trips=1)
    df["headway_branch_seconds"] = np.nan

    lstm.fit_fill_values(df)

    assert lstm.fill_values["headway_branch_seconds"] == 0


def test_preprocess_before_fitting_fill_values_raises(parquet_files):
    lstm = CountdownClockLSTM()
    lstm.fit_encoders(parquet_files)

    with pytest.raises(RuntimeError, match="fit_fill_values"):
        lstm.preprocess_data(lamp_frame(20250901, "Red", stations(0, 3), n_trips=1))


def test_impossible_travel_times_are_treated_as_missing(lstm):
    df = lamp_frame(20250901, "Red", stations(0, 5), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 64486.0, 40.0, 50.0]

    out = lstm.preprocess_data(with_timestamps(df))

    # Filled from the schedule (120s) and flagged, and never used as a target
    assert out["travel_time_seconds"].tolist() == [10.0, 20.0, 120.0, 40.0, 50.0]
    assert out["travel_time_missing"].tolist() == [0, 0, 1, 0, 0]
    unknown = [False, True, False, False, True]
    assert out["target_1"].isna().tolist() == unknown


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


def test_upcoming_schedule_is_the_scheduled_legs_ahead(lstm3):
    df = lamp_frame(20250901, "Red", stations(0, 5), n_trips=1)
    df["scheduled_travel_time"] = [100.0, 110.0, 130.0, 140.0, 150.0]

    out = lstm3.preprocess_data(with_timestamps(df))

    legs = out[lstm3.upcoming_schedule_columns]
    assert legs.loc[0].tolist() == [110.0, 130.0, 140.0]
    # Past the end of the trip there's no leg
    assert legs.loc[3].tolist() == [150.0, 0.0, 0.0]
    assert legs.loc[4].tolist() == [0.0, 0.0, 0.0]
    assert out["stops_remaining"].tolist() == [4, 3, 2, 1, 0]


def test_missing_upcoming_schedule_uses_the_training_median(lstm3):
    df = lamp_frame(20250901, "Red", stations(0, 5), n_trips=1)
    df["scheduled_travel_time"] = [100.0, 110.0, np.nan, 140.0, 150.0]

    out = lstm3.preprocess_data(with_timestamps(df))

    # The synthetic training files all schedule 120s legs
    assert lstm3.fill_values["scheduled_travel_time"] == 120.0
    assert out.loc[0, lstm3.upcoming_schedule_columns].tolist() == [110.0, 120.0, 140.0]


def test_red_line_branches_get_their_own_codes(lstm, tmp_path):
    # Ashmont (Red-A) and Braintree (Red-B) trains share route_id "Red"
    files = []
    for branch in ["Red-A", "Red-B"]:
        df = lamp_frame(20250901, "Red", stations(0, 3), n_trips=1)
        df["branch_route_id"] = branch
        df["trip_id"] = branch
        files.append(tmp_path / f"{branch}.parquet")
        df.to_parquet(files[-1])
    orange = lamp_frame(20250901, "Orange", stations(0, 3), n_trips=1)
    files.append(tmp_path / "orange.parquet")
    orange.to_parquet(files[-1])
    lstm.fit_encoders(files)

    out = lstm.preprocess_data(
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    )

    codes = out.groupby("trip_id")["branch_encoded"].first()
    assert codes["Red-A"] != codes["Red-B"]
    # Lines without branches use their route
    assert out.loc[out["route_id"] == "Orange", "branch_encoded"].nunique() == 1
    assert set(lstm.branch_encoder.classes_) == {"Red-A", "Red-B", "Orange"}


# --- create_sequences ---


def test_sequence_window_and_target(lstm):
    df = lamp_frame(20250901, "Red", stations(0, 6), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    X, y = lstm.create_sequences(lstm.preprocess_data(with_timestamps(df)))

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


def test_return_index_points_at_each_windows_last_row(lstm):
    df = lstm.preprocess_data(lamp_frame(20250901, "Red", stations(0, 8), n_trips=2))

    X, y, end_rows = lstm.create_sequences(df, return_index=True)

    last_stops = X[:, -1, lstm.feature_columns.index("stop_sequence")]
    assert len(end_rows) == len(X) == len(y)
    assert df.loc[end_rows, "stop_sequence"].tolist() == last_stops.tolist()
    assert df.loc[end_rows, "target_1"].tolist() == y.ravel().tolist()


def test_windows_never_span_two_trips(lstm):
    df = lstm.preprocess_data(lamp_frame(20250901, "Red", stations(0, 8), n_trips=3))

    X, _ = lstm.create_sequences(df)

    stop_sequence = X[:, :, lstm.feature_columns.index("stop_sequence")]
    assert (np.diff(stop_sequence, axis=1) == 1).all()


def test_recording_gap_keeps_the_target_when_the_arrival_was_recorded(lstm):
    # A missing travel time usually means leaving the previous stop wasn't
    # recorded but arriving was, so the target is still known from timestamps
    df = lamp_frame(20250901, "Red", stations(0, 7), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0]
    df = with_timestamps(df)  # reaches stop 3 its scheduled 120s after stop 2

    X, y = lstm.create_sequences(lstm.preprocess_data(df))

    stop_sequence = X[:, :, lstm.feature_columns.index("stop_sequence")]
    assert stop_sequence.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert y.tolist() == [[120.0], [50.0], [60.0], [70.0]]


def test_window_with_no_known_target_is_skipped_and_others_stay_consecutive(lstm):
    # Regression: the stop before a missing value used to be dropped, so later
    # windows silently jumped over it
    df = lamp_frame(20250901, "Red", stations(0, 7), n_trips=1)
    df["travel_time_seconds"] = [10.0, 20.0, 30.0, np.nan, 50.0, 60.0, 70.0]
    df = with_timestamps(df)
    df.loc[3, "stop_timestamp"] = np.nan  # the arrival wasn't recorded either

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


def per_trip_loop(lstm, df):
    """The straightforward loop create_sequences replaced, as a reference"""
    X, y, end_rows = [], [], []
    for _, trip in df.groupby(lstm.TRIP_KEY):
        features = trip[lstm.feature_columns].to_numpy(dtype=float)
        targets = trip[lstm.target_columns].to_numpy(dtype=float)
        for i in range(len(trip) - lstm.sequence_length + 1):
            end = i + lstm.sequence_length - 1
            if not np.isnan(targets[end]).all():
                X.append(features[i : end + 1])
                y.append(targets[end])
                end_rows.append(trip.index[end])
    return np.array(X), np.array(y), np.array(end_rows)


@pytest.mark.parametrize("shuffle", [False, True], ids=["sorted", "shuffled"])
def test_create_sequences_matches_a_per_trip_loop(lstm3, shuffle):
    trips = [("Red", 8), ("Orange", 4), ("Blue", 2), ("Green-B", 6)]
    df = pd.concat(
        [
            lamp_frame(20250901 + day, route, stations(0, n), n_trips=3, seed=day)
            for day, (route, n) in enumerate(trips)
        ],
        ignore_index=True,
    )
    # Some arrivals weren't recorded, so some targets are unknown
    rng = np.random.default_rng(0)
    df.loc[rng.choice(len(df), 8, replace=False), "stop_timestamp"] = np.nan
    out = lstm3.preprocess_data(df)
    if shuffle:
        out = out.sample(frac=1, random_state=0)

    X, y, end_rows = lstm3.create_sequences(out, return_index=True)
    X_ref, y_ref, end_rows_ref = per_trip_loop(lstm3, out)

    assert len(X) > 0
    np.testing.assert_array_equal(X, X_ref)
    np.testing.assert_array_equal(y, y_ref)  # treats NaN as equal to NaN
    np.testing.assert_array_equal(end_rows, end_rows_ref)


def test_fewer_rows_than_the_window_gives_empty_arrays_of_the_right_shape(lstm):
    df = lstm.preprocess_data(lamp_frame(20250901, "Red", stations(0, 2), n_trips=1))

    X, y = lstm.create_sequences(df)

    assert X.shape == (0, 3, len(lstm.feature_columns))
    assert y.shape == (0, 1)


# --- predicting several stops ahead ---


def trip_with(lstm3, travel, dwell=None, scheduled=None):
    """One trip with the given times, preprocessed"""
    df = lamp_frame(20250901, "Red", stations(0, len(travel)), n_trips=1)
    df["travel_time_seconds"] = travel
    if dwell is not None:
        df["dwell_time_seconds"] = dwell
    if scheduled is not None:
        df["scheduled_travel_time"] = scheduled
    return lstm3.preprocess_data(with_timestamps(df))


def test_targets_are_time_from_leaving_to_arriving_k_stops_ahead(lstm3):
    out = trip_with(
        lstm3,
        travel=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        dwell=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    targets = out[lstm3.target_columns]
    # From leaving stop 0: travel to 1; + dwell at 1 + travel to 2; + dwell at 2
    # + travel to 3
    assert targets.loc[0].tolist() == [20.0, 20 + 2 + 30.0, 20 + 2 + 30 + 3 + 40.0]
    # Stops past the end of the trip are unknown
    assert targets.loc[3].tolist()[:2] == [50.0, 50 + 5 + 60.0]
    assert targets.loc[3].isna().tolist() == [False, False, True]
    assert targets.loc[5].isna().all()


def test_targets_through_a_recording_error_are_unknown(lstm3):
    out = trip_with(lstm3, travel=[10.0, 20.0, 30.0, 64486.0, 50.0, 60.0])

    targets = out[lstm3.target_columns]
    # Leaving stop 0, stops 1 and 2 are fine but the way to stop 3 isn't
    assert targets.loc[0].isna().tolist() == [False, False, True]
    # Leaving stop 2, everything passes through the bad leg
    assert targets.loc[2].isna().all()
    assert targets.loc[3, "target_1"] == 50.0


def test_scheduled_targets_sum_scheduled_legs_minus_the_dwell_done(lstm3):
    out = trip_with(
        lstm3,
        travel=[10.0, 20.0, 30.0, 40.0, 50.0],
        dwell=[1.0, 2.0, 3.0, 4.0, 5.0],
        scheduled=[100.0, 110.0, 120.0, 130.0, 140.0],
    )

    # Scheduled times run arrival to arrival, so leaving stop 0 after a 1s
    # dwell, the schedule expects stop 1 in 110 - 1 seconds
    assert out.loc[0, lstm3.scheduled_target_columns].tolist() == [
        110 - 1.0,
        110 + 120 - 1.0,
        110 + 120 + 130 - 1.0,
    ]


def test_windows_with_some_unknown_targets_are_kept(lstm3):
    out = trip_with(lstm3, travel=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    _, y = lstm3.create_sequences(out)

    # Windows end on stops 2-4; the trip's last stop has nothing ahead
    assert y.shape == (3, 3)
    assert np.isnan(y).tolist() == [
        [False, False, False],
        [False, False, True],
        [False, True, True],
    ]


def test_masked_loss_and_metric_ignore_unknown_targets():
    y_true = np.array([[1.0, np.nan], [2.0, 3.0]], dtype="float32")
    y_pred = np.array([[0.0, 5.0], [2.0, 1.0]], dtype="float32")

    loss = keras.ops.convert_to_numpy(masked_huber(y_true, y_pred))
    mae = keras.ops.convert_to_numpy(masked_mae(y_true, y_pred))

    # Huber: a miss of 1 is 0.5, a miss of 2 is 1.5 (linear past delta=1), and
    # the unknown target adds nothing; averaged over each row's 2 outputs
    np.testing.assert_allclose(loss, [0.25, 0.75])
    # MAE averages over known targets only: row 1 has one miss of 1
    np.testing.assert_allclose(mae, [1.0, 1.0])


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
        lstm = CountdownClockLSTM(sequence_length=3, prediction_horizon=3, seed=seed)
        lstm.fit_encoders(parquet_files)
        lstm.fit_fill_values(lstm.load_data(parquet_files))
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


@pytest.mark.filterwarnings("ignore:Your input ran out of data")
def test_training_with_unknown_far_targets_has_finite_loss(
    lstm3, parquet_files, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    history = lstm3.train(parquet_files, epochs=2, batch_size=8, file_batch_size=2)

    assert np.isfinite(history.history["loss"]).all()
    assert np.isfinite(history.history["val_loss"]).all()


def test_training_fails_clearly_when_no_trip_reaches_the_horizon(
    parquet_files, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    # The synthetic trips have 8 stops and the earliest window ends on stop 2,
    # so 5 is the furthest ahead any window reaches
    lstm = CountdownClockLSTM(sequence_length=3, prediction_horizon=6, seed=0)

    with pytest.raises(ValueError, match="reach 6 stops ahead"):
        lstm.train(parquet_files, epochs=1, batch_size=8, file_batch_size=2)


# --- save / load ---


# The model is saved untrained, so there's no optimizer state to restore
@pytest.mark.filterwarnings("ignore:Skipping variable loading for optimizer")
def test_save_and_load_round_trip(lstm3, parquet_files, tmp_path):
    X, _ = fit_scalers(lstm3, parquet_files)
    lstm3.build_model(input_shape=X.shape[1:])
    expected = lstm3.predict(X)

    lstm3.save(str(tmp_path / "model"))
    loaded = CountdownClockLSTM()
    loaded.load(str(tmp_path / "model"))

    assert expected.shape == (len(X), 3)
    np.testing.assert_allclose(loaded.predict(X), expected, rtol=1e-5)
    assert loaded.prediction_horizon == 3
    assert loaded.sequence_length == lstm3.sequence_length
    assert loaded.feature_columns == lstm3.feature_columns
    assert list(loaded.stop_encoder.classes_) == list(lstm3.stop_encoder.classes_)
    assert list(loaded.branch_encoder.classes_) == list(lstm3.branch_encoder.classes_)
    assert loaded.fill_values == lstm3.fill_values
