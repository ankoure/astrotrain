import numpy as np
import pandas as pd
import pytest

from astrotrain.network import AHEAD_COLUMNS, LINE_AHEAD_COLUMNS
from tests.lamp_data import lamp_frame, stations, with_timestamps

START = 6 * 3600


def trip(trip_id, start_time, route="Red", trunk=None):
    """A 5-stop trip with 100s legs and 10s dwells"""
    df = lamp_frame(20250901, route, stations(0, 5), n_trips=1)
    df["trip_id"] = trip_id
    df["start_time"] = start_time
    df["trunk_route_id"] = trunk
    df["travel_time_seconds"] = [np.nan, 100.0, 100.0, 100.0, 100.0]
    df["dwell_time_seconds"] = 10.0
    return with_timestamps(df)


def ours(out):
    return out[out["trip_id"] == "ours"].set_index("stop_sequence")


def test_train_ahead_state_at_departure(lstm):
    # The train ahead leaves place-s0 at +10s and reaches place-s1 at +110s and
    # place-s2 at +220s. Ours arrives 150s after it and leaves place-s0 at +160s.
    df = pd.concat([trip("ahead", START), trip("ours", START + 150)])

    leaving_first_stop = ours(lstm.preprocess_data(df)).loc[0]

    assert leaving_first_stop["ahead_missing"] == 0
    assert leaving_first_stop["ahead_gap"] == 150.0  # it left 150s before us
    assert leaving_first_stop["ahead_stops_on"] == 1  # reached place-s1, not s2
    assert leaving_first_stop["ahead_since_seen"] == 50.0  # arrived 50s ago
    # It took 100s to reach place-s1, against this station's typical time
    typical = leaving_first_stop["typical_time_1"]
    assert leaving_first_stop["ahead_late"] == pytest.approx(100.0 - typical)


@pytest.mark.parametrize(
    "behind",
    [150, 205],
    # At 205s behind, ours leaves each stop 5s before the train ahead next
    # arrives somewhere, so peeking even slightly ahead would change the answer
    ids=["150s-behind", "205s-behind"],
)
def test_features_use_nothing_after_our_departure(lstm, behind):
    df = pd.concat([trip("ahead", START), trip("ours", START + behind)])
    out = ours(lstm.preprocess_data(df))

    for stop, departure in out["departure_time"].dropna().items():
        # Erase everything the train ahead did after we left this stop
        cut = df.copy()
        later = (cut["trip_id"] == "ahead") & (cut["stop_timestamp"] > departure)
        cut.loc[later, ["stop_timestamp", "move_timestamp"]] = np.nan
        moved_later = (cut["trip_id"] == "ahead") & (cut["move_timestamp"] > departure)
        cut.loc[moved_later, "move_timestamp"] = np.nan

        recomputed = ours(lstm.preprocess_data(cut)).loc[stop, AHEAD_COLUMNS]

        pd.testing.assert_series_equal(recomputed, out.loc[stop, AHEAD_COLUMNS])


def test_first_train_has_no_train_ahead(lstm):
    out = ours(lstm.preprocess_data(trip("ours", START)))

    assert (out["ahead_missing"] == 1).all()
    assert (out[AHEAD_COLUMNS[:-1]] == 0).all().all()


@pytest.mark.parametrize(
    ("trunks", "shared"),
    [(("Green", "Green"), True), ((None, None), False)],
    ids=["shared-trunk", "separate-lines"],
)
def test_train_ahead_can_be_on_another_branch_of_the_trunk(lstm, trunks, shared):
    lstm.route_encoder.fit([*lstm.route_encoder.classes_, "Green-C"])
    df = pd.concat(
        [
            trip("ahead", START, route="Green-B", trunk=trunks[0]),
            trip("ours", START + 150, route="Green-C", trunk=trunks[1]),
        ]
    )

    leaving_first_stop = ours(lstm.preprocess_data(df)).loc[0]

    assert leaving_first_stop["ahead_missing"] == (0 if shared else 1)
    assert leaving_first_stop["ahead_gap"] == (150.0 if shared else 0.0)


def test_a_train_that_left_long_ago_is_not_ahead(lstm):
    df = pd.concat([trip("ahead", START), trip("ours", START + 31 * 60)])

    leaving_first_stop = ours(lstm.preprocess_data(df)).loc[0]

    assert leaving_first_stop["ahead_missing"] == 1


# --- the next stations' state ---


def test_line_ahead_state_at_departure(lstm):
    # As before: the train ahead reaches place-s1 at +110s and leaves at +120s,
    # and reaches place-s2 at +220s; ours leaves place-s0 at +160s
    df = pd.concat([trip("ahead", START), trip("ours", START + 150)])

    leaving = ours(lstm.preprocess_data(df)).loc[0]

    # The leg into place-s1: finished by the train ahead 50s ago, taking 100s
    assert leaving["leg_excess_now_1"] == pytest.approx(
        100.0 - leaving["typical_time_1"]
    )
    assert leaving["leg_age_1"] == 50.0
    # Nobody at place-s1 now; the last train there stayed 10s
    assert leaving["occupied_for_1"] == 0.0
    assert leaving["last_dwell_1"] == 10.0
    # Nobody has finished the leg into place-s2 or left it yet
    assert np.isnan(leaving["leg_excess_now_2"])
    assert np.isnan(leaving["last_dwell_2"])


def test_a_train_at_the_next_stop_counts_as_occupying_it(lstm):
    # Ours leaves place-s0 at +115s: the train ahead got to place-s1 at +110s
    # and won't leave until +120s
    df = pd.concat([trip("ahead", START), trip("ours", START + 105)])

    leaving = ours(lstm.preprocess_data(df)).loc[0]

    assert leaving["occupied_for_1"] == 5.0


@pytest.mark.parametrize("behind", [105, 150, 205])
def test_line_ahead_uses_nothing_after_our_departure(lstm, behind):
    df = pd.concat([trip("ahead", START), trip("ours", START + behind)])
    out = ours(lstm.preprocess_data(df))

    for stop, departure in out["departure_time"].dropna().items():
        cut = df.copy()
        later = (cut["trip_id"] == "ahead") & (cut["stop_timestamp"] > departure)
        cut.loc[later, ["stop_timestamp", "move_timestamp"]] = np.nan
        moved_later = (cut["trip_id"] == "ahead") & (cut["move_timestamp"] > departure)
        cut.loc[moved_later, "move_timestamp"] = np.nan

        recomputed = ours(lstm.preprocess_data(cut)).loc[stop, LINE_AHEAD_COLUMNS]

        pd.testing.assert_series_equal(recomputed, out.loc[stop, LINE_AHEAD_COLUMNS])
