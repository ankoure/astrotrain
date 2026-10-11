"""
What's known about the rest of the line when a train leaves a stop.

Delays pass backward along a line: a train held up ahead holds up the trains
behind it. On the test week, when the train ahead was over 3 minutes late, the
train behind was 3-4x as likely to be over 2 minutes late a few stops on.

For each departure this adds the state of the train ahead: the previous train to
leave the same station, in the same direction, on the same trunk (so a Green-C
train's train ahead can be a Green-B train, and Ashmont and Braintree trains
share the Red Line trunk). add_line_ahead adds the state of the next few stations
and legs of the trip, from every train on the trunk. Only events that had
already happened when our train left are used.
"""

from typing import List

import numpy as np
import pandas as pd

# A train that left this station longer ago than this isn't meaningfully ahead
# (on real data the gap reached 17 hours at rarely served stations)
MAX_AHEAD_GAP_SECONDS = 30 * 60

AHEAD_COLUMNS = [
    "ahead_gap",
    "ahead_stops_on",
    "ahead_late",
    "ahead_since_seen",
    "ahead_missing",
]


def add_train_ahead(
    df: pd.DataFrame, trip_key: List[str], typical_columns: List[str]
) -> None:
    """
    Add the train ahead's state at each row's departure:

      ahead_gap: seconds since the train ahead left this station
      ahead_stops_on: how many stops beyond this station it had reached
      ahead_late: how much longer than typical it took to reach that stop,
                  using this station's typical times (0 before it reaches the
                  next stop, or beyond the typical times' reach)
      ahead_since_seen: seconds since it last arrived at a stop; large when
                        it's held at a station or stuck between two
      ahead_missing: 1 when there's no train ahead (e.g. the day's first train,
                     or the last one left over MAX_AHEAD_GAP_SECONDS ago)

    Unknown values are 0. df needs a RangeIndex and must be sorted by trip and
    stop, with departure_time and the typical time columns already added.
    """
    trips = df.groupby(trip_key)
    trip = trips.ngroup().to_numpy()
    position = trips.cumcount().to_numpy()

    departures = pd.DataFrame(
        {
            "row": np.arange(len(df)),
            "trip": trip,
            "position": position,
            "service_date": df["service_date"].to_numpy(),
            "trunk": df["trunk_route_id"].fillna(df["route_id"]).to_numpy(),
            "direction_id": df["direction_id"].to_numpy(),
            "station": df["parent_station"].to_numpy(),
            "departure": df["departure_time"].to_numpy(),
        }
    ).dropna(subset=["departure"])
    departures = departures.sort_values("departure", kind="stable")

    # The train ahead is the previous one to leave the same station
    same_place = departures.groupby(
        ["service_date", "trunk", "direction_id", "station"]
    )
    departures["ahead_trip"] = same_place["trip"].shift(1)
    departures["ahead_left"] = same_place["departure"].shift(1)
    departures["ahead_position"] = same_place["position"].shift(1)
    recent = departures["departure"] - departures["ahead_left"] <= MAX_AHEAD_GAP_SECONDS
    departures = departures[recent]
    departures["ahead_trip"] = departures["ahead_trip"].astype(trip.dtype)

    # Its latest arrival at or before our departure, so nothing from the future
    arrivals = pd.DataFrame(
        {
            "ahead_trip": trip,
            "seen_at": df["stop_timestamp"].to_numpy(),
            "seen_position": position,
        }
    ).dropna(subset=["seen_at"])
    state = pd.merge_asof(
        departures,
        arrivals.sort_values("seen_at", kind="stable"),
        left_on="departure",
        right_on="seen_at",
        by="ahead_trip",
        direction="backward",
    ).dropna(subset=["seen_at"])

    stops_on = np.maximum(state["seen_position"] - state["ahead_position"], 0)
    stops_on = stops_on.to_numpy().astype(int)

    # How late it was reaching that stop, against this station's typical time
    rows = state["row"].to_numpy()
    typical = df[typical_columns].to_numpy()[rows]
    reach = (stops_on >= 1) & (stops_on <= len(typical_columns))
    typical_to_seen = np.full(len(state), np.nan)
    typical_to_seen[reach] = typical[np.flatnonzero(reach), stops_on[reach] - 1]
    took = (state["seen_at"] - state["ahead_left"]).to_numpy()

    values = {
        "ahead_gap": (state["departure"] - state["ahead_left"]).to_numpy(),
        "ahead_stops_on": stops_on,
        "ahead_late": np.nan_to_num(took - typical_to_seen),
        # Capped: when the ahead train's arrivals weren't recorded, it can last
        # have been seen hours ago
        "ahead_since_seen": np.minimum(
            (state["departure"] - state["seen_at"]).to_numpy(), MAX_AHEAD_GAP_SECONDS
        ),
        "ahead_missing": np.zeros(len(state)),
    }
    for column in AHEAD_COLUMNS:
        filled = np.zeros(len(df)) if column != "ahead_missing" else np.ones(len(df))
        filled[rows] = values[column]
        df[column] = filled


# How many stations ahead add_line_ahead describes
LINE_AHEAD_STOPS = 3

LINE_AHEAD_COLUMNS = [
    f"{name}_{j}"
    for j in range(1, LINE_AHEAD_STOPS + 1)
    for name in ("leg_excess_now", "leg_age", "occupied_for", "last_dwell")
]


def add_line_ahead(
    df: pd.DataFrame, trip_key: List[str], max_leg_seconds: float
) -> None:
    """
    Add the state of the next LINE_AHEAD_STOPS stops of each row's trip at its
    departure, from all trains on the same trunk in the same direction:

      leg_excess_now_j: how much slower than typical the most recent train to
                        finish the leg into the j-th stop ahead was (a slow zone
                        or an incident shows up here first)
      leg_age_j: seconds since that train got there; an old reading means little
      occupied_for_j: seconds a train has been at the j-th stop ahead, if one is
                      there now (arrived, not yet left), else 0
      last_dwell_j: seconds the most recent train to leave that stop spent there

    Unknown values are left missing (NaN). df needs a RangeIndex and must be
    sorted by trip and stop, with departure_time and typical_time_1 added.
    """
    trips = df.groupby(trip_key)
    trunk = df["trunk_route_id"].fillna(df["route_id"])
    line = {
        "service_date": df["service_date"],
        "trunk": trunk,
        "direction_id": df["direction_id"],
    }
    travel = df["travel_time_seconds"].where(
        df["travel_time_seconds"] <= max_leg_seconds
    )
    # Every stop visit, as the events other trains will look up
    visits = pd.DataFrame(
        {
            **line,
            "from_station": trips["parent_station"].shift(1),
            "station": df["parent_station"],
            "arrived": df["stop_timestamp"],
            "left": df["departure_time"],
            # Slower than typical on the leg into this stop
            "leg_excess": travel - trips["typical_time_1"].shift(1),
        }
    )
    visits["dwell"] = visits["left"] - visits["arrived"]
    keys = ["service_date", "trunk", "direction_id"]

    # Each row's moment of departure and the stations ahead of it
    queries = pd.DataFrame({**line, "at": df["departure_time"]})
    queries["row"] = np.arange(len(df))
    for j in range(1, LINE_AHEAD_STOPS + 1):
        queries[f"from_{j}"] = trips["parent_station"].shift(-(j - 1))
        queries[f"to_{j}"] = trips["parent_station"].shift(-j)
    queries = queries.dropna(subset=["at"]).sort_values("at", kind="stable")

    def latest(events, time, query_station_columns, event_station_columns):
        """For each query, the latest event at or before its departure"""
        events = events.dropna(subset=[time, *event_station_columns])
        renamed = queries.rename(
            columns=dict(zip(query_station_columns, event_station_columns))
        )
        renamed = renamed.dropna(subset=event_station_columns)
        return pd.merge_asof(
            renamed,
            events.sort_values(time, kind="stable"),
            left_on="at",
            right_on=time,
            by=keys + event_station_columns,
            direction="backward",
        )

    for j in range(1, LINE_AHEAD_STOPS + 1):
        columns = {
            f"{name}_{j}": np.full(len(df), np.nan)
            for name in ("leg_excess_now", "leg_age", "occupied_for", "last_dwell")
        }

        legs = latest(
            visits[keys + ["from_station", "station", "arrived", "leg_excess"]],
            "arrived",
            [f"from_{j}", f"to_{j}"],
            ["from_station", "station"],
        )
        rows = legs["row"].to_numpy()
        columns[f"leg_excess_now_{j}"][rows] = legs["leg_excess"].to_numpy()
        columns[f"leg_age_{j}"][rows] = (legs["at"] - legs["arrived"]).to_numpy()

        # The latest train to arrive there; it's still there if it hasn't left
        # by now (unrecorded departures count as still there)
        arrivals = latest(
            visits[keys + ["station", "arrived", "left"]],
            "arrived",
            [f"to_{j}"],
            ["station"],
        )
        still_there = (
            ~(arrivals["left"] <= arrivals["at"]) & arrivals["arrived"].notna()
        )
        rows = arrivals["row"].to_numpy()
        columns[f"occupied_for_{j}"][rows] = np.where(
            still_there, arrivals["at"] - arrivals["arrived"], 0.0
        )

        departures = latest(
            visits[keys + ["station", "left", "dwell"]],
            "left",
            [f"to_{j}"],
            ["station"],
        )
        rows = departures["row"].to_numpy()
        columns[f"last_dwell_{j}"][rows] = departures["dwell"].to_numpy()

        for name, values in columns.items():
            df[name] = values
